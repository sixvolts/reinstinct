//! Pipeline parallelism for the dense Qwen 3.5-family runtime.
//!
//! A model too big for one card is split by *layers*: stage `s` owns a
//! contiguous block range on its own device, built as an ordinary
//! [`GpuQwen35`] via `new_stage`. A forward walks the stages in order,
//! each one peer-copying the running hidden activation from the one
//! before it (`hidden` floats per token — 20 KB on a 27B, ~7 µs over
//! PCIe 4.0 x8) and running its blocks; the first stage embeds, the last
//! runs the output head. Stages are strictly sequential for a single
//! sequence, so this buys capacity, not speed: decode is still one
//! weight pass per token, now spread across cards, and the handoffs add
//! a few tens of microseconds per stage.
//!
//! Every HIP handle in a stage (buffers, modules, stream, graph) belongs
//! to that stage's device, and HIP resolves streams against the
//! *current* device, so the driver sets the device before every call
//! into a stage. Stage boundaries are the only cross-device edges: an
//! event recorded on the producer's stream, waited on by the consumer's,
//! then `hipMemcpyPeerAsync` on the consumer's stream.

use std::ops::Range;

use crate::gguf::GgufFile;
use crate::hip::{self, GraphExec};
use crate::model::qwen3_5::Qwen35Model;
use crate::runtime::KernelCache;
use crate::runtime::qwen35::{GpuQwen35, Qwen35GpuState, StageOutput};

struct Stage {
    dev: i32,
    gpu: GpuQwen35,
}

/// One `GpuQwen35` per device, driven in layer order.
pub struct Qwen35Pipeline {
    stages: Vec<Stage>,
}

/// The per-stage decode states of a pipeline, one `Qwen35GpuState` per
/// stage holding that stage's KV caches / GDN states.
pub struct Qwen35PipelineState {
    stages: Vec<(i32, Qwen35GpuState)>,
    pub pos: usize,
}

/// A captured decode graph per stage (`Qwen35Pipeline::capture_forward_graph`).
pub struct PipelineGraph {
    execs: Vec<GraphExec>,
}

fn set_dev(dev: i32) -> Result<(), String> {
    hip::Device::set(dev).map(|_| ())
}

impl Qwen35Pipeline {
    /// Bytes a tensor occupies on device: its GGUF size, except BF16,
    /// which the loader requantizes to Q8_0 (34 bytes per 32 weights).
    fn device_bytes(t: &crate::gguf::TensorInfo) -> u64 {
        match t.ggml_type {
            crate::gguf::GgmlType::BF16 => t.n_elements() * 34 / 32,
            _ => t.byte_size().unwrap_or(0),
        }
    }

    /// Bytes of weight data behind main block `i`.
    fn block_bytes(gguf: &GgufFile, i: usize) -> u64 {
        let prefix = format!("blk.{i}.");
        gguf.tensors.iter()
            .filter(|t| t.name.starts_with(&prefix))
            .map(Self::device_bytes)
            .sum()
    }

    /// Bytes of everything that is not a main block: embedding, output
    /// head, MTP heads — the fixed cost the first / last stages carry.
    fn head_bytes(gguf: &GgufFile, model: &Qwen35Model) -> (u64, u64) {
        let sz = |name: &str| gguf.tensor(name).map(Self::device_bytes).unwrap_or(0);
        let embd = sz("token_embd.weight");
        let mut tail = sz("output_norm.weight") + sz("output.weight");
        for (i, _) in model.mtp_block_kinds() {
            tail += Self::block_bytes(gguf, i as usize);
        }
        if model.config.tied_embeddings { tail += embd; }
        (embd, tail)
    }

    /// Split the main blocks into `n_stages` contiguous ranges of roughly
    /// equal weight bytes, charging the embedding to the first stage and
    /// the output head (plus MTP heads) to the last. Weight bytes are
    /// what decode reads, so equal bytes is equal time per stage.
    pub fn plan_split(gguf: &GgufFile, model: &Qwen35Model, n_stages: usize)
        -> Vec<Range<usize>>
    {
        let n_blocks = model.block_kinds.len();
        assert!(n_stages >= 1 && n_stages <= n_blocks,
                "plan_split: {n_stages} stages for {n_blocks} blocks");
        let per_block: Vec<u64> = (0..n_blocks).map(|i| Self::block_bytes(gguf, i)).collect();
        let (embd, tail) = Self::head_bytes(gguf, model);
        let total = per_block.iter().sum::<u64>() + embd + tail;

        let mut ranges = Vec::with_capacity(n_stages);
        let mut start = 0usize;
        let mut cum = embd;
        for s in 0..n_stages {
            if s == n_stages - 1 {
                ranges.push(start..n_blocks);
                break;
            }
            // Stage boundary at the first block whose inclusion crosses
            // this stage's share, leaving at least one block per
            // remaining stage.
            let target = total * (s as u64 + 1) / n_stages as u64;
            let max_end = n_blocks - (n_stages - 1 - s);
            let mut end = start + 1;
            cum += per_block[start];
            while end < max_end && cum + per_block[end] / 2 < target {
                cum += per_block[end];
                end += 1;
            }
            ranges.push(start..end);
            start = end;
        }
        ranges
    }

    /// Build a pipeline over `devices` (one stage each, in order). `split`
    /// gives explicit block counts per stage; None plans by weight bytes.
    /// Peer access is enabled between consecutive devices where the
    /// platform allows it (the copies fall back to a host bounce
    /// otherwise). A single device is the ordinary single-GPU engine.
    pub fn new(model: &Qwen35Model, gguf: &GgufFile, cache: &KernelCache, max_seq: usize,
               devices: &[i32], split: Option<&[usize]>) -> Result<Self, String>
    {
        assert!(!devices.is_empty(), "pipeline needs at least one device");
        let n_blocks = model.block_kinds.len();
        let ranges: Vec<Range<usize>> = match split {
            Some(counts) => {
                if counts.len() != devices.len() {
                    return Err(format!("--split has {} entries for {} devices",
                                       counts.len(), devices.len()));
                }
                if counts.iter().sum::<usize>() != n_blocks || counts.iter().any(|&c| c == 0) {
                    return Err(format!("--split {counts:?} must be non-zero and sum to {n_blocks} blocks"));
                }
                let mut start = 0;
                counts.iter().map(|&c| { let r = start..start + c; start += c; r }).collect()
            }
            None => Self::plan_split(gguf, model, devices.len()),
        };

        let n_dev = hip::device_count()?;
        for &d in devices {
            if d < 0 || d >= n_dev {
                return Err(format!("device {d} out of range (HIP sees {n_dev})"));
            }
        }
        // Peer access between neighbours, both directions (decode hands
        // forward; nothing hands back today, but enabling is symmetric
        // and cheap).
        for w in devices.windows(2) {
            let (a, b) = (w[0], w[1]);
            if a == b { continue; }
            for (from, to) in [(a, b), (b, a)] {
                if hip::Device::can_access_peer(from, to)? {
                    set_dev(from)?;
                    hip::Device::enable_peer_access(to)?;
                }
            }
        }

        let mut stages = Vec::with_capacity(devices.len());
        for (&dev, range) in devices.iter().zip(&ranges) {
            set_dev(dev)?;
            let gpu = GpuQwen35::new_stage(model, gguf, cache, max_seq, range.clone())?;
            stages.push(Stage { dev, gpu });
        }
        Ok(Self { stages })
    }

    pub fn n_stages(&self) -> usize { self.stages.len() }

    /// `(device, block range)` per stage.
    pub fn layout(&self) -> Vec<(i32, Range<usize>)> {
        self.stages.iter().map(|s| {
            let st = s.gpu.stage();
            (s.dev, st.first_block..st.end_block)
        }).collect()
    }

    /// Whole-model engine accessor for a single-stage pipeline (the
    /// spec-decode paths still drive `GpuQwen35` directly).
    pub fn single(&self) -> Option<&GpuQwen35> {
        if self.stages.len() == 1 { Some(&self.stages[0].gpu) } else { None }
    }

    pub fn new_state(&self, model: &Qwen35Model, max_seq: usize)
        -> Result<Qwen35PipelineState, String>
    {
        let mut stages = Vec::with_capacity(self.stages.len());
        for s in &self.stages {
            set_dev(s.dev)?;
            let st = s.gpu.stage();
            stages.push((s.dev, Qwen35GpuState::new_stage(model, max_seq,
                                                          st.first_block..st.end_block)?));
        }
        Ok(Qwen35PipelineState { stages, pos: 0 })
    }

    fn check_state(&self, state: &Qwen35PipelineState) {
        assert_eq!(state.stages.len(), self.stages.len(), "pipeline state / stage count mismatch");
    }

    /// The micro-batch (chunk) sizes a prompt of `n` tokens is prefilled
    /// in on a multi-stage pipeline, in tokens: at least four equal
    /// chunks, none above `REINSTINCT_PREFILL_CHUNK` tokens (default
    /// 256), all 64-multiples (the MMQ token tile) but the last.
    ///
    /// With S stages and M chunks the cards are busy M/(M+S-1) of the
    /// pass, but the fill/drain chunks cannot be made cheap by making
    /// them small: below ~128 tokens the MMQ grid (`out_dim/64` x 1
    /// workgroups) no longer fills 60 CUs and a chunk costs nearly as
    /// much as one twice its size — measured on the 27B at pp644, 64-token
    /// end chunks came out slightly *slower* than four equal 192s. So:
    /// equal chunks, and more of them only as the prompt grows.
    fn prefill_chunks(&self, n: usize) -> Vec<usize> {
        if self.stages.len() == 1 || n < 2 * 64 { return vec![n]; }
        let max_chunk = std::env::var("REINSTINCT_PREFILL_CHUNK").ok()
            .and_then(|v| v.parse::<usize>().ok()).filter(|&c| c >= 64).unwrap_or(256)
            .div_ceil(64) * 64;
        let chunk = n.div_ceil(4).div_ceil(64) * 64;
        let chunk = chunk.clamp(64, max_chunk);
        let mut chunks = Vec::with_capacity(n.div_ceil(chunk));
        let mut left = n;
        while left > 0 { let c = chunk.min(left); chunks.push(c); left -= c; }
        chunks
    }

    /// Batched prefill of `tokens`; returns the logits at the last position.
    ///
    /// On a multi-stage pipeline the prompt goes through in micro-batches
    /// (GPipe-style, sizes from `prefill_chunks`): stage 0 starts chunk
    /// `k+1` as soon as it has handed chunk `k` on, so the stages work
    /// concurrently instead of one card idling while the other runs. Nothing on the host waits between
    /// chunks — the cross-stage events order the copies, and each
    /// stage's single stream orders its chunks — so the overlap falls out
    /// of stream semantics. Later chunks attend to earlier ones through
    /// the KV caches / GDN states the same way a mid-sequence prefill
    /// does.
    pub fn forward_tokens_batched(&self, tokens: &[u32], state: &mut Qwen35PipelineState)
        -> Result<Vec<f32>, String>
    {
        self.check_state(state);
        assert!(!tokens.is_empty(), "forward_tokens_batched needs ≥1 token");
        let sizes = self.prefill_chunks(tokens.len());
        let mut chunks: Vec<&[u32]> = Vec::with_capacity(sizes.len());
        let mut at = 0;
        for c in sizes { chunks.push(&tokens[at..at + c]); at += c; }
        // Every stage-0 activation must outlive stage 1's copy out of
        // it, and a chunk's buffers must not be recycled under the copy
        // by the next chunk on the same stage: hold them all until the
        // pass is done (n × hidden floats in total).
        let mut held = Vec::with_capacity(chunks.len() * self.stages.len());
        let mut logits = None;
        for (ci, ch) in chunks.iter().enumerate() {
            let last_chunk = ci + 1 == chunks.len();
            let mut prev: Option<StageOutput> = None;
            for (stage, (dev, st)) in self.stages.iter().zip(state.stages.iter_mut()) {
                set_dev(*dev)?;
                let (act, out, lg) = stage.gpu.prefill_stage(ch, st, prev.as_ref(), last_chunk)?;
                held.push(act);
                prev = Some(out);
                logits = lg;
            }
        }
        state.pos += tokens.len();
        drop(held);
        logits.ok_or_else(|| "pipeline: last stage returned no logits".to_string())
    }

    /// Decode one token per kernel launch (no graph).
    pub fn forward_token(&self, token: u32, state: &mut Qwen35PipelineState)
        -> Result<Vec<f32>, String>
    {
        self.decode(token, state, None)
    }

    /// Sequential single-token forwards; logits at the last position.
    pub fn forward_tokens(&self, tokens: &[u32], state: &mut Qwen35PipelineState)
        -> Result<Vec<f32>, String>
    {
        assert!(!tokens.is_empty(), "forward_tokens needs at least one token");
        let mut last = Vec::new();
        for &t in tokens { last = self.forward_token(t, state)?; }
        Ok(last)
    }

    /// Capture each stage's decode body into a HIP graph. As with the
    /// single-GPU graph, the capture reads only `d_pos` and persistent
    /// state buffers, so one capture serves every position.
    pub fn capture_forward_graph(&self, state: &mut Qwen35PipelineState)
        -> Result<PipelineGraph, String>
    {
        self.check_state(state);
        let mut execs = Vec::with_capacity(self.stages.len());
        for (stage, (dev, st)) in self.stages.iter().zip(state.stages.iter_mut()) {
            set_dev(*dev)?;
            execs.push(stage.gpu.capture_forward_graph(st)?);
        }
        Ok(PipelineGraph { execs })
    }

    /// Decode one token by replaying the per-stage graphs.
    pub fn forward_token_via_graph(&self, graph: &PipelineGraph, token: u32,
                                   state: &mut Qwen35PipelineState)
        -> Result<Vec<f32>, String>
    {
        assert_eq!(graph.execs.len(), self.stages.len(), "graph / stage count mismatch");
        self.decode(token, state, Some(graph))
    }

    fn decode(&self, token: u32, state: &mut Qwen35PipelineState, graph: Option<&PipelineGraph>)
        -> Result<Vec<f32>, String>
    {
        self.decode_launch(token, state, graph)?;
        self.decode_finish()
    }

    /// Enqueue one decode step on every stage and return without
    /// waiting; `decode_finish` collects the logits. Between the two the
    /// host is free — a server uses the gap for sampling bookkeeping,
    /// text decode and streaming, which otherwise sit serialized with
    /// the GPU.
    pub fn decode_launch(&self, token: u32, state: &mut Qwen35PipelineState,
                         graph: Option<&PipelineGraph>) -> Result<(), String>
    {
        self.check_state(state);
        if let Some(g) = graph {
            assert_eq!(g.execs.len(), self.stages.len(), "graph / stage count mismatch");
        }
        let mut prev: Option<StageOutput> = None;
        for (i, (stage, (dev, st))) in self.stages.iter().zip(state.stages.iter_mut()).enumerate() {
            set_dev(*dev)?;
            let g = graph.map(|g| &g.execs[i]);
            let (out, _) = stage.gpu.decode_stage(token, st, g, prev.as_ref(), false)?;
            prev = Some(out);
        }
        state.pos += 1;
        Ok(())
    }

    /// Wait for the step `decode_launch` enqueued and return its logits.
    pub fn decode_finish(&self) -> Result<Vec<f32>, String> {
        let last = self.stages.last().expect("pipeline has stages");
        set_dev(last.dev)?;
        last.gpu.read_logits()
    }

    /// `REINSTINCT_MOE_PROFILE` buckets, concatenated across stages.
    pub fn moe_prof_report(&self) -> Vec<(&'static str, f64)> {
        self.stages.iter().flat_map(|s| s.gpu.moe_prof_report()).collect()
    }
}

impl Qwen35PipelineState {
    pub fn reset(&mut self) -> Result<(), String> {
        for (dev, st) in &mut self.stages {
            set_dev(*dev)?;
            st.reset()?;
        }
        self.pos = 0;
        Ok(())
    }
}
