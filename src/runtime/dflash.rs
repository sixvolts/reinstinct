//! DFlash block-diffusion drafter — GPU runtime.
//!
//! See `docs/DFLASH_PORT.md` for the spec and where each claim comes
//! from. The short version of what makes this unlike every other forward
//! pass in the engine:
//!
//! * **One pass drafts 16 tokens.** Block position 0 holds an anchor
//!   token already known; 1..15 are `MASK` and are denoised together. The
//!   hidden at position `j` predicts the token **at** `j`, not `j+1`.
//! * **Target features are extra KV entries, not inputs.** Six tapped
//!   target layers are fused by `fc` into one context feature per
//!   position, and every draft layer projects that through its own K and
//!   V to sit *in front of* the block's own keys and values. They never
//!   produce queries, never pass through the output projection, and never
//!   touch the residual or the FFN.
//! * **The last layer is unmasked.** Layers 0-3 are causal with a 2048
//!   window; layer 4 sees the whole `[ctx | block]` span in both
//!   directions, so the masked positions can condition on each other.
//! * **The body is Qwen3-shaped** — pre-norm, SwiGLU, per-head-dim q/k
//!   norms — even though the target is Gemma 4. No sandwich norms.
//!
//! Layout trick worth knowing: the context cache is sized
//! `max_seq + block_size` and the block's own K/V are written straight
//! past `ctx_len`. That makes `[ctx | block]` one contiguous run at
//! absolute positions `0..ctx_len+16`, which is exactly the flash
//! kernel's contract — no concatenation, and the block's entries are
//! implicitly discarded when the next context append overwrites them.

use std::ffi::c_void;

use crate::gguf::GgufFile;
use crate::hip::{DeviceBuf, Module, Stream};
use crate::model::dflash::{DFlashAttn, DFlashConfig, DFlashModel};
use crate::runtime::KernelCache;
use crate::runtime::gemma4::GpuGemma4;
use crate::runtime::prefill::PrefillGemm;
use crate::runtime::qwen35::GpuMatvecTensor;

const RMSNORM_SRC:    &str = include_str!("../../kernels/rmsnorm.cpp");
const RMSNORM_MH_SRC: &str = include_str!("../../kernels/rmsnorm_multihead.cpp");
const ROPE_SRC:       &str = include_str!("../../kernels/rope_batched.cpp");
const SWIGLU_SRC:     &str = include_str!("../../kernels/swiglu.cpp");
const ADD_SRC:        &str = include_str!("../../kernels/add_inplace.cpp");
const ATTN_SRC:       &str = include_str!("../../kernels/attn_prefill_flash.cpp");

/// Rows of context fused per `append_context` chunk. Bounds the scratch:
/// a whole-prompt append would otherwise want `P * 5376` floats.
const CTX_CHUNK: usize = 256;

/// Per-layer draft weights. Qwen3 block: pre-norm, GQA with per-head-dim
/// q/k norms, SwiGLU MLP.
struct DFlashBlock {
    attn_norm:   DeviceBuf<f32>,
    attn_q:      GpuMatvecTensor,
    attn_k:      GpuMatvecTensor,
    attn_v:      GpuMatvecTensor,
    attn_q_norm: DeviceBuf<f32>,
    attn_k_norm: DeviceBuf<f32>,
    attn_output: GpuMatvecTensor,
    ffn_norm:    DeviceBuf<f32>,
    ffn_gate:    GpuMatvecTensor,
    ffn_up:      GpuMatvecTensor,
    ffn_down:    GpuMatvecTensor,
    kind:        DFlashAttn,
}

impl DFlashBlock {
    fn from_gguf(gguf: &GgufFile, layer: u32, kind: DFlashAttn) -> Result<Self, String> {
        let p = format!("blk.{layer}.");
        Ok(Self {
            attn_norm:   load_fp32(gguf, &format!("{p}attn_norm.weight"))?,
            attn_q:      GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}attn_q.weight"))?,
            attn_k:      GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}attn_k.weight"))?,
            attn_v:      GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}attn_v.weight"))?,
            attn_q_norm: load_fp32(gguf, &format!("{p}attn_q_norm.weight"))?,
            attn_k_norm: load_fp32(gguf, &format!("{p}attn_k_norm.weight"))?,
            attn_output: GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}attn_output.weight"))?,
            ffn_norm:    load_fp32(gguf, &format!("{p}ffn_norm.weight"))?,
            ffn_gate:    GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}ffn_gate.weight"))?,
            ffn_up:      GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}ffn_up.weight"))?,
            ffn_down:    GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}ffn_down.weight"))?,
            kind,
        })
    }
}

/// Per-sequence draft state: the context K/V the drafter conditions on.
///
/// Sized `max_seq + block_size` per layer so a block's transient K/V can
/// live immediately after the context without a separate buffer.
pub struct DFlashState {
    ctx_k: Vec<DeviceBuf<f32>>,
    ctx_v: Vec<DeviceBuf<f32>>,
    /// Context rows written so far — equals the number of accepted tokens.
    pub ctx_len: usize,
}

impl DFlashState {
    pub fn new(cfg: &DFlashConfig, max_seq: usize) -> Result<Self, String> {
        let rows = max_seq + cfg.block_size as usize;
        let width = (cfg.n_kv_heads * cfg.head_dim) as usize;
        let mut ctx_k = Vec::with_capacity(cfg.block_count as usize);
        let mut ctx_v = Vec::with_capacity(cfg.block_count as usize);
        for _ in 0..cfg.block_count {
            ctx_k.push(DeviceBuf::new(rows * width)?);
            ctx_v.push(DeviceBuf::new(rows * width)?);
        }
        Ok(Self { ctx_k, ctx_v, ctx_len: 0 })
    }

    pub fn reset(&mut self) { self.ctx_len = 0; }
}

pub struct GpuDFlash {
    pub config: DFlashConfig,
    blocks: Vec<DFlashBlock>,
    /// Cross-layer fusion `Wc`: [n_taps * hidden → hidden].
    fc: GpuMatvecTensor,
    /// RMSNorm applied to `fc`'s output (`enc.output_norm`).
    enc_norm: DeviceBuf<f32>,
    /// Final norm before the target's LM head.
    out_norm: DeviceBuf<f32>,

    gemm: PrefillGemm,
    stream: Stream,

    m_rmsnorm: Module,
    m_rmsnorm_mh: Module,
    m_rope: Module,
    m_swiglu: Module,
    m_add: Module,
    m_attn: Module,

    rope_cos: DeviceBuf<f32>,
    rope_sin: DeviceBuf<f32>,

    // --- context-append scratch, CTX_CHUNK rows ---
    ctx_fused: DeviceBuf<f32>,   // [CTX_CHUNK, hidden]
    ctx_norm:  DeviceBuf<f32>,   // [CTX_CHUNK, hidden]

    // --- block scratch, block_size rows ---
    blk_x:     DeviceBuf<f32>,   // [B, hidden]  residual stream
    blk_norm:  DeviceBuf<f32>,   // [B, hidden]
    blk_q:     DeviceBuf<f32>,   // [B, n_heads * head_dim]
    blk_attn:  DeviceBuf<f32>,   // [B, n_heads * head_dim]
    blk_proj:  DeviceBuf<f32>,   // [B, hidden]
    blk_gate:  DeviceBuf<f32>,   // [B, ffn]
    blk_up:    DeviceBuf<f32>,   // [B, ffn]
    blk_logits: DeviceBuf<f32>,  // [B, vocab]
    blk_tokens: DeviceBuf<u32>,  // [B] token ids for the embed lookup
}

impl GpuDFlash {
    pub fn new(model: &DFlashModel, gguf: &GgufFile, cache: &KernelCache,
               target: &GpuGemma4, max_seq: usize) -> Result<Self, String>
    {
        let cfg = model.config.clone();
        cfg.validate_against_target(target.block_count() as u32)
            .map_err(|e| e.to_string())?;
        if cfg.hidden_size as usize != target.hidden_size() {
            return Err(format!(
                "dflash hidden {} != target hidden {} — the context features are \
                 projected into the draft width, so they must match",
                cfg.hidden_size, target.hidden_size()));
        }

        let mut blocks = Vec::with_capacity(cfg.block_count as usize);
        for l in 0..cfg.block_count {
            blocks.push(DFlashBlock::from_gguf(gguf, l, cfg.attn_kinds[l as usize])?);
        }

        let h = cfg.hidden_size as usize;
        let b = cfg.block_size as usize;
        let q_dim = (cfg.n_heads * cfg.head_dim) as usize;
        let ff = cfg.ffn_size as usize;
        let rot = cfg.head_dim as usize;

        // RoPE tables over the full head_dim at the draft's own theta.
        let rc = crate::cpu::rope::RopeCache::new(rot, max_seq + b, cfg.rope_freq_base);
        let mut cos = vec![0.0f32; (max_seq + b) * rot];
        let mut sin = vec![0.0f32; (max_seq + b) * rot];
        for pos in 0..(max_seq + b) {
            let (c, s) = rc.get(pos);
            cos[pos * rot..(pos + 1) * rot].copy_from_slice(c);
            sin[pos * rot..(pos + 1) * rot].copy_from_slice(s);
        }

        let ld = |name: &str, src: &str| -> Result<Module, String> {
            Module::load(&cache.compile(name, src)?)
        };
        // Widest weight / activation the prefill GEMM will see.
        let max_w = (cfg.fused_ctx_width() as usize * h).max(ff * h);
        let gemm = PrefillGemm::new(cache, max_w, CTX_CHUNK * cfg.fused_ctx_width() as usize,
                                    CTX_CHUNK * h)?;

        Ok(Self {
            blocks,
            fc:       GpuMatvecTensor::from_gguf_matvec(gguf, "fc.weight")?,
            enc_norm: load_fp32(gguf, "enc.output_norm.weight")?,
            out_norm: load_fp32(gguf, "output_norm.weight")?,
            gemm,
            stream: Stream::new()?,
            m_rmsnorm:    ld("rmsnorm", RMSNORM_SRC)?,
            m_rmsnorm_mh: ld("rmsnorm_multihead", RMSNORM_MH_SRC)?,
            m_rope:       ld("rope_batched", ROPE_SRC)?,
            m_swiglu:     ld("swiglu", SWIGLU_SRC)?,
            m_add:        ld("add_inplace", ADD_SRC)?,
            m_attn:       ld("attn_prefill_flash", ATTN_SRC)?,
            rope_cos: DeviceBuf::from_slice(&cos)?,
            rope_sin: DeviceBuf::from_slice(&sin)?,
            ctx_fused: DeviceBuf::new(CTX_CHUNK * h)?,
            ctx_norm:  DeviceBuf::new(CTX_CHUNK * h)?,
            blk_x:     DeviceBuf::new(b * h)?,
            blk_norm:  DeviceBuf::new(b * h)?,
            blk_q:     DeviceBuf::new(b * q_dim)?,
            blk_attn:  DeviceBuf::new(b * q_dim)?,
            blk_proj:  DeviceBuf::new(b * h)?,
            blk_gate:  DeviceBuf::new(b * ff)?,
            blk_up:    DeviceBuf::new(b * ff)?,
            blk_logits: DeviceBuf::new(b * target.vocab_size())?,
            blk_tokens: DeviceBuf::new(b)?,
            config: cfg,
        })
    }

    /// Block indices this drafter taps from the target — pass straight to
    /// `GpuGemma4::enable_target_tap`.
    pub fn target_layers(&self) -> &[u32] { &self.config.target_layers }

    /// Fuse `n_new` rows of tapped target features starting at absolute
    /// position `base_pos`, and project them into every draft layer's
    /// context K/V.
    ///
    /// `tap` is the target's tap buffer, `[rows, n_taps * hidden]`; rows
    /// `0..n_new` are consumed. Call after the target's prefill (with all
    /// prompt rows) and after each verify (with the rows it produced).
    pub fn append_context(&self, state: &mut DFlashState, tap: &DeviceBuf<f32>,
                          n_new: usize, base_pos: usize) -> Result<(), String>
    {
        let h = self.config.hidden_size as usize;
        let ctx_w = self.config.fused_ctx_width() as usize;
        let kv_dim = (self.config.n_kv_heads * self.config.head_dim) as usize;

        for off in (0..n_new).step_by(CTX_CHUNK) {
            let rows = CTX_CHUNK.min(n_new - off);
            let src = unsafe { (tap.raw_ptr() as *mut u8).add(off * ctx_w * 4) as *mut c_void };

            // Ht = RMSNorm_enc(Wc · [H(l0); …; H(l5)])
            self.gemm.matmul_into_raw(&self.stream, self.ctx_fused.raw_ptr(), &self.fc.data,
                                      self.fc.dtype, self.fc.repacked, ctx_w, h,
                                      src, rows)?;
            self.rmsnorm(self.ctx_fused.raw_ptr(), self.enc_norm.raw_ptr(),
                         self.ctx_norm.raw_ptr(), h as u32, rows as u32)?;

            // Per layer: project into K/V, norm the heads, rotate, store.
            for (li, blk) in self.blocks.iter().enumerate() {
                let pos = base_pos + off;
                let k_dst = row_ptr(&state.ctx_k[li], pos, kv_dim);
                let v_dst = row_ptr(&state.ctx_v[li], pos, kv_dim);
                self.project_kv(blk, self.ctx_norm.raw_ptr(), rows, k_dst, v_dst, pos)?;
            }
        }
        state.ctx_len = base_pos + n_new;
        self.stream.synchronize()
    }

    /// Draft a block in one pass. `anchor` is the already-known token at
    /// absolute position `state.ctx_len`, which occupies block position 0;
    /// positions 1.. are `MASK` and are denoised.
    ///
    /// `block` is how many positions to denoise, `2..=block_size`. The
    /// checkpoint's `block_size` is a *training* width, not a structural
    /// one: nothing in the weights is tied to it (RoPE is absolute, the
    /// masks are computed, and no tensor is block-shaped), and the paper
    /// states models trained at larger blocks generalise to smaller
    /// inference-time blocks — offered precisely so block size can be cut
    /// "under compute-bound settings", which is what an MI50 is at K=16.
    /// The reference implementation likewise takes block_size per call.
    ///
    /// Returns **all** `block` predictions. Position 0 is not a draft
    /// — the anchor was visible there — but it is a useful self-check:
    /// a correctly wired drafter should reproduce the anchor almost
    /// always, so a mismatch points at the forward pass rather than at
    /// weak draft quality. Callers take `[1..]` as the actual proposal.
    pub fn draft_block(&self, state: &DFlashState, target: &GpuGemma4,
                       anchor: u32, mask_token: u32, block: usize)
        -> Result<Vec<u32>, String>
    {
        let cfg = &self.config;
        let h = cfg.hidden_size as usize;
        let b = block.clamp(2, cfg.block_size as usize);
        let q_dim = (cfg.n_heads * cfg.head_dim) as usize;
        let kv_dim = (cfg.n_kv_heads * cfg.head_dim) as usize;
        let ff = cfg.ffn_size as usize;
        let start = state.ctx_len;

        // [anchor, MASK × (b-1)], embedded from the TARGET's table with no
        // √hidden scale — the reference takes the raw embedding weight.
        // blk_tokens is sized for the checkpoint's full block; the upload
        // has to match that length even when denoising fewer positions, so
        // pad with MASK and let the embed read only the first `b`.
        let mut ids = vec![mask_token; self.config.block_size as usize];
        ids[0] = anchor;
        self.blk_tokens.copy_from_host(&ids)?;
        target.launch_embed_batched(target.vocab_head(), self.blk_x.raw_ptr(),
                                    self.blk_tokens.raw_ptr(), b as u32)?;
        target.stream().synchronize()?;

        for (li, blk) in self.blocks.iter().enumerate() {
            // --- attention ---
            self.rmsnorm(self.blk_x.raw_ptr(), blk.attn_norm.raw_ptr(),
                         self.blk_norm.raw_ptr(), h as u32, b as u32)?;
            self.gemm.matmul_into_raw(&self.stream, self.blk_q.raw_ptr(), &blk.attn_q.data,
                                      blk.attn_q.dtype, blk.attn_q.repacked, h, q_dim,
                                      self.blk_norm.raw_ptr(), b)?;
            self.rmsnorm_mh(self.blk_q.raw_ptr(), blk.attn_q_norm.raw_ptr(),
                            self.blk_q.raw_ptr(), cfg.n_heads, cfg.head_dim, b as u32)?;
            self.rope(self.blk_q.raw_ptr(), cfg.n_heads, b as u32, start as u32)?;

            // The block's own K/V go immediately after the context, so
            // `[ctx | block]` is one contiguous cache run.
            let k_dst = row_ptr(&state.ctx_k[li], start, kv_dim);
            let v_dst = row_ptr(&state.ctx_v[li], start, kv_dim);
            self.project_kv(blk, self.blk_norm.raw_ptr(), b, k_dst, v_dst, start)?;

            let (kname, window) = match blk.kind {
                DFlashAttn::SlidingCausal => ("attn_prefill_flash_f32", cfg.sliding_window),
                DFlashAttn::FullBidirectional => ("attn_prefill_flash_nc_f32", 0),
            };
            self.attn(kname, self.blk_q.raw_ptr(),
                      state.ctx_k[li].raw_ptr(), state.ctx_v[li].raw_ptr(),
                      self.blk_attn.raw_ptr(), window, b as u32, start as u32)?;

            self.gemm.matmul_into(&self.stream, &self.blk_proj, &blk.attn_output.data,
                                  blk.attn_output.dtype, blk.attn_output.repacked,
                                  q_dim, h, &self.blk_attn, b)?;
            self.add(self.blk_x.raw_ptr(), self.blk_proj.raw_ptr(), (b * h) as u32)?;

            // --- SwiGLU MLP (SiLU, not Gemma's GeGLU) ---
            self.rmsnorm(self.blk_x.raw_ptr(), blk.ffn_norm.raw_ptr(),
                         self.blk_norm.raw_ptr(), h as u32, b as u32)?;
            self.gemm.matmul_into_raw(&self.stream, self.blk_gate.raw_ptr(), &blk.ffn_gate.data,
                                      blk.ffn_gate.dtype, blk.ffn_gate.repacked, h, ff,
                                      self.blk_norm.raw_ptr(), b)?;
            self.gemm.matmul_into_raw(&self.stream, self.blk_up.raw_ptr(), &blk.ffn_up.data,
                                      blk.ffn_up.dtype, blk.ffn_up.repacked, h, ff,
                                      self.blk_norm.raw_ptr(), b)?;
            self.swiglu(self.blk_gate.raw_ptr(), self.blk_up.raw_ptr(),
                        self.blk_gate.raw_ptr(), (b * ff) as u32)?;
            self.gemm.matmul_into_raw(&self.stream, self.blk_proj.raw_ptr(), &blk.ffn_down.data,
                                      blk.ffn_down.dtype, blk.ffn_down.repacked, ff, h,
                                      self.blk_gate.raw_ptr(), b)?;
            self.add(self.blk_x.raw_ptr(), self.blk_proj.raw_ptr(), (b * h) as u32)?;
        }

        // Final norm → the target's tied LM head → softcap.
        self.rmsnorm(self.blk_x.raw_ptr(), self.out_norm.raw_ptr(),
                     self.blk_norm.raw_ptr(), h as u32, b as u32)?;
        self.stream.synchronize()?;
        target.launch_lm_head_rows(self.blk_norm.raw_ptr(), self.blk_logits.raw_ptr(), b)?;
        if target.logit_softcap() > 0.0 {
            target.launch_softcap(self.blk_logits.raw_ptr(), (b * target.vocab_size()) as u32)?;
        }
        target.stream().synchronize()?;

        // The hidden at block position j predicts the token AT j, so
        // positions 1..b-1 are the draft; position 0 is the anchor we
        // already had.
        // Same story as blk_tokens: the readback must cover the whole
        // buffer, so size it by the checkpoint's block and read the first
        // `b` rows out of it.
        let vocab = target.vocab_size();
        let mut logits = vec![0.0f32; self.config.block_size as usize * vocab];
        self.blk_logits.copy_to_host(&mut logits)?;
        let mut out = Vec::with_capacity(b);
        for j in 0..b {
            let row = &logits[j * vocab..(j + 1) * vocab];
            let mut best = 0usize;
            for (i, v) in row.iter().enumerate() {
                if *v > row[best] { best = i; }
            }
            let _ = &mut best;
            out.push(best as u32);
        }
        Ok(out)
    }

    // ----- kernel launches -----

    /// K/V projection shared by the context path and the block path: the
    /// only difference between them is where the rows land and at what
    /// position they rotate.
    fn project_kv(&self, blk: &DFlashBlock, src: *mut c_void, rows: usize,
                  k_dst: *mut c_void, v_dst: *mut c_void, base_pos: usize)
        -> Result<(), String>
    {
        let cfg = &self.config;
        let h = cfg.hidden_size as usize;
        let kv_dim = (cfg.n_kv_heads * cfg.head_dim) as usize;
        self.gemm.matmul_into_raw(&self.stream, k_dst, &blk.attn_k.data,
                                  blk.attn_k.dtype, blk.attn_k.repacked,
                                  h, kv_dim, src, rows)?;
        self.gemm.matmul_into_raw(&self.stream, v_dst, &blk.attn_v.data,
                                  blk.attn_v.dtype, blk.attn_v.repacked,
                                  h, kv_dim, src, rows)?;
        // k_norm and RoPE apply to keys — including the context rows.
        // V is left alone.
        self.rmsnorm_mh(k_dst, blk.attn_k_norm.raw_ptr(), k_dst,
                        cfg.n_kv_heads, cfg.head_dim, rows as u32)?;
        self.rope(k_dst, cfg.n_kv_heads, rows as u32, base_pos as u32)?;
        Ok(())
    }

    fn rmsnorm(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void, n: u32, p: u32)
        -> Result<(), String>
    {
        let f = self.m_rmsnorm.function("rmsnorm_f32")?;
        let block: u32 = 256;
        let mut xa=x; let mut wa=w; let mut ya=y; let mut na=n;
        let mut ea=self.config.rms_norm_eps;
        let mut args: [*mut c_void; 5] = [
            &mut xa as *mut _ as *mut c_void, &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void];
        unsafe { f.launch((1,p,1),(block,1,1), block*4, Some(&self.stream), &mut args) }
    }

    fn rmsnorm_mh(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void,
                  n_heads: u32, head_dim: u32, p: u32) -> Result<(), String>
    {
        let f = self.m_rmsnorm_mh.function("rmsnorm_multihead_f32")?;
        let block: u32 = 256;
        let mut xa=x; let mut wa=w; let mut ya=y;
        let mut nh=n_heads; let mut hd=head_dim; let mut ea=self.config.rms_norm_eps;
        let mut args: [*mut c_void; 6] = [
            &mut xa as *mut _ as *mut c_void, &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ea as *mut _ as *mut c_void];
        unsafe { f.launch((n_heads,p,1),(block,1,1), block*4, Some(&self.stream), &mut args) }
    }

    fn rope(&self, x: *mut c_void, n_heads: u32, rows: u32, base_pos: u32)
        -> Result<(), String>
    {
        let f = self.m_rope.function("rope_apply_batched_f32")?;
        let rot = self.config.head_dim;
        let block: u32 = 64;
        let mut xa=x; let mut ca=self.rope_cos.raw_ptr(); let mut sa=self.rope_sin.raw_ptr();
        let mut hd=rot; let mut rd=rot; let mut nh=n_heads; let mut bp=base_pos;
        let mut args: [*mut c_void; 7] = [
            &mut xa as *mut _ as *mut c_void, &mut ca as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut rd as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void];
        let half = rot / 2;
        unsafe { f.launch(((half + block - 1)/block, n_heads, rows), (block,1,1),
                          0, Some(&self.stream), &mut args) }
    }

    fn swiglu(&self, gate: *mut c_void, up: *mut c_void, out: *mut c_void, n: u32)
        -> Result<(), String>
    {
        let f = self.m_swiglu.function("swiglu_mul_f32")?;
        let block: u32 = 256;
        let mut ga=gate; let mut ua=up; let mut oa=out; let mut na=n;
        let mut args: [*mut c_void; 4] = [
            &mut ga as *mut _ as *mut c_void, &mut ua as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void];
        unsafe { f.launch(((n + block - 1)/block,1,1),(block,1,1), 0,
                          Some(&self.stream), &mut args) }
    }

    fn add(&self, x: *mut c_void, y: *mut c_void, n: u32) -> Result<(), String> {
        let f = self.m_add.function("add_inplace_f32")?;
        let block: u32 = 256;
        let mut xa=x; let mut ya=y; let mut na=n;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void, &mut ya as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void];
        unsafe { f.launch(((n + block - 1)/block,1,1),(block,1,1), 0,
                          Some(&self.stream), &mut args) }
    }

    #[allow(clippy::too_many_arguments)]
    fn attn(&self, kname: &str, q: *mut c_void, k: *mut c_void, v: *mut c_void,
            out: *mut c_void, window: u32, rows: u32, base_pos: u32)
        -> Result<(), String>
    {
        const BQ: u32 = 8;
        const BK: u32 = 8;
        let cfg = &self.config;
        let f = self.m_attn.function(kname)?;
        let block = 64 * BQ;
        let smem = 2 * BK * cfg.head_dim * 4;
        let mut qa=q; let mut ka=k; let mut va=v; let mut oa=out;
        let mut nh=cfg.n_heads; let mut nkv=cfg.n_kv_heads; let mut hd=cfg.head_dim;
        let mut wn=window;
        let mut sc = 1.0f32 / (cfg.head_dim as f32).sqrt();
        let mut pr=rows; let mut bp=base_pos;
        let mut args: [*mut c_void; 11] = [
            &mut qa as *mut _ as *mut c_void, &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut wn as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void, &mut pr as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void];
        unsafe { f.launch((cfg.n_heads, (rows + BQ - 1)/BQ, 1), (block,1,1),
                          smem, Some(&self.stream), &mut args) }
    }
}

fn row_ptr(buf: &DeviceBuf<f32>, row: usize, width: usize) -> *mut c_void {
    unsafe { (buf.raw_ptr() as *mut u8).add(row * width * 4) as *mut c_void }
}

fn load_fp32(gguf: &GgufFile, name: &str) -> Result<DeviceBuf<f32>, String> {
    let info = gguf.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
    if info.ggml_type != crate::gguf::GgmlType::F32 {
        return Err(format!("tensor {name}: expected F32, got {:?}", info.ggml_type));
    }
    let bytes = gguf.tensor_data(name).map_err(|e| format!("{name}: {e}"))?
        .ok_or_else(|| format!("{name}: no data"))?;
    DeviceBuf::from_slice(bytemuck::cast_slice::<u8, f32>(bytes))
}
