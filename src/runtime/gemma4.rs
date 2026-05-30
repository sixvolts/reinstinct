//! GPU forward path for Gemma 4 (dense variant — the 31B).
//!
//! Mirrors `cpu::gemma4` on the MI50: weights stay resident in their
//! on-disk quantized form, the forward chains HIP kernels on one
//! stream. Reuses the matvec / rmsnorm / rope / add kernels; the
//! Gemma-specific ones (geglu, logit soft-cap, scale, windowed
//! attention) were added alongside.

use std::ffi::c_void;

use crate::gguf::{GgmlType, GgufFile};
use crate::hip::{DeviceBuf, Event, Graph, GraphExec, Module, Stream};
use crate::hip::sys::HipStreamCaptureMode;
use crate::model::gemma4::{AttnKind, Gemma4Model};
use crate::runtime::KernelCache;
use crate::runtime::qwen35::{DeviceBufPool, GpuMatvecTensor, PooledBuf};

// Reused kernel sources.
const RMSNORM_SRC:           &str = include_str!("../../kernels/rmsnorm.cpp");
const RMSNORM_ADD_SRC:       &str = include_str!("../../kernels/rmsnorm_add.cpp");
const RMSNORM_Q8_SRC:        &str = include_str!("../../kernels/rmsnorm_q8.cpp");
const RMSNORM_MULTIHEAD_SRC: &str = include_str!("../../kernels/rmsnorm_multihead.cpp");
const ROPE_SRC:              &str = include_str!("../../kernels/rope_dpos.cpp");
const ADD_INPLACE_SRC:       &str = include_str!("../../kernels/add_inplace.cpp");
const MATVEC_F32_B256_SRC:   &str = include_str!("../../kernels/matvec_f32_b256.cpp");
const MATVEC_Q4K_W_SRC:      &str = include_str!("../../kernels/matvec_q4_k_rowblock.cpp");
const MATVEC_Q5K_W_SRC:      &str = include_str!("../../kernels/matvec_q5_k_rowblock.cpp");
const MATVEC_Q6K_W_SRC:      &str = include_str!("../../kernels/matvec_q6_k_rowblock.cpp");
const QUANTIZE_Q8_SRC:       &str = include_str!("../../kernels/quantize_q8.cpp");
const MATVEC_Q4K_DP4A_SRC:   &str = include_str!("../../kernels/matvec_q4_k_dp4a.cpp");
const MATVEC_Q5K_DP4A_SRC:   &str = include_str!("../../kernels/matvec_q5_k_dp4a.cpp");
const MATVEC_Q5K_DP4A_BATCHED_SRC: &str =
    include_str!("../../kernels/matvec_q5_k_dp4a_batched.cpp");
const MATVEC_Q6K_DP4A_SRC:   &str = include_str!("../../kernels/matvec_q6_k_dp4a.cpp");
const MATVEC_Q4K_REPACKED_SRC: &str = include_str!("../../kernels/matvec_q4k_repacked.cpp");
const MATVEC_Q5K_REPACKED_SRC: &str = include_str!("../../kernels/matvec_q5k_repacked.cpp");
const MATVEC_Q6K_REPACKED_SRC: &str = include_str!("../../kernels/matvec_q6k_repacked.cpp");
/// Output rows per wavefront in the row-blocked K-quant matvecs — must
/// match `ROWS` in matvec_q{4,5,6}_k_rowblock.cpp.
const Q4K_ROWBLOCK: u32 = 2;
const MATVEC_Q8_0_W_SRC:     &str = include_str!("../../kernels/matvec_q8_0_wave64.cpp");
const MATVEC_F16_W_SRC:      &str = include_str!("../../kernels/matvec_f16_wave64.cpp");
// Gemma-specific kernel sources.
const GEGLU_SRC:             &str = include_str!("../../kernels/geglu.cpp");
const LOGIT_SOFTCAP_SRC:     &str = include_str!("../../kernels/logit_softcap.cpp");
const SCALE_INPLACE_SRC:     &str = include_str!("../../kernels/scale_inplace.cpp");
const ATTN_WINDOW_SRC:       &str = include_str!("../../kernels/attn_step_q8.cpp");
const ATTN_PARTIAL_Q8_SRC:   &str = include_str!("../../kernels/attn_partial_q8.cpp");
const ATTN_PARTIAL_SQ_SRC:   &str = include_str!("../../kernels/attn_partial_superquant.cpp");
const ATTN_PARTIAL_SQRS_SRC: &str = include_str!("../../kernels/attn_partial_superquant_rs.cpp");
const ATTN_PARTIAL_SQWP_SRC: &str = include_str!("../../kernels/attn_partial_superquant_wp.cpp");
const ROTATE_Q_RHT_SRC:      &str = include_str!("../../kernels/rotate_q_rht.cpp");
const ATTN_MERGE_SRC:        &str = include_str!("../../kernels/attn_merge.cpp");
/// Max split-K splits per KV head — bounds the partial-attention scratch.
const ATTN_MAX_SPLITS: u32 = 16;

/// Max K (drafted tokens per round) supported by `verify_forward`'s
/// preallocated scratch. Spec-decode rarely exceeds 4-8; sized small
/// to keep the resident allocation cheap.
const MAX_VERIFY_K: usize = 8;

/// Prefill MoE batch size — `prefill_forward`'s MoE branch processes up
/// to this many tokens per set of expert launches. Bounds the per-call
/// expert-intermediate scratch; `moe_logits` / `moe_ids` / `moe_weights`
/// are sized for one chunk so the batched topk + matvecs reuse them.
const MOE_PREFILL_CHUNK: usize = 256;
const KV_WRITE_SRC:          &str = include_str!("../../kernels/kv_write_q8.cpp");
const EMBED_Q5K_SRC:         &str = include_str!("../../kernels/embed_lookup_q5_k.cpp");
const EMBED_Q8_0_SRC:        &str = include_str!("../../kernels/embed_lookup_q8_0.cpp");
// MoE kernel sources.
const MATVEC_Q8_0_DP4A_SRC:  &str = include_str!("../../kernels/matvec_q8_0_dp4a.cpp");
const MATVEC_Q8_0_REPACKED_SRC: &str = include_str!("../../kernels/matvec_q8_0_repacked.cpp");
// Prefill kernel sources.
const ROPE_PREFILL_SRC:      &str = include_str!("../../kernels/rope_prefill.cpp");
const ATTN_PREFILL_SRC:      &str = include_str!("../../kernels/attn_prefill_flash.cpp");
const PERMUTE_PLE_SRC:       &str = include_str!("../../kernels/permute_ple.cpp");
const KV_QUANT_PREFILL_SRC:  &str = include_str!("../../kernels/kv_quant_prefill.cpp");
const ROPE_BATCHED_SRC:      &str = include_str!("../../kernels/rope_batched.cpp");
const ATTN_STEP_Q8_BATCHED_SRC: &str = include_str!("../../kernels/attn_step_q8_batched.cpp");
const MOE_TOPK_SRC:          &str = include_str!("../../kernels/moe_topk.cpp");
const MOE_MATVEC_Q6K_SRC:    &str = include_str!("../../kernels/moe_matvec_q6k_dp4a.cpp");
const MOE_MV_Q6K_REPACKED_SRC: &str = include_str!("../../kernels/moe_matvec_q6k_repacked.cpp");
const MOE_MATVEC_Q8_0_SRC:   &str = include_str!("../../kernels/moe_matvec_q8_0_dp4a.cpp");
const MOE_MATVEC_Q8_0_DOWN_SRC: &str = include_str!("../../kernels/moe_matvec_q8_0_down.cpp");
const MOE_GEGLU_SRC:         &str = include_str!("../../kernels/moe_geglu.cpp");
const MOE_GEGLU_Q8_SRC:      &str = include_str!("../../kernels/moe_geglu_q8.cpp");
const MOE_COMBINE_SRC:       &str = include_str!("../../kernels/moe_combine.cpp");
const MOE_EXPERT_SORT_SRC:   &str = include_str!("../../kernels/moe_expert_sort.cpp");
const MMQ_Q6K_GROUPED_SRC:   &str = include_str!("../../kernels/mmq_gemm_q6k_grouped.cpp");
const MMQ_Q8_0_GROUPED_SRC:  &str = include_str!("../../kernels/mmq_gemm_q8_0_grouped.cpp");
/// Grouped-expert GEMM token-tile width — must match `BN` in the
/// `mmq_gemm_*_grouped` kernels.
const MOE_GEMM_BN: u32 = 32;

/// Offset an f32 device pointer by `elems` elements (prefill row indexing).
fn pf_off(p: *mut c_void, elems: usize) -> *mut c_void {
    unsafe { (p as *mut f32).add(elems) as *mut c_void }
}

/// Load an fp32 GGUF tensor straight to device.
fn load_fp32(gguf: &GgufFile, name: &str) -> Result<DeviceBuf<f32>, String> {
    let info = gguf.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
    if info.ggml_type != GgmlType::F32 {
        return Err(format!("tensor {name}: expected F32, got {:?}", info.ggml_type));
    }
    let bytes = gguf.tensor_data(name).map_err(|e| format!("{name}: {e}"))?
        .ok_or_else(|| format!("{name}: no data"))?;
    let floats: &[f32] = bytemuck::cast_slice(bytes);
    DeviceBuf::from_slice(floats)
}

/// Load an F32 tensor and multiply every element by `scale` host-side
/// before uploading. Lets us fold downstream `launch_scale` calls into
/// the weight — eg the gate_inp_s RMSNorm weight gets pre-scaled by
/// 1/sqrt(hidden) so the per-layer router skips one launch.
fn load_fp32_scaled(gguf: &GgufFile, name: &str, scale: f32)
    -> Result<DeviceBuf<f32>, String>
{
    let info = gguf.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
    if info.ggml_type != GgmlType::F32 {
        return Err(format!("tensor {name}: expected F32, got {:?}", info.ggml_type));
    }
    let bytes = gguf.tensor_data(name).map_err(|e| format!("{name}: {e}"))?
        .ok_or_else(|| format!("{name}: no data"))?;
    let mut floats: Vec<f32> = bytemuck::cast_slice::<u8, f32>(bytes).to_vec();
    for v in &mut floats { *v *= scale; }
    DeviceBuf::from_slice(&floats)
}

/// Load a 2D BF16 weight, converting it to f32 on the host and wrapping
/// it as an F32 [`GpuMatvecTensor`] so the plain f32 matvec serves it.
/// E4B's `per_layer_model_proj` is the only BF16 tensor; converting to
/// f32 (vs f16) avoids overflowing f16's narrower exponent range.
fn load_bf16_as_f32_matvec(gguf: &GgufFile, name: &str) -> Result<GpuMatvecTensor, String> {
    let info = gguf.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
    if info.ggml_type != GgmlType::BF16 {
        return Err(format!("tensor {name}: expected BF16, got {:?}", info.ggml_type));
    }
    let bytes = gguf.tensor_data(name).map_err(|e| format!("{name}: {e}"))?
        .ok_or_else(|| format!("{name}: no data"))?;
    let shape = info.shape();
    if shape.len() != 2 {
        return Err(format!("tensor {name}: expected 2D, got {shape:?}"));
    }
    let src: &[u16] = bytemuck::cast_slice(bytes);
    let f32s: Vec<f32> = src.iter().map(|&b| crate::quant::half::bf16_to_f32(b)).collect();
    Ok(GpuMatvecTensor {
        data:    DeviceBuf::from_slice(bytemuck::cast_slice::<f32, u8>(&f32s))?,
        dtype:   GgmlType::F32,
        in_dim:  shape[0] as u32,
        out_dim: shape[1] as u32,
        repacked: false,
    })
}

/// Global Per-Layer-Embedding weights (E4B only). The per-layer token
/// embedding is a second, layer-indexed embedding table; it is projected
/// from the main embedding and gated into each block — see
/// `enqueue_forward` / `block_forward`.
struct PleGlobal {
    /// Q5_K embedding table, one `[n_embd_per_layer · n_layer]` row per
    /// vocab token — kept in on-disk form for the embed-lookup kernel.
    tok_embd:   GpuMatvecTensor,
    /// `[n_embd, n_embd_per_layer · n_layer]` projection of the main
    /// embedding (BF16 on disk, converted to f32).
    model_proj: GpuMatvecTensor,
    /// RMSNorm weight over each layer's `n_embd_per_layer` slice.
    proj_norm:  DeviceBuf<f32>,
}

/// Per-block Per-Layer-Embedding weights (E4B only).
struct PleBlock {
    /// `[n_embd, n_embd_per_layer]` — gates the hidden state down.
    inp_gate:  GpuMatvecTensor,
    /// `[n_embd_per_layer, n_embd]` — projects the gated PLE back up.
    proj:      GpuMatvecTensor,
    /// RMSNorm weight applied to the PLE branch output.
    post_norm: DeviceBuf<f32>,
}

/// A 3D expert-weight tensor `[in_dim, out_dim, n_expert]` resident on
/// device in its on-disk quantized form. Each expert is a contiguous
/// `[in_dim, out_dim]` matrix `bytes_per_expert` apart; the moe_matvec
/// kernel offsets into the slab by the device-resident expert id.
pub struct ExpertTensor {
    data:  DeviceBuf<u8>,
    /// Quant type — Unsloth's UD recipe varies it per layer (the 26B's
    /// last-layer gate_up is Q8_0 while the rest are Q6_K), so the
    /// moe_matvec dispatch must be per-tensor, not hard-coded.
    dtype: GgmlType,
    bytes_per_expert: usize,
    /// True when each expert slice was repacked into the contiguous
    /// `quant::q6_k::repack_for_matvec` layout (Q6_K experts only).
    repacked: bool,
}

impl ExpertTensor {
    fn from_gguf(gguf: &GgufFile, name: &str) -> Result<Self, String> {
        let info = gguf.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
        let bytes = gguf.tensor_data(name)
            .map_err(|e| format!("read {name}: {e}"))?
            .ok_or_else(|| format!("tensor {name} has no data"))?;
        let shape = info.shape();
        if shape.len() != 3 {
            return Err(format!("expert tensor {name}: expected 3D, got {shape:?}"));
        }
        let in_dim   = shape[0] as usize;
        let out_dim  = shape[1] as usize;
        let n_expert = shape[2] as usize;
        // Q6_K experts: repack each expert slice into the contiguous
        // matvec layout (same win as the dense Q6_K repack). Q8_0 experts
        // are left on-disk — that layout is already contiguous-friendly.
        if info.ggml_type == GgmlType::Q6_K {
            let bpe = bytes.len() / n_expert;
            let mut packed = Vec::new();
            for e in 0..n_expert {
                packed.extend_from_slice(&crate::quant::q6_k::repack_for_matvec(
                    &bytes[e * bpe..(e + 1) * bpe], in_dim, out_dim));
            }
            return Ok(Self {
                bytes_per_expert: packed.len() / n_expert,
                dtype: info.ggml_type,
                data: DeviceBuf::from_slice(&packed)?,
                repacked: true,
            });
        }
        Ok(Self {
            bytes_per_expert: bytes.len() / n_expert,
            dtype: info.ggml_type,
            data: DeviceBuf::from_slice(bytes)?,
            repacked: false,
        })
    }

    /// Load an expert tensor and repack every expert slice into the
    /// contiguous matvec layout — Q8_0 too (unlike `from_gguf`, which
    /// leaves Q8_0 on-disk). Used to build the grouped-GEMM down slab.
    fn from_gguf_repacked(gguf: &GgufFile, name: &str) -> Result<Self, String> {
        let info = gguf.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
        let bytes = gguf.tensor_data(name)
            .map_err(|e| format!("read {name}: {e}"))?
            .ok_or_else(|| format!("tensor {name} has no data"))?;
        let shape = info.shape();
        if shape.len() != 3 {
            return Err(format!("expert tensor {name}: expected 3D, got {shape:?}"));
        }
        let in_dim   = shape[0] as usize;
        let out_dim  = shape[1] as usize;
        let n_expert = shape[2] as usize;
        let bpe = bytes.len() / n_expert;
        let mut packed = Vec::new();
        for e in 0..n_expert {
            let slice = &bytes[e * bpe..(e + 1) * bpe];
            let rep = match info.ggml_type {
                GgmlType::Q6_K => crate::quant::q6_k::repack_for_matvec(slice, in_dim, out_dim),
                GgmlType::Q8_0 => crate::quant::q8_0::repack_for_matvec(slice, in_dim, out_dim),
                other => return Err(format!(
                    "from_gguf_repacked: unsupported expert dtype {other:?} for {name}")),
            };
            packed.extend_from_slice(&rep);
        }
        Ok(Self {
            bytes_per_expert: packed.len() / n_expert,
            dtype: info.ggml_type,
            data: DeviceBuf::from_slice(&packed)?,
            repacked: true,
        })
    }

}

/// MoE-layer weights: the routed-expert branch that runs alongside the
/// shared MLP. Present only on MoE models (the 26B-A4B).
pub struct MoeBlock {
    post_ffw_norm_1: DeviceBuf<f32>,
    pre_ffw_norm_2:  DeviceBuf<f32>,
    post_ffw_norm_2: DeviceBuf<f32>,
    /// Router projection, F32 [hidden, n_expert].
    gate_inp:    GpuMatvecTensor,
    /// Router input scale, F32 [hidden].
    gate_inp_s:  DeviceBuf<f32>,
    /// Fused gate+up experts, [hidden, 2·expert_ff, n_expert].
    gate_up_exps: ExpertTensor,
    /// Down experts, [expert_ff, hidden, n_expert].
    down_exps:    ExpertTensor,
    /// Repacked-Q8_0 copy of the down experts for the grouped-GEMM
    /// prefill path. `down_exps` stays on-disk for decode + the matvec
    /// fallback, so the verified decode path is untouched.
    down_grouped: ExpertTensor,
    /// Per-expert down-output scalar, F32 [n_expert] — device-resident
    /// so the combine kernel can index it by the device expert id.
    down_exps_s:  DeviceBuf<f32>,
}

/// Per-prefill-call scratch for the grouped-expert GEMM MoE path
/// (`REINSTINCT_MOE_GROUPED`). Mirrors qwen35's `MoeRuntime` sort fields.
/// Pooled so the first prefill at each P warms the allocator and
/// subsequent prefills capture the kernel chain into a HIP graph
/// (hipMalloc is forbidden inside `Graph::begin_capture`).
struct MoeGroupedScratch<'a> {
    count:  PooledBuf<'a, i32>,   // [n_expert]   routing histogram
    cursor: PooledBuf<'a, i32>,   // [n_expert]   scatter cursor
    eoff:   PooledBuf<'a, i32>,   // [n_expert+1] expert entry offsets
    toff:   PooledBuf<'a, i32>,   // [n_expert+1] expert GEMM-tile offsets
    perm:   PooledBuf<'a, i32>,   // [cw*n_used]  entries grouped by expert
    g_in:   PooledBuf<'a, u8>,    // [cw*n_used, hidden/32] gathered gate_up input
    e_act:  PooledBuf<'a, f32>,   // [cw*n_used, expert_ff] sorted GeGLU output
    g_out:  PooledBuf<'a, f32>,   // [cw*n_used, hidden] sorted grouped-down output
}

/// All weights for one Gemma 4 transformer block on device.
pub struct GpuGemma4Block {
    attn_norm:      DeviceBuf<f32>,
    attn_q:         GpuMatvecTensor,
    attn_k:         GpuMatvecTensor,
    /// `None` on full-attention layers — V reuses the K projection.
    attn_v:         Option<GpuMatvecTensor>,
    attn_q_norm:    DeviceBuf<f32>,
    attn_k_norm:    DeviceBuf<f32>,
    attn_output:    GpuMatvecTensor,
    post_attn_norm: DeviceBuf<f32>,
    ffn_norm:       DeviceBuf<f32>,
    ffn_gate:       GpuMatvecTensor,
    ffn_up:         GpuMatvecTensor,
    ffn_down:       GpuMatvecTensor,
    post_ffw_norm:  DeviceBuf<f32>,
    layer_output_scale: f32,
    kind:     AttnKind,
    head_dim: usize,
    n_kv:     usize,
    /// `Some` on MoE layers — the routed-expert branch.
    moe:      Option<MoeBlock>,
    /// `Some(donor)` on KV-sharing layers — this layer computes only Q
    /// and attends against layer `donor`'s KV cache. `None` ⇒ the layer
    /// owns its KV.
    kv_donor: Option<usize>,
    /// `Some` on PLE models (E4B) — the per-layer-embedding gate branch.
    ple:      Option<PleBlock>,
}

impl GpuGemma4Block {
    fn from_gguf(gguf: &GgufFile, layer: u32, kind: AttnKind,
                 head_dim: usize, n_kv: usize, moe: bool,
                 kv_donor: Option<usize>, ple: bool,
                 hidden: usize) -> Result<Self, String> {
        let p = format!("blk.{layer}.");
        let moe_block = if moe {
            // Pre-scale gate_inp_s by 1/sqrt(hidden) so the router's
            // `launch_scale` after the rmsnorm becomes unnecessary —
            // rmsnorm(x) * (w * 1/sqrt(h)) = rmsnorm(x) * w * 1/sqrt(h).
            let inv_sqrt_h = 1.0 / (hidden as f32).sqrt();
            Some(MoeBlock {
                post_ffw_norm_1: load_fp32(gguf, &format!("{p}post_ffw_norm_1.weight"))?,
                pre_ffw_norm_2:  load_fp32(gguf, &format!("{p}pre_ffw_norm_2.weight"))?,
                post_ffw_norm_2: load_fp32(gguf, &format!("{p}post_ffw_norm_2.weight"))?,
                gate_inp:    GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}ffn_gate_inp.weight"))?,
                gate_inp_s:  load_fp32_scaled(gguf, &format!("{p}ffn_gate_inp.scale"), inv_sqrt_h)?,
                gate_up_exps: ExpertTensor::from_gguf(gguf, &format!("{p}ffn_gate_up_exps.weight"))?,
                down_exps:    ExpertTensor::from_gguf(gguf, &format!("{p}ffn_down_exps.weight"))?,
                down_grouped: ExpertTensor::from_gguf_repacked(
                                  gguf, &format!("{p}ffn_down_exps.weight"))?,
                down_exps_s:  load_fp32(gguf, &format!("{p}ffn_down_exps.scale"))?,
            })
        } else { None };
        // V projection: present on all E4B layers, but only sliding
        // layers on the 31B (its full layers reuse the K projection).
        // Load by tensor presence rather than attention kind.
        let attn_v = if gguf.tensor(&format!("{p}attn_v.weight")).is_some() {
            Some(GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}attn_v.weight"))?)
        } else { None };
        let ple_block = if ple {
            Some(PleBlock {
                inp_gate:  GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}inp_gate.weight"))?,
                proj:      GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}proj.weight"))?,
                post_norm: load_fp32(gguf, &format!("{p}post_norm.weight"))?,
            })
        } else { None };
        // layer_output_scale is a [1] f32 — read it to host.
        let los_info = gguf.tensor(&format!("{p}layer_output_scale.weight"))
            .ok_or_else(|| format!("{p}layer_output_scale.weight missing"))?;
        let los_bytes = gguf.tensor_data(&format!("{p}layer_output_scale.weight"))
            .map_err(|e| e.to_string())?.ok_or("los no data")?;
        let layer_output_scale = if los_info.ggml_type == GgmlType::F32 {
            bytemuck::cast_slice::<u8, f32>(los_bytes)[0]
        } else { return Err("layer_output_scale not F32".into()); };

        Ok(Self {
            attn_norm:      load_fp32(gguf, &format!("{p}attn_norm.weight"))?,
            attn_q:         GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}attn_q.weight"))?,
            attn_k:         GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}attn_k.weight"))?,
            attn_v,
            attn_q_norm:    load_fp32(gguf, &format!("{p}attn_q_norm.weight"))?,
            attn_k_norm:    load_fp32(gguf, &format!("{p}attn_k_norm.weight"))?,
            attn_output:    GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}attn_output.weight"))?,
            post_attn_norm: load_fp32(gguf, &format!("{p}post_attention_norm.weight"))?,
            ffn_norm:       load_fp32(gguf, &format!("{p}ffn_norm.weight"))?,
            ffn_gate:       GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}ffn_gate.weight"))?,
            ffn_up:         GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}ffn_up.weight"))?,
            ffn_down:       GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}ffn_down.weight"))?,
            post_ffw_norm:  load_fp32(gguf, &format!("{p}post_ffw_norm.weight"))?,
            layer_output_scale, kind, head_dim, n_kv,
            moe: moe_block,
            kv_donor,
            ple: ple_block,
        })
    }
}

/// Per-layer int8 KV cache. K and V are stored as symmetric int8 with
/// one f32 scale per (token, head) — 4× smaller than f32, and the
/// attention kernel dots K against a quantized Q via dp4a. Sliding and
/// full layers have different (n_kv, head_dim), so each sizes its own.
pub struct Gemma4KvCache {
    k:  DeviceBuf<i8>,    // [max_seq, n_kv, head_dim]
    v:  DeviceBuf<i8>,
    ks: DeviceBuf<f32>,   // [max_seq, n_kv]
    vs: DeviceBuf<f32>,
    n_kv: usize,
    head_dim: usize,
    max_seq: usize,
    len: usize,
}

impl Gemma4KvCache {
    fn new(n_kv: usize, head_dim: usize, max_seq: usize) -> Result<Self, String> {
        let kv_dim = n_kv * head_dim;
        Ok(Self {
            k:  DeviceBuf::new(max_seq * kv_dim)?,
            v:  DeviceBuf::new(max_seq * kv_dim)?,
            ks: DeviceBuf::new(max_seq * n_kv)?,
            vs: DeviceBuf::new(max_seq * n_kv)?,
            n_kv, head_dim, max_seq, len: 0,
        })
    }
}

/// Per-token mutable state: one KV cache per layer.
pub struct Gemma4GpuState {
    caches: Vec<Gemma4KvCache>,
    /// SuperQuant 2-tier KV alternative — Some(...) iff opt-in via
    /// `Gemma4GpuState::new_with_superquant`. When set, `caches`
    /// above is still allocated but unused (the cost is small
    /// relative to model weights and lets the same struct shape
    /// support both modes without a runtime enum).
    pub superquant: Option<Vec<crate::runtime::kv_superquant::SuperQuantKvCache>>,
    pub pos: usize,
    /// Per-state cache of captured prefill graphs, keyed by token count
    /// `P`. After the first capture at each P (which costs hundreds of
    /// ms — graph compilation, kernel-arg resolution, dispatch list
    /// build), subsequent prefills at the same P launch the stored
    /// `GraphExec` directly, skipping `begin_capture` / kernel-enqueue
    /// / `end_capture` / `instantiate` entirely. Per-state (not
    /// per-runtime) because each captured graph hardcodes this state's
    /// KV cache pointers.
    ///
    /// Empty + unused when SuperQuant is enabled: the demote-cascade
    /// kernels do D2D memcpys on the null stream which can't be
    /// captured into a HIP graph.
    prefill_graphs: std::collections::HashMap<usize, GraphExec>,
}

impl Gemma4GpuState {
    pub fn new(model: &Gemma4Model, max_seq: usize) -> Result<Self, String> {
        let cfg = &model.config;
        let mut caches = Vec::with_capacity(cfg.block_count as usize);
        for layer in 0..cfg.block_count as usize {
            caches.push(Gemma4KvCache::new(
                cfg.kv_heads[layer] as usize,
                cfg.head_dim(layer) as usize,
                max_seq)?);
        }
        Ok(Self { caches, superquant: None, pos: 0,
                  prefill_graphs: std::collections::HashMap::new() })
    }

    /// Opt-in constructor — allocates BOTH the standard int8 caches
    /// (unused but cheap to keep) AND the SuperQuant per-layer caches.
    /// The cache is then used by the forward pass when
    /// `superquant.is_some()`.
    pub fn new_with_superquant(
        model: &Gemma4Model,
        max_seq: usize,
        kernel_cache: &crate::runtime::KernelCache,
        config: crate::runtime::kv_superquant::SuperQuantConfig,
    ) -> Result<Self, String> {
        use crate::runtime::kv_superquant::SuperQuantKvCache;
        let mut state = Self::new(model, max_seq)?;
        let cfg = &model.config;
        let mut sq = Vec::with_capacity(cfg.block_count as usize);
        for layer in 0..cfg.block_count as usize {
            sq.push(SuperQuantKvCache::new(
                kernel_cache,
                cfg.kv_heads[layer] as usize,
                cfg.head_dim(layer) as usize,
                config)?);
        }
        state.superquant = Some(sq);
        Ok(state)
    }

    /// True if this state was built with SuperQuant enabled.
    pub fn is_superquant(&self) -> bool { self.superquant.is_some() }

    /// One-shot migration of the populated int8 KV cache contents
    /// (typically just after `prefill_forward`) into the SuperQuant
    /// per-layer caches. The Warm tier gets the most recent
    /// `warm_cap` positions; anything older demotes to Cold via the
    /// q8→turbo3 promote kernel. After this, `self.pos` is unchanged
    /// (it tracks the logical prompt position, which both cache
    /// representations share); decode-step writes hereafter route to
    /// SuperQuant by virtue of `superquant.is_some()`.
    ///
    /// Caller is responsible for making sure the prefill populated
    /// `self.caches[..]` with `self.pos` positions. No-op when
    /// SuperQuant is not enabled.
    pub fn migrate_int8_to_superquant(&self, kernel_cache: &crate::runtime::KernelCache)
        -> Result<(), String>
    {
        let Some(sq_caches) = &self.superquant else { return Ok(()); };
        let p = self.pos;
        if p == 0 { return Ok(()); }
        for (i, (c, sq)) in self.caches.iter().zip(sq_caches.iter()).enumerate() {
            if c.len != p {
                return Err(format!(
                    "migrate: layer {i} int8 cache len {} doesn't match state.pos {}",
                    c.len, p));
            }
            if p > sq.max_seq() {
                return Err(format!(
                    "migrate: layer {i} prompt of {p} positions exceeds SuperQuant capacity {}",
                    sq.max_seq()));
            }
            let warm_cap = sq.config.warm_cap;
            let warm_start = p.saturating_sub(warm_cap);
            let n_cold = warm_start;
            let n_warm = p - warm_start;

            // Cold tier: demote int8 positions [0, n_cold) → turbo3.
            // Uses the existing q8→turbo3 promote kernel which expects
            // contiguous int8 + per-(slot,head) scale source.
            if n_cold > 0 {
                use crate::runtime::kv_turbo3::launch_promote_q8_to_turbo3;
                launch_promote_q8_to_turbo3(kernel_cache,
                    c.k.raw_ptr(), c.ks.raw_ptr(),
                    sq.signs1_k.raw_ptr(), sq.signs2_k.raw_ptr(),
                    sq.cold_k.raw_ptr(),
                    n_cold as u32, c.n_kv as u32, c.head_dim as u32)?;
                launch_promote_q8_to_turbo3(kernel_cache,
                    c.v.raw_ptr(), c.vs.raw_ptr(),
                    sq.signs1_v.raw_ptr(), sq.signs2_v.raw_ptr(),
                    sq.cold_v.raw_ptr(),
                    n_cold as u32, c.n_kv as u32, c.head_dim as u32)?;
            }

            // Warm tier: D2D copy int8 positions [warm_start, p) → Warm
            // positions [0, n_warm). Same format on both sides.
            if n_warm > 0 {
                let row_elems = c.n_kv * c.head_dim;
                sq.warm_k.copy_range_from_device(&c.k,
                    warm_start * row_elems, 0, n_warm * row_elems)?;
                sq.warm_v.copy_range_from_device(&c.v,
                    warm_start * row_elems, 0, n_warm * row_elems)?;
                sq.warm_ks.copy_range_from_device(&c.ks,
                    warm_start * c.n_kv, 0, n_warm * c.n_kv)?;
                sq.warm_vs.copy_range_from_device(&c.vs,
                    warm_start * c.n_kv, 0, n_warm * c.n_kv)?;
            }

            sq.warm_count.set(n_warm);
            sq.cold_count.set(n_cold);
        }
        crate::hip::Device(0).synchronize()?;
        Ok(())
    }

    pub fn reset(&mut self) {
        for c in &mut self.caches { c.len = 0; }
        if let Some(sq) = &self.superquant {
            for c in sq { c.reset(); }
        }
        self.pos = 0;
    }

    /// Truncate the cache to `new_len` populated entries on every layer
    /// and set `pos = new_len`. Used by spec-decode verify after a
    /// rejection: the rejected slot's KV is unused (overwritten when
    /// the replacement token is forwarded), so just reset the
    /// high-water-marks.
    pub fn truncate(&mut self, new_len: usize) {
        // SuperQuant doesn't support truncate (would need per-tier
        // rollback + potential cold→warm re-promotion). Spec-decode
        // is incompatible with SuperQuant for now; the public path
        // catches this earlier when the caller opts into both.
        if self.superquant.is_some() {
            panic!("Gemma4GpuState::truncate not supported with SuperQuant \
                    (would need per-tier rollback). Disable SuperQuant for \
                    spec-decode workloads.");
        }
        for c in &mut self.caches {
            c.len = new_len.min(c.max_seq);
        }
        self.pos = new_len;
    }

    /// Snapshot the populated portion of every layer's KV cache plus
    /// the current decode position. The snapshot lives on the device
    /// (no host roundtrip) and is sized to exactly the bytes in use,
    /// so it's cheap to take and restore for prefix-caching workflows.
    pub fn snapshot(&self) -> Result<Gemma4StateSnapshot, String> {
        if self.superquant.is_some() {
            return Err("Gemma4GpuState::snapshot not supported with SuperQuant \
                        (3-tier rollback requires reverse demotion kernels not \
                        yet implemented). Disable SuperQuant to use snapshot/restore.".into());
        }
        let mut layers = Vec::with_capacity(self.caches.len());
        for c in &self.caches {
            let kv_dim = c.n_kv * c.head_dim;
            let k  = DeviceBuf::new(c.len * kv_dim)?;
            let v  = DeviceBuf::new(c.len * kv_dim)?;
            let ks = DeviceBuf::new(c.len * c.n_kv)?;
            let vs = DeviceBuf::new(c.len * c.n_kv)?;
            if c.len > 0 {
                k .copy_range_from_device(&c.k,  0, 0, c.len * kv_dim)?;
                v .copy_range_from_device(&c.v,  0, 0, c.len * kv_dim)?;
                ks.copy_range_from_device(&c.ks, 0, 0, c.len * c.n_kv)?;
                vs.copy_range_from_device(&c.vs, 0, 0, c.len * c.n_kv)?;
            }
            layers.push(Gemma4LayerSnapshot {
                k, v, ks, vs,
                n_kv: c.n_kv, head_dim: c.head_dim, len: c.len,
            });
        }
        Ok(Gemma4StateSnapshot { layers, pos: self.pos })
    }

    /// Restore a previously-taken snapshot. The state must have been
    /// built from a model with the same per-layer (n_kv, head_dim)
    /// shape — they're checked, and the `max_seq` of the live cache
    /// must hold at least `snapshot.pos` tokens.
    pub fn restore(&mut self, snap: &Gemma4StateSnapshot) -> Result<(), String> {
        if snap.layers.len() != self.caches.len() {
            return Err(format!("restore: layer count mismatch (snap {}, state {})",
                               snap.layers.len(), self.caches.len()));
        }
        for (i, (c, l)) in self.caches.iter_mut().zip(snap.layers.iter()).enumerate() {
            if c.n_kv != l.n_kv || c.head_dim != l.head_dim {
                return Err(format!("restore: layer {i} shape mismatch \
                    (snap n_kv={} head_dim={}, state n_kv={} head_dim={})",
                    l.n_kv, l.head_dim, c.n_kv, c.head_dim));
            }
            if l.len > c.max_seq {
                return Err(format!("restore: layer {i} snapshot len {} > cache max_seq {}",
                                   l.len, c.max_seq));
            }
            if l.len > 0 {
                let kv_dim = c.n_kv * c.head_dim;
                c.k .copy_range_from_device(&l.k,  0, 0, l.len * kv_dim)?;
                c.v .copy_range_from_device(&l.v,  0, 0, l.len * kv_dim)?;
                c.ks.copy_range_from_device(&l.ks, 0, 0, l.len * c.n_kv)?;
                c.vs.copy_range_from_device(&l.vs, 0, 0, l.len * c.n_kv)?;
            }
            c.len = l.len;
        }
        self.pos = snap.pos;
        Ok(())
    }
}

/// Read-only view of one layer's int8 KV cache + scales. Returned by
/// [`Gemma4GpuState::layer_kv_view`] so an external consumer (e.g. the
/// MTP drafter, which attends over the target's KV) can launch
/// attention kernels against the target's cache without touching the
/// runtime's private fields.
pub struct LayerKvView<'a> {
    pub k:  &'a DeviceBuf<i8>,
    pub v:  &'a DeviceBuf<i8>,
    pub ks: &'a DeviceBuf<f32>,
    pub vs: &'a DeviceBuf<f32>,
    pub n_kv: usize,
    pub head_dim: usize,
    pub len: usize,
    pub max_seq: usize,
}

impl Gemma4GpuState {
    /// View a specific layer's KV cache. Panics if `layer` is out of
    /// range. `view.len` is the current populated length.
    pub fn layer_kv_view(&self, layer: usize) -> LayerKvView<'_> {
        let c = &self.caches[layer];
        LayerKvView {
            k: &c.k, v: &c.v, ks: &c.ks, vs: &c.vs,
            n_kv: c.n_kv, head_dim: c.head_dim,
            len: c.len, max_seq: c.max_seq,
        }
    }
}

/// Device-resident snapshot of a `Gemma4GpuState`, used to reuse a
/// prefilled prefix (system + saved-user context) across multiple
/// per-turn completions. Construct with [`Gemma4GpuState::snapshot`]
/// and apply with [`Gemma4GpuState::restore`].
pub struct Gemma4StateSnapshot {
    layers: Vec<Gemma4LayerSnapshot>,
    pos: usize,
}

impl Gemma4StateSnapshot {
    pub fn pos(&self) -> usize { self.pos }
}

struct Gemma4LayerSnapshot {
    k:  DeviceBuf<i8>,
    v:  DeviceBuf<i8>,
    ks: DeviceBuf<f32>,
    vs: DeviceBuf<f32>,
    n_kv: usize,
    head_dim: usize,
    len: usize,
}

pub struct GpuGemma4 {
    token_embd:  GpuMatvecTensor,   // also the tied output projection
    output_norm: DeviceBuf<f32>,
    blocks:      Vec<GpuGemma4Block>,

    // RoPE tables — sliding (rotary 256, base 1e4) and full (512, 1e6).
    rope_cos_swa: DeviceBuf<f32>,
    rope_sin_swa: DeviceBuf<f32>,
    rope_cos_full: DeviceBuf<f32>,
    rope_sin_full: DeviceBuf<f32>,

    // Scratch (sized to the per-layer maxima).
    hidden_a:    DeviceBuf<f32>,
    hidden_b:    DeviceBuf<f32>,
    normed:      DeviceBuf<f32>,
    q_buf:       DeviceBuf<f32>,
    k_proj:      DeviceBuf<f32>,
    k_norm:      DeviceBuf<f32>,
    v_norm:      DeviceBuf<f32>,
    attn_concat: DeviceBuf<f32>,
    ffn_a:       DeviceBuf<f32>,
    ffn_b:       DeviceBuf<f32>,
    logits:      DeviceBuf<f32>,
    /// All-ones weight for the plain (unweighted) V RMSNorm.
    ones:        DeviceBuf<f32>,

    /// Per-Layer-Embedding state (E4B). `ple_raw`/`ple_proj` hold the
    /// `[n_embd_per_layer · n_layer]` per-layer embeddings for the
    /// current token; `ple_gate`/`ple_tmp` are per-block scratch.
    ple:        Option<PleGlobal>,
    ple_raw:    DeviceBuf<f32>,
    ple_proj:   DeviceBuf<f32>,
    ple_gate:   DeviceBuf<f32>,
    ple_tmp:    DeviceBuf<f32>,
    n_embd_per_layer: usize,
    /// Layers `[0, n_layer_kv_from_start)` own their KV cache; later
    /// layers share a donor's. Equals `block_count` with no sharing.
    n_layer_kv_from_start: usize,

    // MoE scratch (allocated for all models; tiny when unused).
    moe_logits:  DeviceBuf<f32>,   // [MOE_PREFILL_CHUNK, n_expert]
    moe_ids:     DeviceBuf<i32>,   // [MOE_PREFILL_CHUNK, n_expert_used]
    moe_weights: DeviceBuf<f32>,   // [MOE_PREFILL_CHUNK, n_expert_used]
    moe_acc:     DeviceBuf<f32>,   // [hidden] — expert mixture accumulator
    cur_mlp:     DeviceBuf<f32>,   // [hidden] — shared-MLP result, kept live
    expert_gu:   DeviceBuf<f32>,   // [n_used · 2·expert_ff] — fused gate_up
    expert_outs: DeviceBuf<f32>,   // [n_used · hidden]       — per-expert down
    xq8_experts: DeviceBuf<u8>,    // batched int8 activation for the 8 experts

    // Kernel modules.
    m_rmsnorm:   Module,
    m_rmsnorm_add: Module,
    m_rmsnorm_q8: Module,
    m_rmsnorm_mh: Module,
    m_rope:      Module,
    m_add:       Module,
    m_geglu:     Module,
    m_softcap:   Module,
    m_scale:     Module,
    m_attn_win:  Module,
    /// Split-K decode attention (partial + merge) — see attn_partial_q8.cpp.
    m_attn_partial: Module,
    /// SuperQuant 2-tier attention (opt-in). Always loaded so opt-in
    /// at state construction time doesn't need to recompile.
    m_attn_superquant: Module,
    /// Rotated-space variant of SuperQuant attention. Q is pre-rotated
    /// (m_rotate_q_rht) once per attention call, then cold scoring +
    /// V accumulation happen in rotated space, with a single per-(head,
    /// group) iRHT at the end. Skips the per-position FWHT that
    /// dominates the v1 path's cold latency.
    m_attn_superquant_rs: Module,
    /// Wave-parallel rotated-space SuperQuant attention — 4 waves
    /// dispatch 4 cached positions in parallel, no per-position
    /// `__syncthreads`. Default cold path; opt out via
    /// REINSTINCT_KV_SUPERQUANT_RS=1 (single-wave) or _NAIVE=1.
    m_attn_superquant_wp: Module,
    m_rotate_q_rht:        Module,
    /// Per-call Q-rotation scratch — fp32 [n_heads × head_dim_max].
    /// Holds Q × R_K (K's RHT applied to Q) for the rotated-space
    /// attention kernel. Allocated once at GpuGemma4::new.
    q_rot_scratch: DeviceBuf<f32>,
    m_attn_merge:   Module,
    /// Partial-attention scratch: [n_heads, ATTN_MAX_SPLITS, head_dim_max]
    /// and [n_heads, ATTN_MAX_SPLITS] for the running max / denominator.
    attn_o_partial: DeviceBuf<f32>,
    attn_m_partial: DeviceBuf<f32>,
    attn_l_partial: DeviceBuf<f32>,
    /// `REINSTINCT_OLD_ATTN` — use the original single-block kernel.
    use_old_attn:   bool,
    /// `REINSTINCT_MOE_PROFILE` — per-stage decode timing.
    moe_prof_on:    bool,
    prof_mark:      std::cell::Cell<std::time::Instant>,
    prof_buckets:   std::cell::RefCell<Vec<(&'static str, f64)>>,
    m_embed_q5k: Module,
    m_embed_q8_0: Module,
    m_mv_f32:    Module,
    m_mv_q4k:    Module,
    m_mv_q5k:    Module,
    m_mv_q6k:    Module,
    m_mv_q8_0:   Module,
    m_mv_f16:    Module,
    m_quantize:  Module,
    m_mv_q4k_dp4a: Module,
    m_mv_q5k_dp4a: Module,
    /// Batched (K=2..4 input rows) Q5_K dp4a matvec — used by
    /// verify_forward's lm_head to compute all K logits rows in one
    /// kernel launch instead of K separate calls.
    m_mv_q5k_dp4a_batched: Module,
    m_mv_q6k_dp4a: Module,
    m_mv_q8_0_dp4a: Module,
    m_mv_q8_0_repacked: Module,
    m_mv_q4k_repacked: Module,
    m_mv_q5k_repacked: Module,
    m_mv_q6k_repacked: Module,
    m_moe_topk:  Module,
    m_moe_mv_q6k:  Module,
    m_moe_mv_q6k_repacked: Module,
    m_moe_mv_q8_0: Module,
    m_moe_mv_q8_0_down: Module,
    m_moe_geglu:   Module,
    m_moe_geglu_q8: Module,
    m_moe_combine: Module,
    /// Grouped-expert GEMM modules (MoE prefill, `REINSTINCT_MOE_GROUPED`).
    m_expert_sort: Module,
    m_grouped_q6k: Module,
    m_grouped_q8_0: Module,
    m_kv_write:    Module,
    /// Scratch for the int8-quantized activation feeding the dp4a matvec.
    xq8: DeviceBuf<u8>,

    /// Pre-allocated scratch for `verify_forward` (MTP spec-decode).
    /// Sized to `MAX_VERIFY_K` rows of the worst-case per-layer dim,
    /// reused across spec-decode rounds to avoid the ~600 hipMalloc
    /// per call that would otherwise dominate verify time.
    v_x:        DeviceBuf<f32>,
    v_normed:   DeviceBuf<f32>,
    v_q:        DeviceBuf<f32>,
    v_k:        DeviceBuf<f32>,
    v_v:        DeviceBuf<f32>,
    v_k_norm:   DeviceBuf<f32>,
    v_v_norm:   DeviceBuf<f32>,
    v_attn:     DeviceBuf<f32>,
    v_attn_out: DeviceBuf<f32>,
    v_gate:     DeviceBuf<f32>,
    v_up:       DeviceBuf<f32>,
    v_mlp:      DeviceBuf<f32>,
    v_logits:   DeviceBuf<f32>,
    v_tokens:   DeviceBuf<u32>,
    /// Device-resident base_pos for verify_forward. Set once per call
    /// (before kernels are issued / before the captured graph is
    /// replayed) so the rope-batched, kv-quant-prefill, and attn-step
    /// kernels can read the per-call value without being re-captured.
    v_base_pos: DeviceBuf<u32>,
    /// Max K supported per verify_forward call.
    max_verify_k: usize,

    /// Decode token + position, device-resident so the embed / rope /
    /// attention / KV-write kernels read them at execution time — which
    /// makes the whole forward capturable into one parametric HIP graph.
    d_token: DeviceBuf<u32>,
    d_pos:   DeviceBuf<u32>,
    max_seq: usize,

    stream: Stream,
    /// Kernel cache reference — needed only by the SuperQuant write
    /// path, which calls `SuperQuantKvCache::write_step(&cache, ...)`.
    /// Standard int8 path doesn't touch this field.
    kernel_cache: KernelCache,

    /// Prefill context — rocBLAS handle + kernels built once at load and
    /// reused by every `prefill_forward` call. Recreating these per call
    /// (rocBLAS init + ~16 module loads, ~150 ms) dominated small-model
    /// prefill latency.
    rocblas:       crate::hip::rocblas::Handle,
    prefill_gemm:  crate::runtime::prefill::PrefillGemm,
    m_rope_pf:     Module,
    m_attn_pf:     Module,
    m_kvq_pf:      Module,
    m_rope_b:      Module,        // rope_apply_batched_f32 (base_pos param)
    m_attn_step_q8_b: Module,     // batched int8 attn over the decode KV cache
    m_permute_pf:  Module,

    // Dimensions.
    hidden:     usize,
    ffn:        usize,
    vocab:      usize,
    n_heads:    usize,
    rms_eps:    f32,
    softcap:    f32,
    sliding_window: usize,
    rope_dim_swa:  usize,
    rope_dim_full: usize,
    // MoE dimensions (0 on dense models).
    n_expert:      usize,
    n_expert_used: usize,
    expert_ff:     usize,

    /// Per-call activation/scratch pool for `prefill_forward`. First
    /// call at each token count `P` runs uncaptured to fill the pool;
    /// subsequent calls capture the kernel chain into a HIP graph.
    pool_f32: DeviceBufPool<f32>,
    pool_u8:  DeviceBufPool<u8>,
    pool_u32: DeviceBufPool<u32>,
    pool_i32: DeviceBufPool<i32>,
    prefill_warm_p: std::cell::RefCell<std::collections::HashSet<usize>>,
}

impl GpuGemma4 {
    pub fn new(model: &Gemma4Model, gguf: &GgufFile, cache: &KernelCache, max_seq: usize)
        -> Result<Self, String>
    {
        let cfg = &model.config;
        let hidden = cfg.hidden_size as usize;
        let ffn    = cfg.ffn_size as usize;
        let vocab  = cfg.vocab_size as usize;
        let n_heads = cfg.n_heads as usize;
        let hd_max = cfg.head_dim_full.max(cfg.head_dim_swa) as usize;
        let q_max  = n_heads * hd_max;
        let kv_max = cfg.kv_heads.iter().copied().max().unwrap_or(0) as usize * hd_max;

        let token_embd  = GpuMatvecTensor::from_gguf(gguf, "token_embd.weight")?;
        let output_norm = load_fp32(gguf, "output_norm.weight")?;

        let moe = cfg.is_moe();
        // MoE scratch sizes — .max(1) keeps the buffers non-empty on the
        // dense 31B (which leaves the expert counts at 0).
        let n_used_a    = (cfg.expert_used_count as usize).max(1);
        let expert_ff_a = (cfg.expert_ff_size as usize).max(32);
        let mut blocks = Vec::with_capacity(cfg.block_count as usize);
        for layer in 0..cfg.block_count {
            let l = layer as usize;
            let kind = cfg.attn_kinds[l];
            let kv_donor = if cfg.layer_has_kv(l) { None } else { Some(cfg.kv_donor(l)) };
            blocks.push(GpuGemma4Block::from_gguf(
                gguf, layer, kind,
                cfg.head_dim(l) as usize,
                cfg.kv_heads[l] as usize, moe, kv_donor, cfg.has_ple(),
                hidden)?);
        }

        // Per-Layer-Embedding global weights (E4B only).
        let ple = if cfg.has_ple() {
            Some(PleGlobal {
                tok_embd:   GpuMatvecTensor::from_gguf(gguf, "per_layer_token_embd.weight")?,
                model_proj: load_bf16_as_f32_matvec(gguf, "per_layer_model_proj.weight")?,
                proj_norm:  load_fp32(gguf, "per_layer_proj_norm.weight")?,
            })
        } else { None };
        let ple_dim = (cfg.n_embd_per_layer as usize * cfg.block_count as usize).max(1);

        // RoPE tables for both kinds.
        let build_rope = |rotary: usize, base: f32| -> Result<(DeviceBuf<f32>, DeviceBuf<f32>), String> {
            let rc = crate::cpu::rope::RopeCache::new(rotary, max_seq, base);
            let mut cos = vec![0.0f32; max_seq * rotary];
            let mut sin = vec![0.0f32; max_seq * rotary];
            for pos in 0..max_seq {
                let (c, s) = rc.get(pos);
                cos[pos*rotary..(pos+1)*rotary].copy_from_slice(c);
                sin[pos*rotary..(pos+1)*rotary].copy_from_slice(s);
            }
            Ok((DeviceBuf::from_slice(&cos)?, DeviceBuf::from_slice(&sin)?))
        };
        let (rope_cos_swa, rope_sin_swa) =
            build_rope(cfg.rope_dim_swa as usize, cfg.rope_freq_base_swa)?;
        let (rope_cos_full, rope_sin_full) =
            build_rope(cfg.rope_dim_full as usize, cfg.rope_freq_base)?;

        let ld = |name: &str, src: &str| -> Result<Module, String> {
            Module::load(&cache.compile(name, src)?)
        };

        let ones = DeviceBuf::from_slice(&vec![1.0f32; hd_max])?;

        // Scratch for the quantized activation: one BlockQ8 (40 bytes)
        // per 32 input elements, sized to the widest matvec.
        let max_in_dim = blocks.iter()
            .flat_map(|b| [b.attn_q.in_dim, b.attn_k.in_dim, b.attn_output.in_dim,
                           b.ffn_gate.in_dim, b.ffn_up.in_dim, b.ffn_down.in_dim])
            .chain(std::iter::once(token_embd.in_dim))
            .max().unwrap_or(0) as usize;
        let xq8 = DeviceBuf::<u8>::new((max_in_dim / 32) * 40)?;

        // --- Prefill context: built once, reused every prefill call. ---
        let stream = Stream::new()?;
        let (mut pmax_w, mut pmax_in, mut pmax_out) = (0usize, 0usize, 0usize);
        for b in &blocks {
            let mut ws: Vec<&GpuMatvecTensor> = vec![
                &b.attn_q, &b.attn_k, &b.attn_output, &b.ffn_gate, &b.ffn_up, &b.ffn_down];
            if let Some(wv) = &b.attn_v { ws.push(wv); }
            if let Some(pb) = &b.ple { ws.push(&pb.inp_gate); ws.push(&pb.proj); }
            for w in ws {
                pmax_w   = pmax_w.max(w.in_dim as usize * w.out_dim as usize);
                pmax_in  = pmax_in.max(w.in_dim as usize);
                pmax_out = pmax_out.max(w.out_dim as usize);
            }
        }
        if let Some(pg) = &ple {
            let w = &pg.model_proj;
            pmax_w   = pmax_w.max(w.in_dim as usize * w.out_dim as usize);
            pmax_in  = pmax_in.max(w.in_dim as usize);
            pmax_out = pmax_out.max(w.out_dim as usize);
        }
        let rocblas = crate::hip::rocblas::Handle::new()?;
        rocblas.set_stream(&stream)?;
        let prefill_gemm = crate::runtime::prefill::PrefillGemm::new(
            cache, pmax_w.max(1), max_seq * pmax_in.max(1), max_seq * pmax_out.max(1))?;
        let m_rope_pf    = ld("rope_prefill", ROPE_PREFILL_SRC)?;
        let m_attn_pf    = ld("attn_prefill_flash", ATTN_PREFILL_SRC)?;
        let m_kvq_pf     = ld("kv_quant_prefill", KV_QUANT_PREFILL_SRC)?;
        let m_rope_b     = ld("rope_batched", ROPE_BATCHED_SRC)?;
        let m_attn_step_q8_b = ld("attn_step_q8_batched", ATTN_STEP_Q8_BATCHED_SRC)?;
        let m_permute_pf = ld("permute_ple", PERMUTE_PLE_SRC)?;

        Ok(Self {
            token_embd, output_norm, blocks,
            rope_cos_swa, rope_sin_swa, rope_cos_full, rope_sin_full,
            hidden_a:    DeviceBuf::new(hidden)?,
            hidden_b:    DeviceBuf::new(hidden)?,
            normed:      DeviceBuf::new(hidden)?,
            q_buf:       DeviceBuf::new(q_max)?,
            k_proj:      DeviceBuf::new(kv_max)?,
            k_norm:      DeviceBuf::new(kv_max)?,
            v_norm:      DeviceBuf::new(kv_max)?,
            attn_concat: DeviceBuf::new(q_max)?,
            ffn_a:       DeviceBuf::new(ffn)?,
            ffn_b:       DeviceBuf::new(ffn)?,
            logits:      DeviceBuf::new(vocab)?,
            ones,
            ple,
            ple_raw:  DeviceBuf::new(ple_dim)?,
            ple_proj: DeviceBuf::new(ple_dim)?,
            ple_gate: DeviceBuf::new((cfg.n_embd_per_layer as usize).max(1))?,
            ple_tmp:  DeviceBuf::new(hidden)?,
            n_embd_per_layer: cfg.n_embd_per_layer as usize,
            n_layer_kv_from_start: cfg.n_layer_kv_from_start as usize,
            m_rmsnorm:     ld("rmsnorm", RMSNORM_SRC)?,
            m_rmsnorm_add: ld("rmsnorm_add", RMSNORM_ADD_SRC)?,
            m_rmsnorm_q8:  ld("rmsnorm_q8", RMSNORM_Q8_SRC)?,
            m_rmsnorm_mh: ld("rmsnorm_multihead", RMSNORM_MULTIHEAD_SRC)?,
            m_rope:       ld("rope", ROPE_SRC)?,
            m_add:        ld("add_inplace", ADD_INPLACE_SRC)?,
            m_geglu:      ld("geglu", GEGLU_SRC)?,
            m_softcap:    ld("logit_softcap", LOGIT_SOFTCAP_SRC)?,
            m_scale:      ld("scale_inplace", SCALE_INPLACE_SRC)?,
            m_attn_win:   ld("attn_step_window", ATTN_WINDOW_SRC)?,
            m_attn_partial: ld("attn_partial_q8", ATTN_PARTIAL_Q8_SRC)?,
            m_attn_superquant: ld("attn_partial_superquant", ATTN_PARTIAL_SQ_SRC)?,
            m_attn_superquant_rs: ld("attn_partial_superquant_rs", ATTN_PARTIAL_SQRS_SRC)?,
            m_attn_superquant_wp: ld("attn_partial_superquant_wp", ATTN_PARTIAL_SQWP_SRC)?,
            m_rotate_q_rht:    ld("rotate_q_rht", ROTATE_Q_RHT_SRC)?,
            q_rot_scratch:     DeviceBuf::new(n_heads * hd_max)?,
            m_attn_merge:   ld("attn_merge", ATTN_MERGE_SRC)?,
            attn_o_partial: DeviceBuf::new(n_heads * ATTN_MAX_SPLITS as usize * hd_max)?,
            attn_m_partial: DeviceBuf::new(n_heads * ATTN_MAX_SPLITS as usize)?,
            attn_l_partial: DeviceBuf::new(n_heads * ATTN_MAX_SPLITS as usize)?,
            use_old_attn:   std::env::var_os("REINSTINCT_OLD_ATTN").is_some(),
            moe_prof_on:    std::env::var_os("REINSTINCT_MOE_PROFILE").is_some(),
            prof_mark:      std::cell::Cell::new(std::time::Instant::now()),
            prof_buckets:   std::cell::RefCell::new(Vec::new()),
            m_embed_q5k:  ld("embed_lookup_q5_k", EMBED_Q5K_SRC)?,
            m_embed_q8_0: ld("embed_lookup_q8_0", EMBED_Q8_0_SRC)?,
            m_mv_f32:     ld("matvec_f32_b256", MATVEC_F32_B256_SRC)?,
            m_mv_q4k:     ld("matvec_q4_k_rowblock", MATVEC_Q4K_W_SRC)?,
            m_mv_q5k:     ld("matvec_q5_k_rowblock", MATVEC_Q5K_W_SRC)?,
            m_mv_q6k:     ld("matvec_q6_k_rowblock", MATVEC_Q6K_W_SRC)?,
            m_mv_q8_0:    ld("matvec_q8_0_wave64", MATVEC_Q8_0_W_SRC)?,
            m_mv_f16:     ld("matvec_f16_wave64", MATVEC_F16_W_SRC)?,
            m_quantize:     ld("quantize_q8", QUANTIZE_Q8_SRC)?,
            m_mv_q4k_dp4a:  ld("matvec_q4_k_dp4a", MATVEC_Q4K_DP4A_SRC)?,
            m_mv_q5k_dp4a:  ld("matvec_q5_k_dp4a", MATVEC_Q5K_DP4A_SRC)?,
            m_mv_q5k_dp4a_batched: ld("matvec_q5_k_dp4a_batched",
                                       MATVEC_Q5K_DP4A_BATCHED_SRC)?,
            m_mv_q6k_dp4a:  ld("matvec_q6_k_dp4a", MATVEC_Q6K_DP4A_SRC)?,
            m_mv_q8_0_dp4a: ld("matvec_q8_0_dp4a", MATVEC_Q8_0_DP4A_SRC)?,
            m_mv_q8_0_repacked: ld("matvec_q8_0_repacked", MATVEC_Q8_0_REPACKED_SRC)?,
            m_mv_q4k_repacked: ld("matvec_q4k_repacked", MATVEC_Q4K_REPACKED_SRC)?,
            m_mv_q5k_repacked: ld("matvec_q5k_repacked", MATVEC_Q5K_REPACKED_SRC)?,
            m_mv_q6k_repacked: ld("matvec_q6k_repacked", MATVEC_Q6K_REPACKED_SRC)?,
            m_moe_topk:     ld("moe_topk", MOE_TOPK_SRC)?,
            m_moe_mv_q6k:   ld("moe_matvec_q6k_dp4a", MOE_MATVEC_Q6K_SRC)?,
            m_moe_mv_q6k_repacked: ld("moe_matvec_q6k_repacked", MOE_MV_Q6K_REPACKED_SRC)?,
            m_moe_mv_q8_0:  ld("moe_matvec_q8_0_dp4a", MOE_MATVEC_Q8_0_SRC)?,
            m_moe_mv_q8_0_down: ld("moe_matvec_q8_0_down", MOE_MATVEC_Q8_0_DOWN_SRC)?,
            m_moe_geglu:    ld("moe_geglu", MOE_GEGLU_SRC)?,
            m_moe_geglu_q8: ld("moe_geglu_q8", MOE_GEGLU_Q8_SRC)?,
            m_moe_combine:  ld("moe_combine", MOE_COMBINE_SRC)?,
            m_expert_sort:  ld("moe_expert_sort", MOE_EXPERT_SORT_SRC)?,
            m_grouped_q6k:  ld("mmq_gemm_q6k_grouped", MMQ_Q6K_GROUPED_SRC)?,
            m_grouped_q8_0: ld("mmq_gemm_q8_0_grouped", MMQ_Q8_0_GROUPED_SRC)?,
            m_kv_write:     ld("kv_write", KV_WRITE_SRC)?,
            d_token: DeviceBuf::new(1)?,
            d_pos:   DeviceBuf::new(1)?,
            max_seq,
            // Sized for one prefill chunk: batched topk + the routed
            // matvecs reuse these. Decode / per-token paths use row 0.
            moe_logits:  DeviceBuf::new(MOE_PREFILL_CHUNK * (cfg.expert_count as usize).max(1))?,
            moe_ids:     DeviceBuf::new(MOE_PREFILL_CHUNK * (cfg.expert_used_count as usize).max(1))?,
            moe_weights: DeviceBuf::new(MOE_PREFILL_CHUNK * (cfg.expert_used_count as usize).max(1))?,
            moe_acc:     DeviceBuf::new(hidden)?,
            cur_mlp:     DeviceBuf::new(hidden)?,
            expert_gu:   DeviceBuf::new(n_used_a * 2 * expert_ff_a)?,
            expert_outs: DeviceBuf::new(n_used_a * hidden)?,
            xq8_experts: DeviceBuf::<u8>::new(n_used_a * (expert_ff_a / 32).max(1) * 40)?,
            xq8,

            // Verify scratch: sized to MAX_VERIFY_K × worst-case per-layer dim.
            v_x:        DeviceBuf::<f32>::new(MAX_VERIFY_K * hidden)?,
            v_normed:   DeviceBuf::<f32>::new(MAX_VERIFY_K * hidden)?,
            v_q:        DeviceBuf::<f32>::new(MAX_VERIFY_K * q_max)?,
            v_k:        DeviceBuf::<f32>::new(MAX_VERIFY_K * kv_max)?,
            v_v:        DeviceBuf::<f32>::new(MAX_VERIFY_K * kv_max)?,
            v_k_norm:   DeviceBuf::<f32>::new(MAX_VERIFY_K * kv_max)?,
            v_v_norm:   DeviceBuf::<f32>::new(MAX_VERIFY_K * kv_max)?,
            v_attn:     DeviceBuf::<f32>::new(MAX_VERIFY_K * q_max)?,
            v_attn_out: DeviceBuf::<f32>::new(MAX_VERIFY_K * hidden)?,
            v_gate:     DeviceBuf::<f32>::new(MAX_VERIFY_K * ffn)?,
            v_up:       DeviceBuf::<f32>::new(MAX_VERIFY_K * ffn)?,
            v_mlp:      DeviceBuf::<f32>::new(MAX_VERIFY_K * hidden)?,
            v_logits:   DeviceBuf::<f32>::new(MAX_VERIFY_K * vocab)?,
            v_tokens:   DeviceBuf::<u32>::new(MAX_VERIFY_K)?,
            v_base_pos: DeviceBuf::<u32>::new(1)?,
            max_verify_k: MAX_VERIFY_K,
            stream,
            kernel_cache: cache.clone(),
            rocblas, prefill_gemm,
            m_rope_pf, m_attn_pf, m_kvq_pf, m_permute_pf,
            m_rope_b, m_attn_step_q8_b,
            hidden, ffn, vocab, n_heads,
            rms_eps: cfg.rms_norm_eps,
            softcap: cfg.final_logit_softcapping,
            sliding_window: cfg.sliding_window as usize,
            rope_dim_swa:  cfg.rope_dim_swa as usize,
            rope_dim_full: cfg.rope_dim_full as usize,
            n_expert:      cfg.expert_count as usize,
            n_expert_used: cfg.expert_used_count as usize,
            expert_ff:     cfg.expert_ff_size as usize,
            pool_f32: DeviceBufPool::new(),
            pool_u8:  DeviceBufPool::new(),
            pool_u32: DeviceBufPool::new(),
            pool_i32: DeviceBufPool::new(),
            prefill_warm_p: std::cell::RefCell::new(std::collections::HashSet::new()),
        })
    }

    /// `true` for the 26B-A4B MoE target, `false` for the dense 31B.
    /// Callers (e.g. mtp-gen) use this to skip `capture_verify_graph` —
    /// MoE targets dispatch through `verify_forward_via_decode` (K
    /// sequential `forward_token` calls, each replayed from the
    /// already-captured decode graph), so there's no batched-verify
    /// chain to capture.
    pub fn is_moe(&self) -> bool { self.n_expert > 0 }

    // ---- launch helpers ----------------------------------------------------

    pub(crate) fn launch_rmsnorm(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void, n: u32)
        -> Result<(), String>
    {
        let f = self.m_rmsnorm.function("rmsnorm_f32")?;
        // block=512 picked from a 64/128/256/512 sweep — per-call GPU
        // dispatch overhead (~5-9 μs) dwarfs the actual compute (~130
        // ns), so a bigger WG hides the overhead better. Above 512 the
        // tree-reduction depth costs more than the saving.
        let block: u32 = 512;
        let mut xa=x; let mut wa=w; let mut ya=y; let mut na=n; let mut ea=self.rms_eps;
        let mut args: [*mut c_void; 5] = [
            &mut xa as *mut _ as *mut c_void, &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void];
        let smem = block * 4;
        unsafe { f.launch((1,1,1),(block,1,1), smem, Some(&self.stream), &mut args) }
    }

    /// Fused rmsnorm + residual add: `y[i] += rmsnorm(x)[i] * w[i]`.
    /// Replaces `launch_rmsnorm(x, w, normed) + launch_add(y, normed)` —
    /// halves the per-pair dispatch cost (one launch instead of two).
    pub(crate) fn launch_rmsnorm_add(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void, n: u32)
        -> Result<(), String>
    {
        let f = self.m_rmsnorm_add.function("rmsnorm_add_f32")?;
        let block: u32 = 512;
        let mut xa=x; let mut wa=w; let mut ya=y; let mut na=n; let mut ea=self.rms_eps;
        let mut args: [*mut c_void; 5] = [
            &mut xa as *mut _ as *mut c_void, &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void];
        let smem = block * 4;
        unsafe { f.launch((1,1,1),(block,1,1), smem, Some(&self.stream), &mut args) }
    }

    /// Fused rmsnorm + residual add + per-layer output scale:
    /// `y[i] = (y[i] + rmsnorm(x)[i] * w[i]) * scale`. Used at the FINAL
    /// rmsnorm_add of each decode block to fold `layer_output_scale` into
    /// the kernel, saving one launch per layer.
    pub(crate) fn launch_rmsnorm_add_scale(&self, x: *mut c_void, w: *mut c_void,
                                            y: *mut c_void, n: u32, scale: f32)
        -> Result<(), String>
    {
        let f = self.m_rmsnorm_add.function("rmsnorm_add_scale_f32")?;
        let block: u32 = 512;
        let mut xa=x; let mut wa=w; let mut ya=y; let mut na=n;
        let mut ea=self.rms_eps; let mut sa=scale;
        let mut args: [*mut c_void; 6] = [
            &mut xa as *mut _ as *mut c_void, &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void, &mut sa as *mut _ as *mut c_void];
        let smem = block * 4;
        unsafe { f.launch((1,1,1),(block,1,1), smem, Some(&self.stream), &mut args) }
    }

    /// Fused rmsnorm + Q8 quantize: writes int8 blocks directly from the
    /// normalized output. Replaces
    /// `launch_rmsnorm(x, w, normed) + launch_quantize_q8(normed, xq8)`.
    /// Used on the MoE decode path's expert-prep step.
    pub(crate) fn launch_rmsnorm_q8(&self, x: *mut c_void, w: *mut c_void, out: *mut c_void,
                                    n: u32) -> Result<(), String>
    {
        let f = self.m_rmsnorm_q8.function("rmsnorm_q8_f32")?;
        let block: u32 = 512;
        let mut xa=x; let mut wa=w; let mut oa=out; let mut na=n; let mut ea=self.rms_eps;
        let mut args: [*mut c_void; 5] = [
            &mut xa as *mut _ as *mut c_void, &mut wa as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void];
        let smem = block * 4;
        unsafe { f.launch((1,1,1),(block,1,1), smem, Some(&self.stream), &mut args) }
    }

    pub(crate) fn launch_rmsnorm_mh(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void,
                         n_heads: u32, head_dim: u32) -> Result<(), String>
    {
        let f = self.m_rmsnorm_mh.function("rmsnorm_multihead_f32")?;
        // Block up to min(512, head_dim) — same dispatch-overhead-hiding
        // logic as launch_rmsnorm, capped at head_dim since extra threads
        // would idle (kernel strides over head_dim per block).
        let block: u32 = if head_dim >= 512 { 512 }
                         else if head_dim >= 256 { 256 }
                         else { 128 };
        let mut xa=x; let mut wa=w; let mut ya=y;
        let mut nh=n_heads; let mut hd=head_dim; let mut ea=self.rms_eps;
        let mut args: [*mut c_void; 6] = [
            &mut xa as *mut _ as *mut c_void, &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ea as *mut _ as *mut c_void];
        let smem = block * 4;
        unsafe { f.launch((n_heads,1,1),(block,1,1), smem, Some(&self.stream), &mut args) }
    }

    pub(crate) fn launch_rope(&self, x: *mut c_void, n_heads: u32, head_dim: u32, kind: AttnKind)
        -> Result<(), String>
    {
        let f = self.m_rope.function("rope_apply_f32")?;
        let (cos, sin, rd) = match kind {
            AttnKind::Sliding => (self.rope_cos_swa.raw_ptr(), self.rope_sin_swa.raw_ptr(),
                                  self.rope_dim_swa as u32),
            AttnKind::Full    => (self.rope_cos_full.raw_ptr(), self.rope_sin_full.raw_ptr(),
                                  self.rope_dim_full as u32),
        };
        let half = rd / 2;
        let block: u32 = 64;
        let grid_x = (half + block - 1) / block;
        let mut xa=x; let mut ca=cos; let mut sa=sin;
        let mut hd=head_dim; let mut rdv=rd; let mut nh=n_heads;
        let mut p=self.d_pos.raw_ptr();
        let mut args: [*mut c_void; 7] = [
            &mut xa as *mut _ as *mut c_void, &mut ca as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut rdv as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void];
        unsafe { f.launch((grid_x, n_heads, 1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    /// Quantize a normed K/V vector to int8 and append it to the cache
    /// at `d_pos` — one f32 scale per head. grid = n_kv heads.
    fn launch_kv_write_q8(&self, src: *mut c_void, dst_q: *mut c_void, dst_s: *mut c_void,
                          n_kv: u32, head_dim: u32) -> Result<(), String>
    {
        let f = self.m_kv_write.function("kv_write_q8_f32")?;
        let mut sa=src; let mut dq=dst_q; let mut ds=dst_s;
        let mut pa=self.d_pos.raw_ptr(); let mut nk=n_kv; let mut hd=head_dim;
        let mut args: [*mut c_void; 6] = [
            &mut sa as *mut _ as *mut c_void, &mut dq as *mut _ as *mut c_void,
            &mut ds as *mut _ as *mut c_void, &mut pa as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void];
        unsafe { f.launch((n_kv,1,1),(256,1,1), 0, Some(&self.stream), &mut args) }
    }

    pub(crate) fn launch_add(&self, x: *mut c_void, y: *mut c_void, n: u32) -> Result<(), String> {
        let f = self.m_add.function("add_inplace_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut xa=x; let mut ya=y; let mut na=n;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void, &mut ya as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    pub(crate) fn launch_scale(&self, x: *mut c_void, n: u32, s: f32) -> Result<(), String> {
        let f = self.m_scale.function("scale_inplace_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut xa=x; let mut na=n; let mut sa=s;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    pub(crate) fn launch_geglu(&self, gate: *mut c_void, up: *mut c_void, out: *mut c_void, n: u32)
        -> Result<(), String>
    {
        let f = self.m_geglu.function("geglu_mul_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut g=gate; let mut u=up; let mut o=out; let mut na=n;
        let mut args: [*mut c_void; 4] = [
            &mut g as *mut _ as *mut c_void, &mut u as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    // --- Batched (grid.y = p) variants for the prefill path: one launch
    // over all P token rows instead of P single-vector launches. The
    // kernels offset by blockIdx.y · n; the per-token launch fns above
    // are the grid.y = 1 case.

    fn launch_rmsnorm_batched(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void,
                              n: u32, p: u32) -> Result<(), String>
    {
        let f = self.m_rmsnorm.function("rmsnorm_f32")?;
        let block: u32 = 256;
        let mut xa=x; let mut wa=w; let mut ya=y; let mut na=n; let mut ea=self.rms_eps;
        let mut args: [*mut c_void; 5] = [
            &mut xa as *mut _ as *mut c_void, &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void];
        unsafe { f.launch((1,p,1),(block,1,1), block*4, Some(&self.stream), &mut args) }
    }

    fn launch_rmsnorm_mh_batched(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void,
                                 n_heads: u32, head_dim: u32, p: u32) -> Result<(), String>
    {
        let f = self.m_rmsnorm_mh.function("rmsnorm_multihead_f32")?;
        let block: u32 = 256;
        let mut xa=x; let mut wa=w; let mut ya=y;
        let mut nh=n_heads; let mut hd=head_dim; let mut ea=self.rms_eps;
        let mut args: [*mut c_void; 6] = [
            &mut xa as *mut _ as *mut c_void, &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ea as *mut _ as *mut c_void];
        unsafe { f.launch((n_heads,p,1),(block,1,1), block*4, Some(&self.stream), &mut args) }
    }

    fn launch_add_batched(&self, x: *mut c_void, y: *mut c_void, n: u32, p: u32)
        -> Result<(), String>
    {
        let f = self.m_add.function("add_inplace_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut xa=x; let mut ya=y; let mut na=n;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void, &mut ya as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void];
        unsafe { f.launch((grid,p,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_scale_batched(&self, x: *mut c_void, n: u32, s: f32, p: u32)
        -> Result<(), String>
    {
        let f = self.m_scale.function("scale_inplace_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut xa=x; let mut na=n; let mut sa=s;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void];
        unsafe { f.launch((grid,p,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_geglu_batched(&self, gate: *mut c_void, up: *mut c_void, out: *mut c_void,
                            n: u32, p: u32) -> Result<(), String>
    {
        let f = self.m_geglu.function("geglu_mul_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut g=gate; let mut u=up; let mut o=out; let mut na=n;
        let mut args: [*mut c_void; 4] = [
            &mut g as *mut _ as *mut c_void, &mut u as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void];
        unsafe { f.launch((grid,p,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_softcap(&self, y: *mut c_void, n: u32) -> Result<(), String> {
        let f = self.m_softcap.function("logit_softcap_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut ya=y; let mut na=n; let mut c=self.softcap;
        let mut args: [*mut c_void; 3] = [
            &mut ya as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
            &mut c as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    /// Quantize `n_vec` contiguous f32 activations of `in_dim` elements
    /// each into int8 BlockQ8 blocks at `out`, for the dp4a matvec.
    pub(crate) fn launch_quantize_q8(&self, x: *mut c_void, out: *mut c_void,
                          in_dim: u32, n_vec: u32) -> Result<(), String> {
        let f = self.m_quantize.function("quantize_q8_f32")?;
        let mut xa = x;
        let mut oa = out;
        let mut ia = in_dim;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void];
        unsafe { f.launch(((in_dim + 255) / 256, n_vec, 1), (256, 1, 1),
                          0, Some(&self.stream), &mut args) }
    }

    pub(crate) fn launch_matvec(&self, w: &GpuMatvecTensor, x: *mut c_void, y: *mut c_void)
        -> Result<(), String>
    {
        // Quantize fp32 activation → self.xq8, then dispatch the int8
        // matvec. Use launch_matvec_xq8 when the caller has already
        // quantized once for a SHARED activation (Q/K/V/O on a single
        // post-norm vector; ffn_gate+ffn_up on a single post-norm vector).
        if w.repacked {
            self.launch_quantize_q8(x, self.xq8.raw_ptr(), w.in_dim, 1)?;
            return self.launch_matvec_xq8(w, self.xq8.raw_ptr(), y);
        }
        self.launch_matvec_raw(w.data.raw_ptr(), w.dtype, w.in_dim, w.out_dim, x, y)
    }

    /// Repacked matvec from a pre-quantized int8 activation in `xq8`.
    /// Skips the quantize step — caller is responsible for quantizing the
    /// activation ONCE before calling this for each matvec that shares the
    /// same input (Q/K/V/O on one normed vector; ffn gate+up; etc).
    pub(crate) fn launch_matvec_xq8(&self, w: &GpuMatvecTensor, xq8: *mut c_void, y: *mut c_void)
        -> Result<(), String>
    {
        assert!(w.repacked, "launch_matvec_xq8 requires a repacked weight");
        let (module, kname, grid, kblock): (&Module, &str, u32, u32) = match w.dtype {
            GgmlType::Q5_K => (&self.m_mv_q5k_repacked, "matvec_q5k_repacked_f32",
                               (w.out_dim + 7) / 8, 256),
            GgmlType::Q6_K => (&self.m_mv_q6k_repacked, "matvec_q6k_repacked_f32",
                               (w.out_dim + 7) / 8, 256),
            // Q8_0: ROWS=1 for large out_dim doubles the wavefront
            // count and sustains HBM bandwidth that ROWS=2 starves;
            // ROWS=2 stays best mid-size (see matvec_q8_0_repacked).
            GgmlType::Q8_0 if w.out_dim >= 4096 =>
                (&self.m_mv_q8_0_repacked, "matvec_q8_0_repacked_r1_f32",
                 w.out_dim, 64),
            GgmlType::Q8_0 => (&self.m_mv_q8_0_repacked, "matvec_q8_0_repacked_f32",
                               (w.out_dim + 1) / 2, 64),
            _              => (&self.m_mv_q4k_repacked, "matvec_q4k_repacked_f32",
                               (w.out_dim + 7) / 8, 256),
        };
        let f = module.function(kname)?;
        let mut wa = w.data.raw_ptr();
        let mut xa = xq8;
        let mut ya = y;
        let mut ia = w.in_dim; let mut oa = w.out_dim;
        let mut args: [*mut c_void; 5] = [
            &mut wa as *mut _ as *mut c_void, &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void];
        unsafe {
            f.launch((grid, 1, 1), (kblock, 1, 1), 0, Some(&self.stream), &mut args)
        }
    }

    /// Matvec from an explicit weight pointer — lets the MoE path point
    /// at one expert's slice of a 3D expert tensor.
    fn launch_matvec_raw(&self, w_ptr: *mut c_void, dtype: GgmlType,
                         in_dim: u32, out_dim: u32, x: *mut c_void, y: *mut c_void)
        -> Result<(), String>
    {
        let block: u32 = 64;

        // K-quants + Q8_0: int8 dp4a path — quantize the activation,
        // then matvec with v_dot4_i32_i8. Same stream, ordering implicit.
        // REINSTINCT_GEMMA_NO_DP4A forces the f32/wave64 path (A/B check).
        let dp4a = std::env::var_os("REINSTINCT_GEMMA_NO_DP4A").is_none()
            && match dtype {
                GgmlType::Q4_K => std::env::var_os("REINSTINCT_NO_DP4A_Q4").is_none(),
                GgmlType::Q5_K => std::env::var_os("REINSTINCT_NO_DP4A_Q5").is_none(),
                GgmlType::Q6_K => std::env::var_os("REINSTINCT_NO_DP4A_Q6").is_none(),
                GgmlType::Q8_0 => std::env::var_os("REINSTINCT_NO_DP4A_Q8").is_none(),
                _ => false,
            };
        if dp4a {
            self.launch_quantize_q8(x, self.xq8.raw_ptr(), in_dim, 1)?;
            // Q4_K: 256-thread workgroup (4 independent wavefronts, 8 rows);
            // others: 64-thread, 2 rows per wavefront.
            let (module, kname, rows, kblock) = match dtype {
                GgmlType::Q4_K => (&self.m_mv_q4k_dp4a,  "matvec_q4_k_dp4a_f32", 8u32, 256u32),
                GgmlType::Q5_K => (&self.m_mv_q5k_dp4a,  "matvec_q5_k_dp4a_f32", Q4K_ROWBLOCK, block),
                GgmlType::Q6_K => (&self.m_mv_q6k_dp4a,  "matvec_q6_k_dp4a_f32", Q4K_ROWBLOCK, block),
                _              => (&self.m_mv_q8_0_dp4a, "matvec_q8_0_dp4a_f32", Q4K_ROWBLOCK, block),
            };
            let f = module.function(kname)?;
            let grid = (out_dim + rows - 1) / rows;
            let mut wa = w_ptr;
            let mut xa = self.xq8.raw_ptr();
            let mut ya = y;
            let mut ia = in_dim; let mut oa = out_dim;
            let mut args: [*mut c_void; 5] = [
                &mut wa as *mut _ as *mut c_void, &mut xa as *mut _ as *mut c_void,
                &mut ya as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
                &mut oa as *mut _ as *mut c_void];
            return unsafe {
                f.launch((grid, 1, 1), (kblock, 1, 1), 0, Some(&self.stream), &mut args)
            };
        }

        // Q4/5/6_K use the row-blocked kernel; the rest the wave64 ones.
        let (module, kname, grid) = match dtype {
            GgmlType::F32    => (&self.m_mv_f32,  "matvec_f32_b256",        out_dim),
            GgmlType::Q4_K   => (&self.m_mv_q4k,  "matvec_q4_k_rowblock_f32",
                                 (out_dim + Q4K_ROWBLOCK - 1) / Q4K_ROWBLOCK),
            GgmlType::Q5_K   => (&self.m_mv_q5k,  "matvec_q5_k_rowblock_f32",
                                 (out_dim + Q4K_ROWBLOCK - 1) / Q4K_ROWBLOCK),
            GgmlType::Q6_K   => (&self.m_mv_q6k,  "matvec_q6_k_rowblock_f32",
                                 (out_dim + Q4K_ROWBLOCK - 1) / Q4K_ROWBLOCK),
            GgmlType::Q8_0   => (&self.m_mv_q8_0, "matvec_q8_0_wave64_f32", out_dim),
            GgmlType::F16    => (&self.m_mv_f16,  "matvec_f16_wave64_f32",  out_dim),
            other => return Err(format!(
                "gemma4 matvec: no kernel for {other:?} (weight shape [{in_dim}×{out_dim}]). \
                 Check the GGUF for this dtype on a matmul tensor — most likely a UD-mix \
                 type that needs its own dp4a path.", in_dim=in_dim, out_dim=out_dim)),
        };
        let f = module.function(kname)?;
        // The F32 kernel uses a 256-thread block (4 waves/row); the
        // wave64 kernels use one 64-thread wavefront per row.
        let kblock = if matches!(dtype, GgmlType::F32) { 256 } else { block };
        let mut wa=w_ptr; let mut xa=x; let mut ya=y;
        let mut ia=in_dim; let mut oa=out_dim;
        let mut args: [*mut c_void; 5] = [
            &mut wa as *mut _ as *mut c_void, &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(kblock,1,1), 0, Some(&self.stream), &mut args) }
    }

    /// Router: softmax + top-k over `n_expert` logits → expert ids and
    /// renormalised weights (device buffers `moe_ids` / `moe_weights`).
    /// Routing top-k over `n_tok` tokens. The kernel offsets logits /
    /// out_ids / out_weights by `blockIdx.x * stride` so the input
    /// `moe_logits` must be `[n_tok, n_expert]` row-major and the
    /// outputs land at `moe_ids[n_tok, n_expert_used]` and
    /// `moe_weights[n_tok, n_expert_used]`. Decode passes n_tok=1.
    fn launch_moe_topk(&self, n_tok: usize) -> Result<(), String> {
        let f = self.m_moe_topk.function("moe_topk_f32")?;
        let mut la = self.moe_logits.raw_ptr();
        let mut ne = self.n_expert as i32;
        let mut nu = self.n_expert_used as i32;
        let mut ida = self.moe_ids.raw_ptr();
        let mut wa  = self.moe_weights.raw_ptr();
        let mut args: [*mut c_void; 5] = [
            &mut la as *mut _ as *mut c_void, &mut ne as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void, &mut ida as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void];
        let block: u32 = 128;
        let smem = self.n_expert as u32 * 4;
        unsafe { f.launch((n_tok as u32,1,1),(block,1,1), smem, Some(&self.stream), &mut args) }
    }

    /// One launch covering `n_used × n_tok` routed (expert, token)
    /// pairs: grid.y = expert slot, grid.z = token. Expert IDs are read
    /// from `self.moe_ids[tok * n_used + slot]` on device. Decode passes
    /// `n_tok=1, xq_tok_stride=0`. Verify-MoE passes `n_tok=p,
    /// xq_tok_stride=in_dim/32` (one BlockQ8 sequence per token).
    /// `xq_slot_stride=0` ⇒ all slots within a token share one activation
    /// (fused gate_up). `xq_slot_stride>0` ⇒ per-slot activation (down).
    #[allow(clippy::too_many_arguments)]
    fn launch_moe_matvec(&self, dtype: GgmlType, repacked: bool,
                         slab: *mut c_void, xq: *mut c_void,
                         y: *mut c_void, in_dim: u32, out_dim: u32,
                         bytes_per_expert: u32,
                         xq_tok_stride: u32, xq_slot_stride: u32,
                         n_tok: usize) -> Result<(), String>
    {
        let nu = self.n_expert_used as u32;
        let (module, kname, block, rows): (&Module, &str, u32, u32) = if repacked {
            (&self.m_moe_mv_q6k_repacked, "moe_matvec_q6k_repacked_f32", 256, 8)
        } else {
            match dtype {
                GgmlType::Q6_K => (&self.m_moe_mv_q6k,  "moe_matvec_q6k_dp4a_f32",  64, Q4K_ROWBLOCK),
                GgmlType::Q8_0 => (&self.m_moe_mv_q8_0, "moe_matvec_q8_0_dp4a_f32", 64, Q4K_ROWBLOCK),
                other => return Err(format!(
                    "moe matvec: no kernel for expert type {other:?} \
                     (weight shape [{in_dim}×{out_dim}], bytes/expert {bpe})",
                    in_dim=in_dim, out_dim=out_dim, bpe=bytes_per_expert)),
            }
        };
        let f = module.function(kname)?;
        let grid_x = (out_dim + rows - 1) / rows;
        let mut sa=slab; let mut ida=self.moe_ids.raw_ptr(); let mut xa=xq; let mut ya=y;
        let mut ia=in_dim; let mut oa=out_dim; let mut bpe=bytes_per_expert;
        let mut tst=xq_tok_stride; let mut sst=xq_slot_stride; let mut nu_a=nu;
        let mut args: [*mut c_void; 10] = [
            &mut sa as *mut _ as *mut c_void, &mut ida as *mut _ as *mut c_void,
            &mut xa as *mut _ as *mut c_void, &mut ya as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut bpe as *mut _ as *mut c_void, &mut tst as *mut _ as *mut c_void,
            &mut sst as *mut _ as *mut c_void, &mut nu_a as *mut _ as *mut c_void];
        unsafe {
            f.launch((grid_x, nu, n_tok as u32), (block,1,1), 0,
                     Some(&self.stream), &mut args)
        }
    }

    /// Down projection over the routed experts. The down `in_dim`
    /// (= expert_ff) is small, so `launch_moe_matvec`'s lane→sub-block
    /// mapping leaves most of every wavefront idle; the row-packed Q8_0
    /// kernel keeps all threads busy. Falls back for other dtypes.
    /// Same batched signature as `launch_moe_matvec` — n_tok=1 for
    /// decode, n_tok=p for verify-MoE.
    #[allow(clippy::too_many_arguments)]
    fn launch_moe_down(&self, dtype: GgmlType, repacked: bool,
                       slab: *mut c_void, xq: *mut c_void,
                       y: *mut c_void, in_dim: u32, out_dim: u32,
                       bytes_per_expert: u32,
                       xq_tok_stride: u32, xq_slot_stride: u32,
                       n_tok: usize) -> Result<(), String>
    {
        let n_sub = in_dim >> 5;
        if dtype == GgmlType::Q8_0 && !repacked && n_sub >= 1 && n_sub <= 256 {
            let rpb = 256 / n_sub;
            let f = self.m_moe_mv_q8_0_down.function("moe_matvec_q8_0_down_f32")?;
            let grid_x = (out_dim + rpb - 1) / rpb;
            let mut sa=slab; let mut ida=self.moe_ids.raw_ptr(); let mut xa=xq; let mut ya=y;
            let mut ia=in_dim; let mut oa=out_dim;
            let mut bpe=bytes_per_expert;
            let mut tst=xq_tok_stride; let mut sst=xq_slot_stride;
            let mut nu=self.n_expert_used as u32;
            let mut args: [*mut c_void; 10] = [
                &mut sa as *mut _ as *mut c_void, &mut ida as *mut _ as *mut c_void,
                &mut xa as *mut _ as *mut c_void, &mut ya as *mut _ as *mut c_void,
                &mut ia as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
                &mut bpe as *mut _ as *mut c_void, &mut tst as *mut _ as *mut c_void,
                &mut sst as *mut _ as *mut c_void, &mut nu as *mut _ as *mut c_void];
            return unsafe {
                f.launch((grid_x, self.n_expert_used as u32, n_tok as u32), (256,1,1), 0,
                         Some(&self.stream), &mut args)
            };
        }
        self.launch_moe_matvec(dtype, repacked, slab, xq, y, in_dim, out_dim,
                               bytes_per_expert, xq_tok_stride, xq_slot_stride, n_tok)
    }

    /// Batched GeGLU over all routed experts across `n_tok` tokens:
    /// `gu` [n_tok, n_used, 2·ff_exp] → `act` [n_tok, n_used, ff_exp].
    /// The kernel just iterates the flat product, so we pass
    /// `n_slot = n_tok * n_used` and let the existing kernel handle it.
    fn launch_moe_geglu(&self, gu: *mut c_void, act: *mut c_void, n_tok: usize)
        -> Result<(), String>
    {
        let f = self.m_moe_geglu.function("moe_geglu_f32")?;
        let block: u32 = 256;
        let total = (n_tok * self.n_expert_used * self.expert_ff) as u32;
        let grid = (total + block - 1) / block;
        let mut ga=gu; let mut aa=act;
        let mut ff=self.expert_ff as u32;
        let mut ns=(n_tok * self.n_expert_used) as u32;
        let mut args: [*mut c_void; 4] = [
            &mut ga as *mut _ as *mut c_void, &mut aa as *mut _ as *mut c_void,
            &mut ff as *mut _ as *mut c_void, &mut ns as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    /// Fused MoE GeGLU + Q8 quantize. Replaces
    /// `launch_moe_geglu(gu, act) + launch_quantize_q8(act, xq8, ff, n_slot)`.
    /// Saves one launch per layer + the HBM round-trip of `act` through fp32.
    fn launch_moe_geglu_q8(&self, gu: *mut c_void, out: *mut c_void, n_slot: usize)
        -> Result<(), String>
    {
        let f = self.m_moe_geglu_q8.function("moe_geglu_q8_f32")?;
        let block: u32 = 256;
        let ff = self.expert_ff as u32;
        let n_sub = ff >> 5;
        let grid_x = (n_sub + 7) / 8;       // 8 sub-blocks per WG
        let mut ga=gu; let mut oa=out; let mut ffa=ff;
        let mut args: [*mut c_void; 3] = [
            &mut ga as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut ffa as *mut _ as *mut c_void];
        unsafe { f.launch((grid_x, n_slot as u32, 1), (block, 1, 1),
                          0, Some(&self.stream), &mut args) }
    }

    /// Weighted sum of per-expert down outputs into `out`. For decode
    /// `out` is `[hidden]` and `n_tok=1`. For verify-MoE `out` is
    /// `[n_tok, hidden]`, `experts` is `[n_tok, n_used, hidden]`, and
    /// `moe_ids/weights` are `[n_tok, n_used]`. grid.y = n_tok.
    fn launch_moe_combine(&self, experts: *mut c_void, down_exps_s: *mut c_void,
                          out: *mut c_void, n_tok: usize) -> Result<(), String>
    {
        let f = self.m_moe_combine.function("moe_combine_f32")?;
        let block: u32 = 256;
        let h = self.hidden as u32;
        let grid = (h + block - 1) / block;
        let mut ea=experts; let mut ida=self.moe_ids.raw_ptr();
        let mut wa=self.moe_weights.raw_ptr(); let mut sa=down_exps_s; let mut oa=out;
        let mut ha=h; let mut nu=self.n_expert_used as u32;
        let mut args: [*mut c_void; 7] = [
            &mut ea as *mut _ as *mut c_void, &mut ida as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void, &mut sa as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void, &mut ha as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void];
        unsafe { f.launch((grid,n_tok as u32,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    /// Counting-sort the `n_entries` routing entries (`ids` = topk expert
    /// ids, [n_tok, n_used]) by expert into `gs.perm`, with `gs.eoff`
    /// (entry offsets) and `gs.toff` (GEMM-tile offsets).
    fn launch_moe_sort(&self, gs: &MoeGroupedScratch<'_>, ids: *mut c_void,
                       n_expert: u32, n_entries: u32) -> Result<(), String>
    {
        let zero = |buf: *mut c_void, n: u32| -> Result<(), String> {
            let f = self.m_expert_sort.function("moe_sort_zero")?;
            let mut a0 = buf; let mut a1 = n;
            let mut args: [*mut c_void; 2] = [
                &mut a0 as *mut _ as *mut c_void, &mut a1 as *mut _ as *mut c_void];
            unsafe { f.launch(((n + 255) / 256, 1, 1), (256, 1, 1), 0,
                              Some(&self.stream), &mut args) }
        };
        zero(gs.count.raw_ptr(), n_expert)?;
        {
            let f = self.m_expert_sort.function("moe_sort_histogram")?;
            let mut a0 = ids; let mut a1 = gs.count.raw_ptr();
            let mut a2 = n_entries; let mut a3 = n_expert;
            let mut args: [*mut c_void; 4] = [
                &mut a0 as *mut _ as *mut c_void, &mut a1 as *mut _ as *mut c_void,
                &mut a2 as *mut _ as *mut c_void, &mut a3 as *mut _ as *mut c_void];
            unsafe { f.launch(((n_entries + 255) / 256, 1, 1), (256, 1, 1), 0,
                              Some(&self.stream), &mut args)?; }
        }
        {
            let f = self.m_expert_sort.function("moe_sort_scan")?;
            let mut a0 = gs.count.raw_ptr(); let mut a1 = gs.eoff.raw_ptr();
            let mut a2 = gs.toff.raw_ptr(); let mut a3 = n_expert; let mut a4 = MOE_GEMM_BN;
            let mut args: [*mut c_void; 5] = [
                &mut a0 as *mut _ as *mut c_void, &mut a1 as *mut _ as *mut c_void,
                &mut a2 as *mut _ as *mut c_void, &mut a3 as *mut _ as *mut c_void,
                &mut a4 as *mut _ as *mut c_void];
            unsafe { f.launch((1, 1, 1), (64, 1, 1), 0, Some(&self.stream), &mut args)?; }
        }
        zero(gs.cursor.raw_ptr(), n_expert)?;
        {
            let f = self.m_expert_sort.function("moe_sort_scatter")?;
            let mut a0 = ids; let mut a1 = gs.eoff.raw_ptr();
            let mut a2 = gs.cursor.raw_ptr(); let mut a3 = gs.perm.raw_ptr();
            let mut a4 = n_entries; let mut a5 = n_expert;
            let mut args: [*mut c_void; 6] = [
                &mut a0 as *mut _ as *mut c_void, &mut a1 as *mut _ as *mut c_void,
                &mut a2 as *mut _ as *mut c_void, &mut a3 as *mut _ as *mut c_void,
                &mut a4 as *mut _ as *mut c_void, &mut a5 as *mut _ as *mut c_void];
            unsafe { f.launch(((n_entries + 255) / 256, 1, 1), (256, 1, 1), 0,
                              Some(&self.stream), &mut args)?; }
        }
        Ok(())
    }

    /// Gather per-token gate_up activations (`xq_src`, BlockQ8 [n_tok,
    /// nsub]) into expert-sorted order `gs.g_in`. `nsub = hidden/32`.
    fn launch_moe_gather_xq(&self, gs: &MoeGroupedScratch<'_>, xq_src: *mut c_void,
                            nsub: u32, n_used: u32, n_entries: u32) -> Result<(), String>
    {
        let f = self.m_expert_sort.function("moe_gather_xq")?;
        let mut a0 = xq_src; let mut a1 = gs.perm.raw_ptr();
        let mut a2 = gs.g_in.raw_ptr(); let mut a3 = nsub;
        let mut a4 = n_used; let mut a5 = n_entries;
        let mut args: [*mut c_void; 6] = [
            &mut a0 as *mut _ as *mut c_void, &mut a1 as *mut _ as *mut c_void,
            &mut a2 as *mut _ as *mut c_void, &mut a3 as *mut _ as *mut c_void,
            &mut a4 as *mut _ as *mut c_void, &mut a5 as *mut _ as *mut c_void];
        unsafe { f.launch(((nsub + 255) / 256, n_entries, 1), (256, 1, 1), 0,
                          Some(&self.stream), &mut args) }
    }

    /// Scatter expert-sorted rows `src` back to entry [token,slot] order
    /// `dst` via `gs.perm`. `dim` = floats per row.
    fn launch_moe_scatter_rows(&self, gs: &MoeGroupedScratch<'_>, src: *mut c_void,
                               dst: *mut c_void, dim: u32, n_entries: u32)
        -> Result<(), String>
    {
        let f = self.m_expert_sort.function("moe_scatter_rows")?;
        let mut a0 = src; let mut a1 = gs.perm.raw_ptr(); let mut a2 = dst;
        let mut a3 = dim; let mut a4 = n_entries;
        let mut args: [*mut c_void; 5] = [
            &mut a0 as *mut _ as *mut c_void, &mut a1 as *mut _ as *mut c_void,
            &mut a2 as *mut _ as *mut c_void, &mut a3 as *mut _ as *mut c_void,
            &mut a4 as *mut _ as *mut c_void];
        unsafe { f.launch(((dim + 255) / 256, n_entries, 1), (256, 1, 1), 0,
                          Some(&self.stream), &mut args) }
    }

    /// One grouped-expert MMQ GEMM: `y[n_entries, out_dim]` = the
    /// expert-sorted activations `xq` · each entry's expert weight.
    fn launch_moe_grouped_gemm(&self, et: &ExpertTensor, gs: &MoeGroupedScratch<'_>,
                               xq: *mut c_void, y: *mut c_void,
                               in_dim: u32, out_dim: u32,
                               n_entries: u32, n_expert: u32) -> Result<(), String>
    {
        let (module, kname) = match et.dtype {
            GgmlType::Q6_K => (&self.m_grouped_q6k,  "mmq_gemm_q6k_grouped_f32"),
            GgmlType::Q8_0 => (&self.m_grouped_q8_0, "mmq_gemm_q8_0_grouped_f32"),
            other => return Err(format!("grouped GEMM: unsupported dtype {other:?}")),
        };
        let f = module.function(kname)?;
        let tile_ub = (n_entries + MOE_GEMM_BN - 1) / MOE_GEMM_BN + n_expert;
        let mut a0 = et.data.raw_ptr(); let mut a1 = et.bytes_per_expert as u32;
        let mut a2 = gs.eoff.raw_ptr(); let mut a3 = gs.toff.raw_ptr();
        let mut a4 = n_expert; let mut a5 = xq; let mut a6 = y;
        let mut a7 = in_dim; let mut a8 = out_dim;
        let mut args: [*mut c_void; 9] = [
            &mut a0 as *mut _ as *mut c_void, &mut a1 as *mut _ as *mut c_void,
            &mut a2 as *mut _ as *mut c_void, &mut a3 as *mut _ as *mut c_void,
            &mut a4 as *mut _ as *mut c_void, &mut a5 as *mut _ as *mut c_void,
            &mut a6 as *mut _ as *mut c_void, &mut a7 as *mut _ as *mut c_void,
            &mut a8 as *mut _ as *mut c_void];
        unsafe { f.launch(((out_dim + 63) / 64, tile_ub, 1), (256, 1, 1), 0,
                          Some(&self.stream), &mut args) }
    }

    /// Embedding lookup — the token row is read from `d_token` on device
    /// (capturable). gemma4's token_embd is Q5_K (31B) or Q8_0 (26B).
    fn launch_embed(&self, table: &GpuMatvecTensor, out: *mut c_void) -> Result<(), String> {
        let hidden = table.in_dim;   // [hidden, vocab]
        let (module, kname, threads, grid): (&Module, &str, u32, u32) = match table.dtype {
            GgmlType::Q5_K => (&self.m_embed_q5k, "embed_lookup_q5_k_f32", 256, hidden/256),
            GgmlType::Q8_0 => (&self.m_embed_q8_0, "embed_lookup_q8_0_f32", 256, (hidden + 255)/256),
            other => return Err(format!("gemma4 embed: no kernel for {other:?} \
                 (token_embd weight; needs an embed-lookup kernel for this dtype)")),
        };
        let f = module.function(kname)?;
        let mut t=table.data.raw_ptr(); let mut o=out;
        let mut row=self.d_token.raw_ptr(); let mut h=hidden;
        let mut args: [*mut c_void; 4] = [
            &mut t as *mut _ as *mut c_void, &mut o as *mut _ as *mut c_void,
            &mut row as *mut _ as *mut c_void, &mut h as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(threads,1,1), 0, Some(&self.stream), &mut args) }
    }

    /// Batched embedding lookup for prefill — one launch over all
    /// `n_tokens` token ids (resident in `tokens_dev`), writing row `r`
    /// of `out`. Replaces the per-token launch+sync embed loop.
    fn launch_embed_batched(&self, table: &GpuMatvecTensor, out: *mut c_void,
                            tokens_dev: *mut c_void, n_tokens: u32) -> Result<(), String>
    {
        let hidden = table.in_dim;
        let (module, kname, grid_x): (&Module, &str, u32) = match table.dtype {
            GgmlType::Q5_K => (&self.m_embed_q5k, "embed_lookup_q5_k_batched_f32", hidden/256),
            GgmlType::Q8_0 => (&self.m_embed_q8_0, "embed_lookup_q8_0_batched_f32",
                               (hidden + 255)/256),
            other => return Err(format!("gemma4 batched embed: no kernel for {other:?} \
                 (batched embed kernel covers Q4_K/Q5_K/Q6_K/Q8_0/F16/F32 — \
                 add the missing dtype variant if it appears in a token_embd tensor)")),
        };
        let f = module.function(kname)?;
        let mut t=table.data.raw_ptr(); let mut o=out;
        let mut row=tokens_dev; let mut h=hidden;
        let mut args: [*mut c_void; 4] = [
            &mut t as *mut _ as *mut c_void, &mut o as *mut _ as *mut c_void,
            &mut row as *mut _ as *mut c_void, &mut h as *mut _ as *mut c_void];
        unsafe { f.launch((grid_x, n_tokens, 1),(256,1,1), 0, Some(&self.stream), &mut args) }
    }

    /// Wave-parallel rotated-space SuperQuant attention. Same Q-rotate
    /// pre-pass as the `_rs` variant; the attention kernel parallelizes
    /// position dispatch across the workgroup's 4 wave64 units —
    /// per-position cooperative dequant collapses to within-wave only
    /// (no `__syncthreads` per position). Up to ~4× cold throughput
    /// on long-context workloads.
    pub(crate) fn launch_attn_superquant_wp(&self,
        q: *mut c_void,
        kv: &crate::runtime::kv_superquant::SuperQuantKvCache,
        out: *mut c_void,
        n_kv: u32, head_dim: u32) -> Result<(), String>
    {
        let n_heads = self.n_heads as u32;
        let block: u32 = 256;
        const ROT_GROUP: u32 = 128;
        const N_WAVES: u32 = 4;
        let groups_per_head = head_dim / ROT_GROUP;

        // (1) Pre-rotate Q.
        let f_rot = self.m_rotate_q_rht.function("rotate_q_rht_f32")?;
        let mut q_p   = q;
        let mut s1k_p = kv.signs1_k.raw_ptr();
        let mut s2k_p = kv.signs2_k.raw_ptr();
        let mut qr_p  = self.q_rot_scratch.raw_ptr();
        let mut nh_a  = n_heads;
        let mut hd_a  = head_dim;
        let mut rargs: [*mut c_void; 6] = [
            &mut q_p   as *mut _ as *mut c_void,
            &mut s1k_p as *mut _ as *mut c_void,
            &mut s2k_p as *mut _ as *mut c_void,
            &mut qr_p  as *mut _ as *mut c_void,
            &mut nh_a  as *mut _ as *mut c_void,
            &mut hd_a  as *mut _ as *mut c_void,
        ];
        unsafe {
            f_rot.launch((n_heads, groups_per_head, 1), (128, 1, 1), 0,
                         Some(&self.stream), &mut rargs)?;
        }

        let total_len = (kv.cold_count() + kv.warm_count()) as u32;
        if total_len == 0 { return Ok(()); }
        let n_splits = ((total_len + 255) / 256).clamp(1, ATTN_MAX_SPLITS);
        let chunk = (total_len + n_splits - 1) / n_splits;

        // LDS: q + qrot + scores + tmp + per_wave_v[4×hd] + acc_w + acc_c + fwhtw
        let smem_floats = head_dim + head_dim
                        + chunk + block
                        + N_WAVES * head_dim
                        + head_dim + head_dim
                        + ROT_GROUP;
        let smem = smem_floats * 4;

        let f = self.m_attn_superquant_wp.function("attn_partial_superquant_wp_f32")?;
        let mut q_p2  = q;
        let mut qr_p2 = self.q_rot_scratch.raw_ptr();
        let mut wk_p  = kv.warm_k.raw_ptr();
        let mut wks_p = kv.warm_ks.raw_ptr();
        let mut wv_p  = kv.warm_v.raw_ptr();
        let mut wvs_p = kv.warm_vs.raw_ptr();
        let mut ck_p  = kv.cold_k.raw_ptr();
        let mut cv_p  = kv.cold_v.raw_ptr();
        let mut s1v_p = kv.signs1_v.raw_ptr();
        let mut s2v_p = kv.signs2_v.raw_ptr();
        let mut op_p  = self.attn_o_partial.raw_ptr();
        let mut mp_p  = self.attn_m_partial.raw_ptr();
        let mut lp_p  = self.attn_l_partial.raw_ptr();
        let mut nh    = n_heads;
        let mut nkv_a = n_kv;
        let mut hd    = head_dim;
        let mut cc    = kv.cold_count() as u32;
        let mut wc    = kv.warm_count() as u32;
        let mut sc    = 1.0f32;

        let mut args: [*mut c_void; 19] = [
            &mut q_p2  as *mut _ as *mut c_void,
            &mut qr_p2 as *mut _ as *mut c_void,
            &mut wk_p  as *mut _ as *mut c_void,
            &mut wks_p as *mut _ as *mut c_void,
            &mut wv_p  as *mut _ as *mut c_void,
            &mut wvs_p as *mut _ as *mut c_void,
            &mut ck_p  as *mut _ as *mut c_void,
            &mut cv_p  as *mut _ as *mut c_void,
            &mut s1v_p as *mut _ as *mut c_void,
            &mut s2v_p as *mut _ as *mut c_void,
            &mut op_p  as *mut _ as *mut c_void,
            &mut mp_p  as *mut _ as *mut c_void,
            &mut lp_p  as *mut _ as *mut c_void,
            &mut nh    as *mut _ as *mut c_void,
            &mut nkv_a as *mut _ as *mut c_void,
            &mut hd    as *mut _ as *mut c_void,
            &mut cc    as *mut _ as *mut c_void,
            &mut wc    as *mut _ as *mut c_void,
            &mut sc    as *mut _ as *mut c_void,
        ];
        unsafe {
            f.launch((n_heads, n_splits, 1), (block, 1, 1), smem,
                     Some(&self.stream), &mut args)?;
        }

        let fm = self.m_attn_merge.function("attn_merge_f32")?;
        let mut op2 = self.attn_o_partial.raw_ptr();
        let mut mp2 = self.attn_m_partial.raw_ptr();
        let mut lp2 = self.attn_l_partial.raw_ptr();
        let mut oa  = out;
        let mut hd2 = head_dim;
        let mut ns2 = n_splits;
        let mut margs: [*mut c_void; 6] = [
            &mut op2 as *mut _ as *mut c_void,
            &mut mp2 as *mut _ as *mut c_void,
            &mut lp2 as *mut _ as *mut c_void,
            &mut oa  as *mut _ as *mut c_void,
            &mut hd2 as *mut _ as *mut c_void,
            &mut ns2 as *mut _ as *mut c_void,
        ];
        unsafe { fm.launch((n_heads, 1, 1), (block, 1, 1), 0,
                           Some(&self.stream), &mut margs) }
    }

    /// Rotated-space SuperQuant attention. Two-pass: (1) pre-rotate
    /// Q by K's RHT into `q_rot_scratch`; (2) launch the rs kernel
    /// that scores cold K in rotated space (no per-position iRHT)
    /// and accumulates V also in rotated space, applying ONE iRHT
    /// per (head, group) at the end. Same partial+merge shape as the
    /// non-rotated path.
    pub(crate) fn launch_attn_superquant_rs(&self,
        q: *mut c_void,
        kv: &crate::runtime::kv_superquant::SuperQuantKvCache,
        out: *mut c_void,
        n_kv: u32, head_dim: u32) -> Result<(), String>
    {
        let n_heads = self.n_heads as u32;
        let block: u32 = 256;
        const ROT_GROUP: u32 = 128;
        let groups_per_head = head_dim / ROT_GROUP;

        // (1) Pre-rotate Q with K's signs — grid (n_heads, groups_per_head),
        // block (128). One FWHT-128 per (head, group).
        let f_rot = self.m_rotate_q_rht.function("rotate_q_rht_f32")?;
        let mut q_p   = q;
        let mut s1k_p = kv.signs1_k.raw_ptr();
        let mut s2k_p = kv.signs2_k.raw_ptr();
        let mut qr_p  = self.q_rot_scratch.raw_ptr();
        let mut nh_a  = n_heads;
        let mut hd_a  = head_dim;
        let mut rargs: [*mut c_void; 6] = [
            &mut q_p   as *mut _ as *mut c_void,
            &mut s1k_p as *mut _ as *mut c_void,
            &mut s2k_p as *mut _ as *mut c_void,
            &mut qr_p  as *mut _ as *mut c_void,
            &mut nh_a  as *mut _ as *mut c_void,
            &mut hd_a  as *mut _ as *mut c_void,
        ];
        unsafe {
            f_rot.launch((n_heads, groups_per_head, 1), (128, 1, 1), 0,
                         Some(&self.stream), &mut rargs)?;
        }

        // (2) Rotated-space attention launch.
        let total_len = (kv.cold_count() + kv.warm_count()) as u32;
        if total_len == 0 { return Ok(()); }
        let n_splits = ((total_len + 255) / 256).clamp(1, ATTN_MAX_SPLITS);
        let chunk = (total_len + n_splits - 1) / n_splits;

        // LDS: qf32 + qrot + scores + tmp + acc_w + acc_r + dq_group + fwhtw
        let smem_floats = head_dim + head_dim + chunk + block
                        + head_dim + head_dim
                        + ROT_GROUP + ROT_GROUP;
        let smem = smem_floats * 4;

        let f = self.m_attn_superquant_rs.function("attn_partial_superquant_rs_f32")?;
        let mut q_p2  = q;
        let mut qr_p2 = self.q_rot_scratch.raw_ptr();
        let mut wk_p  = kv.warm_k.raw_ptr();
        let mut wks_p = kv.warm_ks.raw_ptr();
        let mut wv_p  = kv.warm_v.raw_ptr();
        let mut wvs_p = kv.warm_vs.raw_ptr();
        let mut ck_p  = kv.cold_k.raw_ptr();
        let mut cv_p  = kv.cold_v.raw_ptr();
        let mut s1v_p = kv.signs1_v.raw_ptr();
        let mut s2v_p = kv.signs2_v.raw_ptr();
        let mut op_p  = self.attn_o_partial.raw_ptr();
        let mut mp_p  = self.attn_m_partial.raw_ptr();
        let mut lp_p  = self.attn_l_partial.raw_ptr();
        let mut nh    = n_heads;
        let mut nkv_a = n_kv;
        let mut hd    = head_dim;
        let mut cc    = kv.cold_count() as u32;
        let mut wc    = kv.warm_count() as u32;
        let mut sc    = 1.0f32;

        let mut args: [*mut c_void; 19] = [
            &mut q_p2  as *mut _ as *mut c_void,
            &mut qr_p2 as *mut _ as *mut c_void,
            &mut wk_p  as *mut _ as *mut c_void,
            &mut wks_p as *mut _ as *mut c_void,
            &mut wv_p  as *mut _ as *mut c_void,
            &mut wvs_p as *mut _ as *mut c_void,
            &mut ck_p  as *mut _ as *mut c_void,
            &mut cv_p  as *mut _ as *mut c_void,
            &mut s1v_p as *mut _ as *mut c_void,
            &mut s2v_p as *mut _ as *mut c_void,
            &mut op_p  as *mut _ as *mut c_void,
            &mut mp_p  as *mut _ as *mut c_void,
            &mut lp_p  as *mut _ as *mut c_void,
            &mut nh    as *mut _ as *mut c_void,
            &mut nkv_a as *mut _ as *mut c_void,
            &mut hd    as *mut _ as *mut c_void,
            &mut cc    as *mut _ as *mut c_void,
            &mut wc    as *mut _ as *mut c_void,
            &mut sc    as *mut _ as *mut c_void,
        ];
        unsafe {
            f.launch((n_heads, n_splits, 1), (block, 1, 1), smem,
                     Some(&self.stream), &mut args)?;
        }

        // Merge: same kernel as the q8 + non-rs SuperQuant paths.
        let fm = self.m_attn_merge.function("attn_merge_f32")?;
        let mut op2 = self.attn_o_partial.raw_ptr();
        let mut mp2 = self.attn_m_partial.raw_ptr();
        let mut lp2 = self.attn_l_partial.raw_ptr();
        let mut oa  = out;
        let mut hd2 = head_dim;
        let mut ns2 = n_splits;
        let mut margs: [*mut c_void; 6] = [
            &mut op2 as *mut _ as *mut c_void,
            &mut mp2 as *mut _ as *mut c_void,
            &mut lp2 as *mut _ as *mut c_void,
            &mut oa  as *mut _ as *mut c_void,
            &mut hd2 as *mut _ as *mut c_void,
            &mut ns2 as *mut _ as *mut c_void,
        ];
        unsafe { fm.launch((n_heads, 1, 1), (block, 1, 1), 0,
                           Some(&self.stream), &mut margs) }
    }

    /// SuperQuant 2-tier decode attention. Same split-K shape as the
    /// q8 path; the kernel reads K/V from one of two tiers based on
    /// each cached position (Cold = turbo3, Warm = int8). Reuses
    /// `attn_o_partial` / `attn_m_partial` / `attn_l_partial` device
    /// buffers and the existing `attn_merge` kernel for the per-split
    /// combine. Caller passes the per-layer SuperQuantKvCache (the
    /// donor's, in the KV-sharing case).
    pub(crate) fn launch_attn_superquant(&self,
        q: *mut c_void,
        kv: &crate::runtime::kv_superquant::SuperQuantKvCache,
        out: *mut c_void,
        n_kv: u32, head_dim: u32) -> Result<(), String>
    {
        let n_heads = self.n_heads as u32;
        let block: u32 = 256;
        let total_len = (kv.cold_count() + kv.warm_count()) as u32;
        if total_len == 0 {
            // No populated entries — zero the output so the FFN gets a
            // clean attn_concat slot.
            // (Caller writes the first KV before invoking attention, so
            // this only fires in unusual edge cases.)
            return Ok(());
        }
        let n_splits = ((total_len + 255) / 256).clamp(1, ATTN_MAX_SPLITS);
        let chunk = (total_len + n_splits - 1) / n_splits;

        // LDS layout — must match the kernel:
        //   qf32   [head_dim] | scores [chunk] | tmp [bs]
        //   dqbuf  [head_dim] | acc_v [head_dim]
        //   dq_group [ROT_GROUP] | fwhtw [ROT_GROUP]
        const ROT_GROUP: u32 = 128;
        let smem_floats = head_dim + chunk + block
                        + head_dim + head_dim
                        + ROT_GROUP + ROT_GROUP;
        let smem = smem_floats * 4;

        let f = self.m_attn_superquant.function("attn_partial_superquant_f32")?;
        let mut q_p   = q;
        let mut wk_p  = kv.warm_k.raw_ptr();
        let mut wks_p = kv.warm_ks.raw_ptr();
        let mut wv_p  = kv.warm_v.raw_ptr();
        let mut wvs_p = kv.warm_vs.raw_ptr();
        let mut ck_p  = kv.cold_k.raw_ptr();
        let mut cv_p  = kv.cold_v.raw_ptr();
        let mut s1k_p = kv.signs1_k.raw_ptr();
        let mut s2k_p = kv.signs2_k.raw_ptr();
        let mut s1v_p = kv.signs1_v.raw_ptr();
        let mut s2v_p = kv.signs2_v.raw_ptr();
        let mut op_p  = self.attn_o_partial.raw_ptr();
        let mut mp_p  = self.attn_m_partial.raw_ptr();
        let mut lp_p  = self.attn_l_partial.raw_ptr();
        let mut nh    = n_heads;
        let mut nkv_a = n_kv;
        let mut hd    = head_dim;
        let mut cc    = kv.cold_count() as u32;
        let mut wc    = kv.warm_count() as u32;
        let mut sc    = 1.0f32;

        let mut args: [*mut c_void; 20] = [
            &mut q_p   as *mut _ as *mut c_void,
            &mut wk_p  as *mut _ as *mut c_void,
            &mut wks_p as *mut _ as *mut c_void,
            &mut wv_p  as *mut _ as *mut c_void,
            &mut wvs_p as *mut _ as *mut c_void,
            &mut ck_p  as *mut _ as *mut c_void,
            &mut cv_p  as *mut _ as *mut c_void,
            &mut s1k_p as *mut _ as *mut c_void,
            &mut s2k_p as *mut _ as *mut c_void,
            &mut s1v_p as *mut _ as *mut c_void,
            &mut s2v_p as *mut _ as *mut c_void,
            &mut op_p  as *mut _ as *mut c_void,
            &mut mp_p  as *mut _ as *mut c_void,
            &mut lp_p  as *mut _ as *mut c_void,
            &mut nh    as *mut _ as *mut c_void,
            &mut nkv_a as *mut _ as *mut c_void,
            &mut hd    as *mut _ as *mut c_void,
            &mut cc    as *mut _ as *mut c_void,
            &mut wc    as *mut _ as *mut c_void,
            &mut sc    as *mut _ as *mut c_void,
        ];
        unsafe {
            f.launch((n_heads, n_splits, 1), (block, 1, 1), smem,
                     Some(&self.stream), &mut args)?;
        }

        // Reuse the existing merge kernel.
        let fm = self.m_attn_merge.function("attn_merge_f32")?;
        let mut op2 = self.attn_o_partial.raw_ptr();
        let mut mp2 = self.attn_m_partial.raw_ptr();
        let mut lp2 = self.attn_l_partial.raw_ptr();
        let mut oa  = out;
        let mut hd2 = head_dim;
        let mut ns2 = n_splits;
        let mut margs: [*mut c_void; 6] = [
            &mut op2 as *mut _ as *mut c_void,
            &mut mp2 as *mut _ as *mut c_void,
            &mut lp2 as *mut _ as *mut c_void,
            &mut oa  as *mut _ as *mut c_void,
            &mut hd2 as *mut _ as *mut c_void,
            &mut ns2 as *mut _ as *mut c_void,
        ];
        unsafe { fm.launch((n_heads, 1, 1), (block, 1, 1), 0,
                           Some(&self.stream), &mut margs) }
    }

    /// int8-KV decode attention. FlashDecoding split-K: one block per
    /// (kv_head, split) writes a partial (m, l, o) per Q head; a merge
    /// kernel combines the splits. This keeps every CU busy at depth and
    /// shortens the serial P·V scan. `REINSTINCT_OLD_ATTN` falls back to
    /// the original single-block-per-head kernel for A/B comparison.
    pub(crate) fn launch_attn_q8(&self, q: *mut c_void, kq: *mut c_void, ks: *mut c_void,
                      vq: *mut c_void, vs: *mut c_void, out: *mut c_void,
                      n_kv: u32, head_dim: u32, window: u32) -> Result<(), String>
    {
        let n_heads = self.n_heads as u32;
        let block: u32 = 256;

        if self.use_old_attn {
            let f = self.m_attn_win.function("attn_step_q8_f32")?;
            let smem = head_dim + (self.max_seq as u32 + block) * 4;
            let mut qa=q; let mut kqa=kq; let mut ksa=ks; let mut vqa=vq; let mut vsa=vs;
            let mut oa=out;
            let mut nh=n_heads; let mut nkv=n_kv; let mut hd=head_dim;
            let mut tl=self.d_pos.raw_ptr(); let mut wn=window; let mut sc=1.0f32;
            let mut args: [*mut c_void; 12] = [
                &mut qa as *mut _ as *mut c_void, &mut kqa as *mut _ as *mut c_void,
                &mut ksa as *mut _ as *mut c_void, &mut vqa as *mut _ as *mut c_void,
                &mut vsa as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void, &mut tl as *mut _ as *mut c_void,
                &mut wn as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void];
            return unsafe { f.launch((n_heads,1,1),(block,1,1), smem,
                                     Some(&self.stream), &mut args) };
        }

        // Split the context into `n_splits` chunks: grid (n_heads,
        // n_splits) fills the CUs and shortens the serial P·V scan.
        // Depends only on max_seq ⇒ a per-model constant ⇒ graph-safe.
        let n_splits = ((self.max_seq as u32 + 255) / 256).clamp(1, ATTN_MAX_SPLITS);
        let win_max = if window > 0 { window.min(self.max_seq as u32) }
                      else { self.max_seq as u32 };
        let chunk_max = (win_max + n_splits - 1) / n_splits;
        // LDS: qi[head_dim i8] | scores[chunk_max f32] | tmp[block f32]
        let smem = head_dim + (chunk_max + block) * 4;

        // --- partial: grid (n_heads, n_splits) ---
        let fp = self.m_attn_partial.function("attn_partial_q8_f32")?;
        let mut qa=q; let mut kqa=kq; let mut ksa=ks; let mut vqa=vq; let mut vsa=vs;
        let mut op=self.attn_o_partial.raw_ptr();
        let mut mp=self.attn_m_partial.raw_ptr();
        let mut lp=self.attn_l_partial.raw_ptr();
        let mut nh=n_heads; let mut nkv=n_kv; let mut hd=head_dim;
        let mut tl=self.d_pos.raw_ptr(); let mut wn=window; let mut sc=1.0f32;
        let mut ns=n_splits;
        let mut pargs: [*mut c_void; 15] = [
            &mut qa as *mut _ as *mut c_void, &mut kqa as *mut _ as *mut c_void,
            &mut ksa as *mut _ as *mut c_void, &mut vqa as *mut _ as *mut c_void,
            &mut vsa as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut mp as *mut _ as *mut c_void, &mut lp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut tl as *mut _ as *mut c_void,
            &mut wn as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void];
        unsafe {
            fp.launch((n_heads, n_splits, 1), (block,1,1), smem, Some(&self.stream), &mut pargs)?;
        }

        // --- merge: grid (n_heads) ---
        let fm = self.m_attn_merge.function("attn_merge_f32")?;
        let mut op2=self.attn_o_partial.raw_ptr();
        let mut mp2=self.attn_m_partial.raw_ptr();
        let mut lp2=self.attn_l_partial.raw_ptr();
        let mut oa=out; let mut hd2=head_dim; let mut ns2=n_splits;
        let mut margs: [*mut c_void; 6] = [
            &mut op2 as *mut _ as *mut c_void, &mut mp2 as *mut _ as *mut c_void,
            &mut lp2 as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut hd2 as *mut _ as *mut c_void, &mut ns2 as *mut _ as *mut c_void];
        unsafe { fm.launch((n_heads,1,1),(block,1,1), 0, Some(&self.stream), &mut margs) }
    }

    /// One Gemma 4 decode step → vocab-length soft-capped logits.
    /// Stage the decode token + position into the device buffers the
    /// embed / rope / attention / KV-write kernels read.
    fn set_inputs(&self, token: u32, pos: usize) -> Result<(), String> {
        self.d_token.copy_from_host(&[token])?;
        self.d_pos.copy_from_host(&[pos as u32])?;
        Ok(())
    }

    /// `REINSTINCT_MOE_PROFILE` per-stage timer (sync-per-lap).
    fn prof_reset(&self) {
        if !self.moe_prof_on { return; }
        let _ = self.stream.synchronize();
        self.prof_mark.set(std::time::Instant::now());
    }
    fn prof_lap(&self, label: &'static str) {
        if !self.moe_prof_on { return; }
        let _ = self.stream.synchronize();
        let now = std::time::Instant::now();
        let dt  = now.duration_since(self.prof_mark.get()).as_secs_f64();
        self.prof_mark.set(now);
        let mut b = self.prof_buckets.borrow_mut();
        match b.iter_mut().find(|(l, _)| *l == label) {
            Some(e) => e.1 += dt,
            None    => b.push((label, dt)),
        }
    }
    /// Accumulated per-stage decode time (ms) — see `REINSTINCT_MOE_PROFILE`.
    pub fn moe_prof_report(&self) -> Vec<(&'static str, f64)> {
        self.prof_buckets.borrow().iter().map(|(l, t)| (*l, t * 1e3)).collect()
    }

    /// Enqueue the full forward as a pure async kernel chain on
    /// `self.stream` — no host sync, no readback. Reads `d_token` /
    /// `d_pos`, so it is identical for every token/position and can be
    /// captured once into a HIP graph.
    fn enqueue_forward(&self, state: &Gemma4GpuState, debug: bool) -> Result<(), String> {
        let h = self.hidden as u32;
        self.launch_embed(&self.token_embd, self.hidden_a.raw_ptr())?;
        self.launch_scale(self.hidden_a.raw_ptr(), h, (self.hidden as f32).sqrt())?;
        self.enqueue_ple_setup()?;
        self.prof_reset();
        for (li, block) in self.blocks.iter().enumerate() {
            self.block_forward(block, li, state)?;
            if debug {
                self.stream.synchronize()?;
                let mut xh = vec![0.0f32; self.hidden];
                self.hidden_a.copy_to_host(&mut xh)?;
                let nrm = xh.iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("decode layer {li:2} kind={:?}: |x|={nrm:.4}", block.kind);
            }
        }
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), self.output_norm.raw_ptr(),
                            self.hidden_b.raw_ptr(), h)?;
        self.launch_matvec(&self.token_embd, self.hidden_b.raw_ptr(), self.logits.raw_ptr())?;
        if self.softcap > 0.0 {
            self.launch_softcap(self.logits.raw_ptr(), self.vocab as u32)?;
        }
        Ok(())
    }

    /// Per-Layer-Embedding setup (E4B): build `ple_raw`, the
    /// `[n_embd_per_layer · n_layer]` per-layer embeddings for the
    /// current token. Mirrors llama.cpp's `build_inp_per_layer` +
    /// `project_per_layer_inputs`. No-op when the model has no PLE.
    /// Must run after `hidden_a` holds the scaled token embedding.
    fn enqueue_ple_setup(&self) -> Result<(), String> {
        let Some(pg) = &self.ple else { return Ok(()); };
        let pd = (self.n_embd_per_layer * self.blocks.len()) as u32;
        // raw per-layer token-embedding lookup, scaled by √n_embd_per_layer.
        self.launch_embed(&pg.tok_embd, self.ple_raw.raw_ptr())?;
        self.launch_scale(self.ple_raw.raw_ptr(), pd,
                          (self.n_embd_per_layer as f32).sqrt())?;
        // project the main token embedding, scaled by 1/√n_embd.
        self.launch_matvec(&pg.model_proj, self.hidden_a.raw_ptr(),
                           self.ple_proj.raw_ptr())?;
        self.launch_scale(self.ple_proj.raw_ptr(), pd,
                          1.0 / (self.hidden as f32).sqrt())?;
        // RMSNorm each layer's n_embd_per_layer slice (one "head" per layer).
        self.launch_rmsnorm_mh(self.ple_proj.raw_ptr(), pg.proj_norm.raw_ptr(),
                               self.ple_proj.raw_ptr(), self.blocks.len() as u32,
                               self.n_embd_per_layer as u32)?;
        // ple_raw = (proj + raw) · 1/√2
        self.launch_add(self.ple_raw.raw_ptr(), self.ple_proj.raw_ptr(), pd)?;
        self.launch_scale(self.ple_raw.raw_ptr(), pd, 1.0 / 2.0f32.sqrt())?;
        Ok(())
    }

    /// One decode step → vocab-length soft-capped logits.
    pub fn forward_token(&self, token: u32, state: &mut Gemma4GpuState)
        -> Result<Vec<f32>, String>
    {
        self.set_inputs(token, state.pos)?;
        self.enqueue_forward(state, std::env::var("REINSTINCT_DECODE_DEBUG").is_ok())?;
        self.stream.synchronize()?;
        state.pos += 1;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// Width of the model's residual stream — the size of one
    /// `last_hidden_state` vector (`hidden_size` in the config). The MTP
    /// drafter consumes this as its `backbone_hidden_size`.
    pub fn hidden_size(&self) -> usize { self.hidden }

    /// Pre-`output_norm` hidden state of the last forward, on the
    /// device. Stable until the next forward. The post-block buffer
    /// (`hidden_a`) isn't overwritten by the final `rmsnorm → matvec →
    /// softcap` chain, so this returns the same activation the HF
    /// reference exposes as `hidden_states[-1]` and feeds to the MTP
    /// drafter's `pre_projection`.
    /// Hidden state at the position of the last forward, POST-output-norm
    /// — what HF returns as `outputs.hidden_states[-1]`, used by the MTP
    /// drafter as its initial `h_prev` per round. `forward_token` leaves
    /// the post-norm result in `hidden_b`; `verify_forward` syncs the
    /// last verify row's post-norm into `hidden_b` too.
    pub fn last_hidden_state(&self) -> &DeviceBuf<f32> { &self.hidden_b }

    /// Embed a single token via the target's `token_embd` table, writing
    /// the raw lookup (no `√hidden` scale) into the caller-provided
    /// device buffer. Output size = `hidden_size()` floats. Used by the
    /// MTP drafter to form its `concat(target_embed(prev_token), h_prev)`
    /// pre-projection input — drafter never invokes its own input embed.
    pub fn embed_token_raw(&self, token: u32, out: *mut c_void) -> Result<(), String> {
        self.d_token.copy_from_host(&[token])?;
        self.launch_embed(&self.token_embd, out)
        // No stream.synchronize() — the only caller (MTP drafter
        // forward_step) immediately chains more kernels on the same
        // stream, which see the embed output via stream ordering. The
        // final readback in forward_step is the natural sync point.
    }

    /// Stage `pos` into the device-resident position word that rope /
    /// kv-write / attention kernels read. The MTP drafter pins this to
    /// the target's last position across an entire spec-decode round
    /// (`position_ids` is constant for shared-KV layers per the HF
    /// gemma4_assistant reference).
    pub fn set_d_pos(&self, pos: usize) -> Result<(), String> {
        self.d_pos.copy_from_host(&[pos as u32])
    }

    /// Stream the drafter shares for ordering against this target.
    pub fn stream(&self) -> &Stream { &self.stream }
    pub fn n_heads(&self) -> usize { self.n_heads }
    pub fn vocab_size(&self) -> usize { self.vocab }

    /// Highest-indexed layer that *owns* its KV cache and matches the
    /// given attention kind — the layer whose KV the MTP drafter
    /// borrows for its same-kind blocks.
    pub fn last_kv_owning_layer(&self, kind: AttnKind) -> Option<usize> {
        self.blocks.iter().enumerate().rev()
            .find(|(_, b)| b.kv_donor.is_none() && b.kind == kind)
            .map(|(i, _)| i)
    }

    /// Capture the forward as a parametric HIP graph. The graph reads
    /// `d_token` / `d_pos`, so the single captured executable serves
    /// every decode step — `forward_via_graph` just stages those two
    /// device words and replays the graph with one submission, eliding
    /// the ~1300 individual kernel launches the MI50 is bound by.
    pub fn capture_forward_graph(&self, state: &Gemma4GpuState)
        -> Result<GraphExec, String>
    {
        if state.superquant.is_some() {
            return Err("Gemma4 decode-graph capture not supported with SuperQuant \
                        — the warm-tier cascade demote uses D2D memcpys on the null \
                        stream which can't be captured. Use forward_token instead.".into());
        }
        Graph::begin_capture(&self.stream, HipStreamCaptureMode::Global)?;
        if let Err(e) = self.enqueue_forward(state, false) {
            let _ = Graph::end_capture(&self.stream);
            return Err(e);
        }
        let graph = Graph::end_capture(&self.stream)?;
        let exec = graph.instantiate()?;
        drop(graph);
        Ok(exec)
    }

    /// Replay a captured forward graph for `token` at `state.pos`.
    pub fn forward_via_graph(&self, exec: &GraphExec, token: u32, state: &mut Gemma4GpuState)
        -> Result<Vec<f32>, String>
    {
        self.set_inputs(token, state.pos)?;
        exec.launch(&self.stream)?;
        self.stream.synchronize()?;
        state.pos += 1;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// Timed forward: HIP events around embed / each block / output.
    /// Returns (logits, embed_ms, per_block_ms, output_ms).
    pub fn forward_token_timed(&self, token: u32, state: &mut Gemma4GpuState)
        -> Result<(Vec<f32>, f32, Vec<f32>, f32), String>
    {
        let h = self.hidden as u32;
        let n = self.blocks.len();
        self.set_inputs(token, state.pos)?;
        let ev: Vec<Event> = (0..n + 3).map(|_| Event::new()).collect::<Result<_, _>>()?;
        ev[0].record(&self.stream)?;
        self.launch_embed(&self.token_embd, self.hidden_a.raw_ptr())?;
        self.launch_scale(self.hidden_a.raw_ptr(), h, (self.hidden as f32).sqrt())?;
        self.enqueue_ple_setup()?;
        ev[1].record(&self.stream)?;
        for (li, block) in self.blocks.iter().enumerate() {
            self.block_forward(block, li, state)?;
            ev[li + 2].record(&self.stream)?;
        }
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), self.output_norm.raw_ptr(),
                            self.hidden_b.raw_ptr(), h)?;
        self.launch_matvec(&self.token_embd, self.hidden_b.raw_ptr(), self.logits.raw_ptr())?;
        if self.softcap > 0.0 {
            self.launch_softcap(self.logits.raw_ptr(), self.vocab as u32)?;
        }
        ev[n + 2].record(&self.stream)?;
        ev[n + 2].synchronize()?;
        state.pos += 1;

        let embed_ms = Event::elapsed_time(&ev[0], &ev[1])?;
        let mut block_ms = Vec::with_capacity(n);
        for i in 0..n { block_ms.push(Event::elapsed_time(&ev[i + 1], &ev[i + 2])?); }
        let output_ms = Event::elapsed_time(&ev[n + 1], &ev[n + 2])?;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok((out, embed_ms, block_ms, output_ms))
    }

    /// Batched prefill: process all `tokens` in one pass. Each weight is
    /// streamed once and reused across all P tokens via rocBLAS HGEMM,
    /// rather than the decode path's P sequential weight-streaming
    /// passes. Returns the last token's soft-capped logits.
    ///
    /// Handles both the dense 31B and the 26B MoE — on MoE layers the
    /// shared MLP is batched (GEMM) and the routed experts run per token.
    pub fn prefill_forward(&self, tokens: &[u32], state: &mut Gemma4GpuState)
        -> Result<Vec<f32>, String>
    {
        let p = tokens.len();
        let h = self.hidden;
        assert!(p > 0 && p <= self.max_seq, "prefill: bad token count");
        assert_eq!(state.caches.len(), self.blocks.len(), "prefill: state/model mismatch");
        // rocBLAS handle + prefill kernels were built once in new() — see
        // the prefill-context fields. Per-call grouped-MoE scratch below.
        let ne_a = self.n_expert.max(1);
        let nu_a = self.n_expert_used.max(1);
        let ff_a = self.expert_ff.max(32);
        // `cw` rows of expert-intermediate scratch — the MoE branch
        // processes the prefill in MOE_PREFILL_CHUNK-token chunks.
        let cw = p.min(MOE_PREFILL_CHUNK);
        // Pooled per-call scratch — first call at each `p` warms the pool
        // (hipMalloc through `pool_f32.take`); subsequent calls reuse and
        // can capture the kernel chain into a HIP graph (hipMalloc is
        // forbidden inside `Graph::begin_capture`).
        let moe_in_all  = self.pool_f32.take(p * h)?;
        let cur_mlp     = self.pool_f32.take(p * h)?;
        let cur_moe     = self.pool_f32.take(p * h)?;
        let xq8_moe     = self.pool_u8.take(p * (h / 32) * 40)?;
        // Per-chunk routed-expert scratch: [cw, n_used, ·].
        let pf_logits   = self.pool_f32.take(p * ne_a)?;
        let pf_gu       = self.pool_f32.take(cw * nu_a * 2 * ff_a)?;
        let pf_act      = self.pool_f32.take(cw * nu_a * ff_a)?;
        let pf_dn       = self.pool_f32.take(cw * nu_a * h)?;
        let pf_xq8_e    = self.pool_u8.take(cw * nu_a * (ff_a / 32) * 40)?;
        // Grouped-expert GEMM scratch (REINSTINCT_MOE_GROUPED path).
        let gs = MoeGroupedScratch {
            count:  self.pool_i32.take(ne_a)?,
            cursor: self.pool_i32.take(ne_a)?,
            eoff:   self.pool_i32.take(ne_a + 1)?,
            toff:   self.pool_i32.take(ne_a + 1)?,
            perm:   self.pool_i32.take(cw * nu_a)?,
            g_in:   self.pool_u8 .take(cw * nu_a * (h / 32) * 40)?,
            e_act:  self.pool_f32.take(cw * nu_a * ff_a)?,
            g_out:  self.pool_f32.take(cw * nu_a * h)?,
        };

        // --- embed P tokens → x [P, hidden] ---
        // tokens_dev is filled BEFORE begin_capture so the host→device
        // memcpy (sync, allowed only outside capture) lands the current
        // call's token ids into a pooled device buffer. The captured
        // graph then reads them via launch_embed_batched at replay.
        let tokens_dev = self.pool_u32.take(p)?;
        tokens_dev.copy_from_host(tokens)?;
        let x        = self.pool_f32.take(p * h)?;
        let normed   = self.pool_f32.take(p * h)?;
        let hu = h as u32;

        // Three-state prefill execution:
        //  1) state.prefill_graphs has this P → just replay the cached
        //     GraphExec. Pool buffers' pointers are stable across calls
        //     (LIFO + deterministic take order), so the captured
        //     pointers still hit the same physical buffers.
        //  2) pools_warm && !no_graph && cache miss → first capture for
        //     this (state, P): begin_capture, enqueue chain, end_capture,
        //     instantiate, store in state.prefill_graphs, launch.
        //  3) Otherwise → uncaptured warmup (also marks pools_warm for P).
        let trace = std::env::var_os("REINSTINCT_PREFILL_DEBUG").is_some();
        let pools_warm = self.prefill_warm_p.borrow().contains(&p);
        // SuperQuant blocks graph capture: write_step cascade does D2D
        // memcpys on the null stream which can't be captured, AND each
        // write writes to a different LDS slot (warm_count++) so the
        // captured pointer-set would go stale.
        let sq_on = state.superquant.is_some();
        let force_no_graph = trace || sq_on
                           || std::env::var_os("REINSTINCT_PREFILL_NO_GRAPH").is_some();

        // Cache hit: skip the whole kernel chain — the captured graph
        // already encodes it. Pool buffers stay reserved for the launch
        // duration (drops at fn end return them to the pool, in the same
        // LIFO order, so the next call's takes resolve to the same
        // pointers the graph expects).
        if !force_no_graph {
            if let Some(exec) = state.prefill_graphs.get(&p) {
                exec.launch(&self.stream)?;
                if self.softcap > 0.0 {
                    // The captured graph doesn't include softcap (it
                    // operates on self.logits, written by the graph,
                    // so we apply it post-launch — same as the
                    // uncaptured path's tail).
                }
                self.stream.synchronize()?;
                if self.softcap > 0.0 {
                    self.launch_softcap(self.logits.raw_ptr(), self.vocab as u32)?;
                    self.stream.synchronize()?;
                }
                let mut out = vec![0.0f32; self.vocab];
                self.logits.copy_to_host(&mut out)?;
                for c in &mut state.caches { c.len = p; }
                state.pos = p;
                // Drop tokens_dev, x, normed and the other pooled bufs
                // declared above via normal scope exit — they return to
                // the pool with stable LIFO ordering.
                drop((moe_in_all, cur_mlp, cur_moe, xq8_moe, pf_logits,
                      pf_gu, pf_act, pf_dn, pf_xq8_e, gs, tokens_dev,
                      x, normed));
                return Ok(out);
            }
        }

        let no_graph = force_no_graph || !pools_warm;
        if !no_graph {
            Graph::begin_capture(&self.stream, HipStreamCaptureMode::Global)?;
        }
        // RAII: if any of the kernel launches between here and the matching
        // `end_capture` errors out, ensure we don't leave the stream in
        // capture mode. Drop fires on error early-return.
        struct CaptureGuard<'a> { stream: &'a Stream, active: bool }
        impl Drop for CaptureGuard<'_> {
            fn drop(&mut self) { if self.active { let _ = Graph::end_capture(self.stream); } }
        }
        let mut capture_guard = CaptureGuard { stream: &self.stream, active: !no_graph };

        let es = (h as f32).sqrt();
        self.launch_embed_batched(&self.token_embd, x.raw_ptr(),
                                  tokens_dev.raw_ptr(), p as u32)?;
        self.launch_scale(x.raw_ptr(), (p * h) as u32, es)?;

        // The pooled GEMM context (modules + fp16 scratch) is resident in
        // `self.prefill_gemm`; its scratch grows on demand if P exceeds
        // the size assumed at construction.
        //
        // gemm: allocates the output via the pool, writes via matmul_into.
        // Returning a PooledBuf<'_, f32> keeps the per-call DeviceBuf::new
        // out of the captured region.
        let gemm = |w: &GpuMatvecTensor, xin: &DeviceBuf<f32>| -> Result<PooledBuf<'_, f32>, String> {
            let out = self.pool_f32.take(p * w.out_dim as usize)?;
            self.prefill_gemm.matmul_into(&self.rocblas, &self.stream, &out,
                      &w.data, w.dtype, w.repacked,
                      w.in_dim as usize, w.out_dim as usize, xin, p)?;
            Ok(out)
        };

        // --- Per-Layer-Embedding setup (E4B) ---
        // Build `ple_perm` [n_layer][P][np]: each layer's [P, np] slice
        // contiguous. Mirrors llama.cpp build_inp_per_layer +
        // project_per_layer_inputs.
        let ple_perm: Option<PooledBuf<'_, f32>> = if let Some(pg_w) = &self.ple {
            let np = self.n_embd_per_layer;
            let nl = self.blocks.len();
            let pd = (np * nl) as u32;          // per-token PLE width
            // batched lookup of the per-layer token embedding.
            let ple_raw = self.pool_f32.take(p * np * nl)?;
            self.launch_embed_batched(&pg_w.tok_embd, ple_raw.raw_ptr(),
                                      tokens_dev.raw_ptr(), p as u32)?;
            self.launch_scale(ple_raw.raw_ptr(), p as u32 * pd, (np as f32).sqrt())?;
            // project the (scaled) main embedding, normalise each slice.
            let ple_proj = gemm(&pg_w.model_proj, &x)?;
            self.launch_scale(ple_proj.raw_ptr(), p as u32 * pd, 1.0 / (h as f32).sqrt())?;
            self.launch_rmsnorm_mh_batched(ple_proj.raw_ptr(), pg_w.proj_norm.raw_ptr(),
                ple_proj.raw_ptr(), nl as u32, np as u32, p as u32)?;
            // ple_raw = (raw + proj) · 1/√2
            self.launch_add_batched(ple_raw.raw_ptr(), ple_proj.raw_ptr(), pd, p as u32)?;
            self.launch_scale(ple_raw.raw_ptr(), p as u32 * pd, 1.0 / 2.0f32.sqrt())?;
            // permute to layer-major so each layer slice is contiguous.
            let perm = self.pool_f32.take(nl * p * np)?;
            self.launch_permute_ple(&self.m_permute_pf, ple_raw.raw_ptr(), perm.raw_ptr(),
                                    p as u32, nl as u32, np as u32)?;
            Some(perm)
        } else { None };

        // KV-sharing donors: the SWA / full layers whose post-norm K/V
        // the later (shared) layers attend against.
        let sharing = self.n_layer_kv_from_start < self.blocks.len();
        let donor_swa_idx  = self.n_layer_kv_from_start.saturating_sub(2);
        let donor_full_idx = self.n_layer_kv_from_start.saturating_sub(1);
        let mut donor_swa:  Option<(PooledBuf<'_, f32>, PooledBuf<'_, f32>)> = None;
        let mut donor_full: Option<(PooledBuf<'_, f32>, PooledBuf<'_, f32>)> = None;

        let dbg = std::env::var("REINSTINCT_PREFILL_DEBUG").is_ok();
        // Per-category prefill timing (REINSTINCT_PREFILL_DEBUG). `lap`
        // syncs the stream and charges the elapsed time since the last
        // lap to a bucket — so the breakdown is serial-attributed.
        let t_gemm = std::cell::Cell::new(0.0f64);
        let t_attn = std::cell::Cell::new(0.0f64);
        let t_norm = std::cell::Cell::new(0.0f64);
        let mark = std::cell::Cell::new(std::time::Instant::now());
        let lap = |acc: &std::cell::Cell<f64>| -> Result<(), String> {
            if dbg {
                self.stream.synchronize()?;
                let now = std::time::Instant::now();
                acc.set(acc.get() + now.duration_since(mark.get()).as_secs_f64());
                mark.set(now);
            }
            Ok(())
        };
        if dbg { self.stream.synchronize()?; mark.set(std::time::Instant::now()); }
        for (li, b) in self.blocks.iter().enumerate() {
            let hd = b.head_dim;
            let n_kv = b.n_kv;
            let q_dim = self.n_heads * hd;
            let kv_dim = n_kv * hd;

            // --- attention ---
            self.launch_rmsnorm_batched(x.raw_ptr(), b.attn_norm.raw_ptr(),
                                        normed.raw_ptr(), hu, p as u32)?;
            lap(&t_norm)?;
            let q = gemm(&b.attn_q, &normed)?;
            self.launch_rmsnorm_mh_batched(q.raw_ptr(), b.attn_q_norm.raw_ptr(),
                q.raw_ptr(), self.n_heads as u32, hd as u32, p as u32)?;
            self.launch_rope_prefill(&self.m_rope_pf, q.raw_ptr(), self.n_heads as u32,
                                     hd as u32, b.kind, p)?;

            // K/V: computed on KV-owning layers; KV-sharing layers reuse
            // a donor layer's post-norm K/V (see kv_donor).
            let kv_owned: Option<(PooledBuf<'_, f32>, PooledBuf<'_, f32>)> = if b.kv_donor.is_none() {
                let k = gemm(&b.attn_k, &normed)?;
                let v_gemm;
                let v_ptr = match &b.attn_v {
                    Some(wv) => { v_gemm = gemm(wv, &normed)?; v_gemm.raw_ptr() }
                    None     => k.raw_ptr(),     // full layers: V is the K projection
                };
                lap(&t_gemm)?;
                let k_norm = self.pool_f32.take(p * kv_dim)?;
                let v_norm = self.pool_f32.take(p * kv_dim)?;
                self.launch_rmsnorm_mh_batched(k.raw_ptr(), b.attn_k_norm.raw_ptr(),
                    k_norm.raw_ptr(), n_kv as u32, hd as u32, p as u32)?;
                self.launch_rmsnorm_mh_batched(v_ptr, self.ones.raw_ptr(),
                    v_norm.raw_ptr(), n_kv as u32, hd as u32, p as u32)?;
                self.launch_rope_prefill(&self.m_rope_pf, k_norm.raw_ptr(), n_kv as u32,
                                         hd as u32, b.kind, p)?;
                // populate this layer's decode KV cache (positions 0..P-1).
                let kvc = &state.caches[li];
                self.launch_kv_quant_prefill(&self.m_kvq_pf, k_norm.raw_ptr(), kvc.k.raw_ptr(),
                                             kvc.ks.raw_ptr(), n_kv as u32, hd as u32, p)?;
                self.launch_kv_quant_prefill(&self.m_kvq_pf, v_norm.raw_ptr(), kvc.v.raw_ptr(),
                                             kvc.vs.raw_ptr(), n_kv as u32, hd as u32, p)?;
                Some((k_norm, v_norm))
            } else {
                lap(&t_gemm)?;
                None
            };
            // attention reads either this layer's K/V or the donor's.
            // Deref coercion: PooledBuf<f32> → DeviceBuf<f32>.
            let (k_attn, v_attn): (&DeviceBuf<f32>, &DeviceBuf<f32>) = match &kv_owned {
                Some((k, v)) => (k, v),
                None => {
                    let d = match b.kind {
                        AttnKind::Sliding => donor_swa.as_ref(),
                        AttnKind::Full    => donor_full.as_ref(),
                    }.ok_or("prefill: KV donor not yet computed")?;
                    (&d.0, &d.1)
                }
            };
            let window = match b.kind {
                AttnKind::Sliding => self.sliding_window as u32,
                AttnKind::Full    => 0,
            };
            lap(&t_norm)?;
            let attn = self.pool_f32.take(p * q_dim)?;
            self.launch_attn_prefill(&self.m_attn_pf, q.raw_ptr(), k_attn.raw_ptr(), v_attn.raw_ptr(),
                                     attn.raw_ptr(), n_kv as u32, hd as u32, window, p)?;
            lap(&t_attn)?;
            // hand this layer's K/V to the KV-sharing layers that reuse it.
            if sharing && li == donor_swa_idx       { donor_swa = kv_owned; }
            else if sharing && li == donor_full_idx { donor_full = kv_owned; }
            let attn_out = gemm(&b.attn_output, &attn)?;
            lap(&t_gemm)?;
            self.launch_rmsnorm_batched(attn_out.raw_ptr(), b.post_attn_norm.raw_ptr(),
                normed.raw_ptr(), hu, p as u32)?;
            self.launch_add_batched(x.raw_ptr(), normed.raw_ptr(), hu, p as u32)?;

            // --- FFN: shared MLP (GeGLU), batched ---
            // Per-block FFN width (heterogeneous on E2B, uniform elsewhere).
            let ff = b.ffn_gate.out_dim as u32;
            self.launch_rmsnorm_batched(x.raw_ptr(), b.ffn_norm.raw_ptr(),
                normed.raw_ptr(), hu, p as u32)?;
            lap(&t_norm)?;
            let gate = gemm(&b.ffn_gate, &normed)?;
            let up   = gemm(&b.ffn_up,   &normed)?;
            lap(&t_gemm)?;
            self.launch_geglu_batched(gate.raw_ptr(), up.raw_ptr(), gate.raw_ptr(),
                                      ff, p as u32)?;
            lap(&t_norm)?;
            let mlp = gemm(&b.ffn_down, &gate)?;
            lap(&t_gemm)?;

            match &b.moe {
                None => {
                    // dense layer — the shared MLP is the whole FFN.
                    self.launch_rmsnorm_batched(mlp.raw_ptr(), b.post_ffw_norm.raw_ptr(),
                        normed.raw_ptr(), hu, p as u32)?;
                    self.launch_add_batched(x.raw_ptr(), normed.raw_ptr(), hu, p as u32)?;
                    lap(&t_norm)?;
                    // --- Per-Layer Embedding (E4B): gated residual ---
                    if let (Some(perm), Some(pb)) = (&ple_perm, &b.ple) {
                        let np = self.n_embd_per_layer as u32;
                        let gate2 = gemm(&pb.inp_gate, &x)?;
                        let slice = pf_off(perm.raw_ptr(), li * p * self.n_embd_per_layer);
                        self.launch_geglu_batched(gate2.raw_ptr(), slice,
                                                  gate2.raw_ptr(), np, p as u32)?;
                        let pout = gemm(&pb.proj, &gate2)?;
                        self.launch_rmsnorm_batched(pout.raw_ptr(), pb.post_norm.raw_ptr(),
                            normed.raw_ptr(), hu, p as u32)?;
                        self.launch_add_batched(x.raw_ptr(), normed.raw_ptr(), hu, p as u32)?;
                        lap(&t_gemm)?;
                    }
                    self.launch_scale_batched(x.raw_ptr(), hu, b.layer_output_scale, p as u32)?;
                }
                Some(mw) => {
                    // Dual FFN: shared MLP (already in `mlp`) + routed
                    // experts, fully token-batched. The routed branch
                    // processes the prefill in MOE_PREFILL_CHUNK chunks
                    // so the expert-intermediate scratch stays bounded.
                    let ff_exp = self.expert_ff as u32;
                    let ne = self.n_expert;
                    let nu = self.n_expert_used;
                    // Grouped-expert GEMM for the fused gate_up + down —
                    // default-on for MoE; `REINSTINCT_MOE_NO_GROUPED=1`
                    // forces the per-token matvec fallback. The 26B's
                    // last-layer Q8_0 gate_up still falls back (the
                    // grouped path here is gated on repacked Q6_K gate_up).
                    let grouped = std::env::var_os("REINSTINCT_MOE_NO_GROUPED").is_none()
                        && mw.gate_up_exps.dtype == GgmlType::Q6_K
                        && mw.gate_up_exps.repacked;
                    // cur_mlp = post_ffw_norm_1(shared MLP); expert input
                    // = pre_ffw_norm_2(x), quantised once for all P.
                    self.launch_rmsnorm_batched(mlp.raw_ptr(), mw.post_ffw_norm_1.raw_ptr(),
                                                cur_mlp.raw_ptr(), hu, p as u32)?;
                    self.launch_rmsnorm_batched(x.raw_ptr(), mw.pre_ffw_norm_2.raw_ptr(),
                                                moe_in_all.raw_ptr(), hu, p as u32)?;
                    self.launch_quantize_q8(moe_in_all.raw_ptr(), xq8_moe.raw_ptr(),
                                            hu, p as u32)?;
                    // Router: rmsnorm(x) with pre-scaled gate_inp_s
                    // (weights folded ×1/sqrt(hidden) at load) → F32
                    // projection → [P, n_expert].
                    self.launch_rmsnorm_batched(x.raw_ptr(), mw.gate_inp_s.raw_ptr(),
                                                normed.raw_ptr(), hu, p as u32)?;
                    self.prefill_gemm.matmul_into(&self.rocblas, &self.stream, &pf_logits,
                        &mw.gate_inp.data, mw.gate_inp.dtype, mw.gate_inp.repacked,
                        mw.gate_inp.in_dim as usize, mw.gate_inp.out_dim as usize,
                        &normed, p)?;
                    lap(&t_gemm)?;
                    // Routed experts, token-batched in chunks. Each chunk's
                    // logits are staged into moe_logits[0..] so the batched
                    // topk + the grid.z-over-tokens matvecs index from row 0.
                    let mut c0 = 0;
                    while c0 < p {
                        let cn = (p - c0).min(MOE_PREFILL_CHUNK);
                        self.moe_logits.copy_range_from_device_async(
                            &pf_logits, c0 * ne, 0, cn * ne, &self.stream)?;
                        self.launch_moe_topk(cn)?;
                        let xq8_c = unsafe {
                            (xq8_moe.raw_ptr() as *mut u8).add(c0 * (h / 32) * 40) as *mut c_void
                        };
                        if grouped {
                            // Whole FFN stays expert-sorted: sort entries by
                            // expert, gather activations, one tiled GEMM over
                            // the fused gate_up, GeGLU, quantize, one tiled
                            // GEMM over the down experts, then a single
                            // scatter back to [token,slot] order.
                            let n_entries = (cn * nu) as u32;
                            self.launch_moe_sort(&gs, self.moe_ids.raw_ptr(),
                                                 ne as u32, n_entries)?;
                            self.launch_moe_gather_xq(&gs, xq8_c, (h / 32) as u32,
                                                      nu as u32, n_entries)?;
                            self.launch_moe_grouped_gemm(&mw.gate_up_exps, &gs,
                                gs.g_in.raw_ptr(), pf_gu.raw_ptr(),
                                hu, 2 * ff_exp, n_entries, ne as u32)?;
                            self.launch_moe_geglu(pf_gu.raw_ptr(), gs.e_act.raw_ptr(), cn)?;
                            self.launch_quantize_q8(gs.e_act.raw_ptr(), pf_xq8_e.raw_ptr(),
                                                    ff_exp, (cn * nu) as u32)?;
                            self.launch_moe_grouped_gemm(&mw.down_grouped, &gs,
                                pf_xq8_e.raw_ptr(), gs.g_out.raw_ptr(),
                                ff_exp, hu, n_entries, ne as u32)?;
                            self.launch_moe_scatter_rows(&gs, gs.g_out.raw_ptr(),
                                pf_dn.raw_ptr(), hu, n_entries)?;
                        } else {
                            self.launch_moe_matvec(mw.gate_up_exps.dtype,
                                mw.gate_up_exps.repacked,
                                mw.gate_up_exps.data.raw_ptr(), xq8_c, pf_gu.raw_ptr(),
                                hu, 2 * ff_exp, mw.gate_up_exps.bytes_per_expert as u32,
                                hu / 32, 0, cn)?;
                            self.launch_moe_geglu(pf_gu.raw_ptr(), pf_act.raw_ptr(), cn)?;
                            self.launch_quantize_q8(pf_act.raw_ptr(), pf_xq8_e.raw_ptr(),
                                                    ff_exp, (cn * nu) as u32)?;
                            self.launch_moe_down(mw.down_exps.dtype, mw.down_exps.repacked,
                                mw.down_exps.data.raw_ptr(), pf_xq8_e.raw_ptr(),
                                pf_dn.raw_ptr(), ff_exp, hu,
                                mw.down_exps.bytes_per_expert as u32,
                                nu as u32 * (ff_exp / 32), ff_exp / 32, cn)?;
                        }
                        self.launch_moe_combine(pf_dn.raw_ptr(), mw.down_exps_s.raw_ptr(),
                            pf_off(cur_moe.raw_ptr(), c0 * h), cn)?;
                        c0 += cn;
                    }
                    lap(&t_gemm)?;
                    // cur = post_ffw_norm_2(moe) + cur_mlp → post_ffw_norm → residual.
                    self.launch_rmsnorm_batched(cur_moe.raw_ptr(), mw.post_ffw_norm_2.raw_ptr(),
                                                cur_moe.raw_ptr(), hu, p as u32)?;
                    self.launch_add_batched(cur_mlp.raw_ptr(), cur_moe.raw_ptr(), hu, p as u32)?;
                    self.launch_rmsnorm_batched(cur_mlp.raw_ptr(), b.post_ffw_norm.raw_ptr(),
                                                normed.raw_ptr(), hu, p as u32)?;
                    self.launch_add_batched(x.raw_ptr(), normed.raw_ptr(), hu, p as u32)?;
                    self.launch_scale_batched(x.raw_ptr(), hu, b.layer_output_scale, p as u32)?;
                    lap(&t_norm)?;
                }
            }
            if dbg {
                self.stream.synchronize()?;
                let mut xh = vec![0.0f32; p * h];
                x.copy_to_host(&mut xh)?;
                let nrm = |i: usize| -> f32 {
                    xh[i*h..(i+1)*h].iter().map(|v| v*v).sum::<f32>().sqrt()
                };
                eprintln!("layer {li:2} kind={:?} hd={hd}: |x[0]|={:.4} |x[last]|={:.4}",
                          b.kind, nrm(0), nrm(p-1));
            }
            // Don't charge the diagnostic block above to the next layer.
            if dbg { mark.set(std::time::Instant::now()); }
        }

        // --- output: last token only ---
        let last = pf_off(x.raw_ptr(), (p - 1) * h);
        self.launch_rmsnorm(last, self.output_norm.raw_ptr(), self.hidden_b.raw_ptr(), hu)?;
        self.launch_matvec(&self.token_embd, self.hidden_b.raw_ptr(), self.logits.raw_ptr())?;
        lap(&t_gemm)?;
        if dbg {
            let (g, a, n) = (t_gemm.get(), t_attn.get(), t_norm.get());
            let tot = g + a + n;
            eprintln!("prefill breakdown ({p} tok): gemm {:.0} ms ({:.0}%), \
                attn {:.0} ms ({:.0}%), norm/rope/kv {:.0} ms ({:.0}%)",
                g*1e3, g/tot*100.0, a*1e3, a/tot*100.0, n*1e3, n/tot*100.0);
        }
        if self.softcap > 0.0 {
            self.launch_softcap(self.logits.raw_ptr(), self.vocab as u32)?;
        }

        // End capture (if active), instantiate, STORE in the per-state
        // graph cache (so future prefills at the same (state, P) skip
        // the capture+instantiate cost entirely), then launch.
        if !no_graph {
            capture_guard.active = false;
            let g = Graph::end_capture(&self.stream)?;
            let exec = g.instantiate()?;
            drop(g);
            exec.launch(&self.stream)?;
            state.prefill_graphs.insert(p, exec);
        } else {
            // This call warmed the pools for `p`; future calls can capture.
            self.prefill_warm_p.borrow_mut().insert(p);
        }

        self.stream.synchronize()?;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        // The KV cache now holds the P prompt tokens — decode continues
        // from position P.
        for c in &mut state.caches { c.len = p; }
        state.pos = p;
        Ok(out)
    }

    /// Incremental batched forward — process K candidate tokens at
    /// positions `[state.pos, state.pos+K)` and return K logit vectors
    /// (one per query position). The KV cache is APPENDED to (not
    /// overwritten from 0). Used by MTP spec-decode verify.
    ///
    /// Dispatch: dense targets (31B) use the prefill-style batched verify
    /// kernel chain (one launch per stage, HIP-graph-capturable). MoE
    /// targets (26B-A4B) loop the decode `forward_token` K times.
    ///
    /// A batched verify path for MoE was prototyped and benchmarked
    /// ~3.5× slower than the decode-loop on MI50: the per-WG
    /// `(out_row, slot, tok)` grid in the batched moe_matvec kernels
    /// reads each expert's weights cold for every (slot, tok) pair —
    /// adjacent toks route to different experts so there's no cross-
    /// token weight sharing. K=4 batched MoE matvec ends up costing
    /// ~4× K=1 MoE matvec, same as four sequential decodes plus extra
    /// dispatch overhead. Beating decode-loop needs a bin-by-expert
    /// MoE kernel (one WG per (out_row, expert) accumulating dots for
    /// ALL routed tokens) — see the gemma4-mtp memory file for the
    /// sketch. Until then, decode-loop is the win.
    pub fn verify_forward(&self, tokens: &[u32], state: &mut Gemma4GpuState)
        -> Result<Vec<Vec<f32>>, String>
    {
        if self.n_expert > 0 {
            return self.verify_forward_via_decode(tokens, state);
        }
        let p = tokens.len();
        self.verify_setup_host(tokens, state)?;
        self.enqueue_verify_kernels(state, p)?;
        self.verify_finish_host(state, p)
    }

    /// MoE verify: K sequential `forward_token` calls. Each call writes
    /// its token's KV at `state.pos`, advances `pos`, and returns
    /// logits for the NEXT position. After K calls `state.pos` is
    /// `base_pos + K` and `hidden_a`/`hidden_b` hold the last token's
    /// pre/post-output-norm hidden — same end-state as the batched
    /// verify_forward path. Drafter `set_h_prev_from_target` reads
    /// `hidden_b`, so the K+1th round seeds correctly.
    fn verify_forward_via_decode(&self, tokens: &[u32], state: &mut Gemma4GpuState)
        -> Result<Vec<Vec<f32>>, String>
    {
        let p = tokens.len();
        if p == 0 { return Err("verify_forward_via_decode: empty token slice".into()); }
        let base_pos = state.pos;
        if base_pos + p > self.max_seq {
            return Err(format!("verify_forward_via_decode: base_pos {base_pos} + p {p} > \
                                max_seq {}", self.max_seq));
        }
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(p);
        for &t in tokens {
            out.push(self.forward_token(t, state)?);
        }
        Ok(out)
    }

    /// Host-side prep shared by `verify_forward` (inline path) and
    /// `forward_verify_via_graph` (captured-graph path): validate args,
    /// stage `v_tokens` and `v_base_pos`. Does NOT touch the stream.
    fn verify_setup_host(&self, tokens: &[u32], state: &Gemma4GpuState)
        -> Result<(), String>
    {
        if self.ple.is_some() {
            return Err("verify_forward: PLE/E4B not wired".into());
        }
        let p = tokens.len();
        assert!(p > 0, "verify_forward: empty token slice");
        if p > self.max_verify_k {
            return Err(format!("verify_forward: p={p} > max_verify_k={}", self.max_verify_k));
        }
        let base_pos = state.pos;
        if base_pos + p > self.max_seq {
            return Err(format!("verify_forward: base_pos {base_pos} + p {p} > max_seq {}",
                               self.max_seq));
        }
        // v_tokens is sized MAX_VERIFY_K; pad with zeros past p.
        let mut t = vec![0u32; self.max_verify_k];
        for i in 0..p { t[i] = tokens[i]; }
        self.v_tokens.copy_from_host(&t)?;
        // Stage per-call base_pos into the device-resident slot the
        // _offset kernels read. Lets verify_forward run as a captured
        // graph and re-execute with different base_pos per round.
        self.v_base_pos.copy_from_host(&[base_pos as u32])?;
        Ok(())
    }

    /// Host-side teardown: sync, advance KV-cache `len` / state.pos,
    /// DMA the first p × vocab logits back. Shared by inline and
    /// captured-graph paths.
    fn verify_finish_host(&self, state: &mut Gemma4GpuState, p: usize)
        -> Result<Vec<Vec<f32>>, String>
    {
        let base_pos = state.pos;
        self.stream.synchronize()?;
        for c in &mut state.caches { c.len = base_pos + p; }
        state.pos = base_pos + p;
        let mut all = vec![0.0f32; p * self.vocab];
        self.v_logits.copy_range_to_host(&mut all, 0)?;
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(p);
        for i in 0..p {
            out.push(all[i * self.vocab..(i + 1) * self.vocab].to_vec());
        }
        Ok(out)
    }

    /// Pure kernel chain — issues every GPU op of verify_forward to
    /// `self.stream` and returns. Reads `self.v_tokens` and
    /// `self.v_base_pos` (caller staged them in `verify_setup_host`).
    /// No host syncs, no host allocations — captureable into a HIP
    /// graph that replays for every round at different base_pos.
    fn enqueue_verify_kernels(&self, state: &Gemma4GpuState, p: usize)
        -> Result<(), String>
    {
        if self.ple.is_some() {
            return Err("enqueue_verify_kernels: PLE/E4B not wired".into());
        }
        // MoE targets go through `verify_forward_via_decode` — see
        // `verify_forward`'s dispatch. Defensive bail in case a future
        // caller bypasses verify_forward and calls us directly.
        if self.n_expert > 0 {
            return Err("enqueue_verify_kernels: dense-only path (MoE uses \
                        verify_forward_via_decode)".into());
        }
        let base_pos = state.pos;
        let h = self.hidden;
        let hu = h as u32;

        // All working buffers are preallocated in `self.v_*` — sized to
        // MAX_VERIFY_K rows of the worst-case per-layer dim. Reusing
        // them across spec-decode rounds avoids ~600 hipMalloc per call
        // that would otherwise dominate the verify cost.
        let x        = &self.v_x;
        let normed   = &self.v_normed;
        let q_buf    = &self.v_q;
        let k_buf    = &self.v_k;
        let v_buf    = &self.v_v;
        let k_norm   = &self.v_k_norm;
        let v_norm   = &self.v_v_norm;
        let attn     = &self.v_attn;
        let attn_out = &self.v_attn_out;
        let gate_buf = &self.v_gate;
        let up_buf   = &self.v_up;
        let mlp_buf  = &self.v_mlp;
        let logits_all = &self.v_logits;

        // --- embed K tokens → x [K, hidden]  (with √h scale) ---
        // v_tokens / v_base_pos were staged by verify_setup_host above.
        self.launch_embed_batched(&self.token_embd, x.raw_ptr(),
                                  self.v_tokens.raw_ptr(), p as u32)?;
        self.launch_scale(x.raw_ptr(), (p * h) as u32, (h as f32).sqrt())?;

        // `matmul_into` writes directly into the caller-owned dst —
        // no DeviceBuf allocation per call (the dy_f32 alloc inside
        // the original `matmul` was the bulk of verify_forward cost).
        let gemm_into = |w: &GpuMatvecTensor, xin: &DeviceBuf<f32>, dst: &DeviceBuf<f32>|
            -> Result<(), String>
        {
            self.prefill_gemm.matmul_into(&self.rocblas, &self.stream, dst,
                                          &w.data, w.dtype, w.repacked,
                                          w.in_dim as usize, w.out_dim as usize, xin, p)
        };

        for (li, b) in self.blocks.iter().enumerate() {
            let hd = b.head_dim;
            let n_kv = b.n_kv;

            self.launch_rmsnorm_batched(x.raw_ptr(), b.attn_norm.raw_ptr(),
                                        normed.raw_ptr(), hu, p as u32)?;
            gemm_into(&b.attn_q, normed, q_buf)?;
            self.launch_rmsnorm_mh_batched(q_buf.raw_ptr(), b.attn_q_norm.raw_ptr(),
                q_buf.raw_ptr(), self.n_heads as u32, hd as u32, p as u32)?;
            self.launch_rope_batched_offset(q_buf.raw_ptr(), self.n_heads as u32, hd as u32,
                                            b.kind, p)?;

            if b.kv_donor.is_some() {
                return Err(format!("verify_forward: layer {li} is KV-sharing, not supported"));
            }
            gemm_into(&b.attn_k, normed, k_buf)?;
            let v_ptr = match &b.attn_v {
                Some(wv) => { gemm_into(wv, normed, v_buf)?; v_buf.raw_ptr() }
                None     => k_buf.raw_ptr(),       // full layers: V is the K projection
            };
            self.launch_rmsnorm_mh_batched(k_buf.raw_ptr(), b.attn_k_norm.raw_ptr(),
                k_norm.raw_ptr(), n_kv as u32, hd as u32, p as u32)?;
            self.launch_rmsnorm_mh_batched(v_ptr, self.ones.raw_ptr(),
                v_norm.raw_ptr(), n_kv as u32, hd as u32, p as u32)?;
            self.launch_rope_batched_offset(k_norm.raw_ptr(), n_kv as u32, hd as u32,
                                            b.kind, p)?;

            let kvc = &state.caches[li];
            // Pass slot-0 KV base pointers; the *offset* kernel reads
            // base_pos from `self.v_base_pos` and computes the per-call
            // slot internally — same write pattern as the old code
            // (host-resolved dst+offset) but graph-safe.
            self.launch_kv_quant_prefill_offset(k_norm.raw_ptr(),
                                                 kvc.k.raw_ptr(), kvc.ks.raw_ptr(),
                                                 n_kv as u32, hd as u32, p)?;
            self.launch_kv_quant_prefill_offset(v_norm.raw_ptr(),
                                                 kvc.v.raw_ptr(), kvc.vs.raw_ptr(),
                                                 n_kv as u32, hd as u32, p)?;

            let window = match b.kind {
                AttnKind::Sliding => self.sliding_window as u32,
                AttnKind::Full    => 0,
            };
            // Pass the current base_pos as the LDS-sizing upper bound
            // for the captured launch — the kernel itself reads base_pos
            // from v_base_pos. Any later replay must have base_pos in
            // the same magnitude range (which the spec-decode loop
            // satisfies since rounds advance monotonically by ≤K+1).
            self.launch_attn_step_q8_batched_offset(
                q_buf.raw_ptr(),
                kvc.k.raw_ptr(),  kvc.ks.raw_ptr(),
                kvc.v.raw_ptr(),  kvc.vs.raw_ptr(),
                attn.raw_ptr(),
                n_kv as u32, hd as u32,
                base_pos as u32, p as u32, window)?;

            gemm_into(&b.attn_output, attn, attn_out)?;
            self.launch_rmsnorm_batched(attn_out.raw_ptr(), b.post_attn_norm.raw_ptr(),
                normed.raw_ptr(), hu, p as u32)?;
            self.launch_add_batched(x.raw_ptr(), normed.raw_ptr(), hu, p as u32)?;

            // Per-block FFN width — E2B has heterogeneous layers (e.g. 6144
            // for first 15, 12288 for the rest). On uniform models this is
            // just self.ffn.
            let ff = b.ffn_gate.out_dim as u32;
            self.launch_rmsnorm_batched(x.raw_ptr(), b.ffn_norm.raw_ptr(),
                normed.raw_ptr(), hu, p as u32)?;
            gemm_into(&b.ffn_gate, normed, gate_buf)?;
            gemm_into(&b.ffn_up,   normed, up_buf)?;
            self.launch_geglu_batched(gate_buf.raw_ptr(), up_buf.raw_ptr(),
                                      gate_buf.raw_ptr(), ff, p as u32)?;
            gemm_into(&b.ffn_down, gate_buf, mlp_buf)?;

            // Dense FFN: shared MLP is the whole feed-forward. MoE
            // targets early-returned at the top of this function.
            self.launch_rmsnorm_batched(mlp_buf.raw_ptr(), b.post_ffw_norm.raw_ptr(),
                normed.raw_ptr(), hu, p as u32)?;
            self.launch_add_batched(x.raw_ptr(), normed.raw_ptr(), hu, p as u32)?;
            self.launch_scale_batched(x.raw_ptr(), hu, b.layer_output_scale, p as u32)?;
        }

        // --- output norm + tied vocab head ---
        // token_embd is [hidden, vocab=262144] — too big for the
        // prefill_gemm scratch path. For Q5_K (31B) we have a batched
        // dp4a kernel that handles all K input rows in one launch
        // (saves ~6 ms / verify by reading the weight matrix once
        // instead of K times); otherwise fall back to K serial calls.
        self.launch_rmsnorm_batched(x.raw_ptr(), self.output_norm.raw_ptr(),
                                    normed.raw_ptr(), hu, p as u32)?;
        if self.token_embd.dtype == GgmlType::Q5_K && p >= 1 && p <= 4 {
            self.launch_lm_head_q5k_batched(normed.raw_ptr(), logits_all.raw_ptr(),
                                            p as u32)?;
        } else {
            for i in 0..p {
                let in_off  = (i * h) * 4;
                let out_off = (i * self.vocab) * 4;
                let in_ptr  = unsafe {
                    (normed.raw_ptr()     as *mut u8).add(in_off)  as *mut c_void };
                let out_ptr = unsafe {
                    (logits_all.raw_ptr() as *mut u8).add(out_off) as *mut c_void };
                self.launch_matvec(&self.token_embd, in_ptr, out_ptr)?;
            }
        }
        if self.softcap > 0.0 {
            self.launch_softcap(logits_all.raw_ptr(), (p * self.vocab) as u32)?;
        }

        // Keep `self.hidden_a` (pre-output-norm) in sync with what a
        // forward_token of the last accepted token would have left
        // (drafter rounds read it).
        // Async stream-ordered D2D — captureable into a HIP graph.
        self.hidden_a.copy_range_from_device_async(x, (p - 1) * h, 0, h, &self.stream)?;
        // Also keep `self.hidden_b` (post-output-norm) in sync — `normed`
        // already holds the per-row post-output-norm of `x`, so its last
        // row is what `forward_token` would leave in `hidden_b`. The MTP
        // drafter reads `hidden_b` (= POST-norm) as its initial h_prev
        // per HF spec — see `last_hidden_state()`.
        self.hidden_b.copy_range_from_device_async(normed, (p - 1) * h, 0, h, &self.stream)?;
        let _ = (base_pos, logits_all);   // silence unused warnings post-extract
        Ok(())
    }

    /// Capture `enqueue_verify_kernels` as a HIP graph at a SPECIFIC
    /// K. The captured graph reads `v_tokens` and `v_base_pos` on
    /// every replay, so one capture covers every spec-decode round at
    /// that K — only the host-side staging in `verify_setup_host` and
    /// the readback in `verify_finish_host` differ per call.
    ///
    /// Dense-only — MoE targets dispatch via `verify_forward_via_decode`
    /// (the batched-MoE verify path benchmarked ~3.5× slower than K
    /// sequential `forward_token` calls on MI50; see the gemma4-mtp
    /// memory file).
    pub fn capture_verify_graph(&self, state: &Gemma4GpuState, k: usize)
        -> Result<GraphExec, String>
    {
        if self.is_moe() {
            return Err("capture_verify_graph: MoE targets dispatch via \
                        verify_forward_via_decode (use is_moe() to skip)".into());
        }
        if k == 0 || k > self.max_verify_k {
            return Err(format!("capture_verify_graph: k={k} out of 1..={}",
                               self.max_verify_k));
        }
        // Capture-time placeholders: v_base_pos / v_tokens just need
        // valid bytes for the kernels to read; the values don't affect
        // the captured graph's structure. (Use state.pos so the LDS
        // sizing in attn_step_q8_batched_offset matches replay-time
        // base_pos magnitudes — see launch_attn_step_q8_batched_offset.)
        self.v_base_pos.copy_from_host(&[state.pos as u32])?;
        let zeros = vec![0u32; self.max_verify_k];
        self.v_tokens.copy_from_host(&zeros)?;

        Graph::begin_capture(&self.stream, HipStreamCaptureMode::Global)?;
        if let Err(e) = self.enqueue_verify_kernels(state, k) {
            let _ = Graph::end_capture(&self.stream);
            return Err(e);
        }
        let graph = Graph::end_capture(&self.stream)?;
        let exec = graph.instantiate()?;
        drop(graph);
        Ok(exec)
    }

    /// Replay a captured verify graph with new drafted tokens. Same
    /// host setup/teardown as `verify_forward` but the kernel chain
    /// runs as one `hipGraphLaunch` instead of ~1600 individual
    /// kernel launches. K must match the value used at capture time
    /// (the graph is K-specific).
    pub fn forward_verify_via_graph(&self, exec: &GraphExec, captured_k: usize,
                                     tokens: &[u32], state: &mut Gemma4GpuState)
        -> Result<Vec<Vec<f32>>, String>
    {
        let p = tokens.len();
        if p != captured_k {
            return Err(format!(
                "forward_verify_via_graph: K={p} ≠ captured K={captured_k}"));
        }
        self.verify_setup_host(tokens, state)?;
        exec.launch(&self.stream)?;
        self.verify_finish_host(state, p)
    }

    /// Batched (K=2..4) Q5_K dp4a matvec for verify_forward's lm_head.
    /// Quantizes K input rows of `normed` to `self.xq8`, then launches one
    /// batched kernel that reads each weight superblock once and dots it
    /// against all K rows. Saves K-1 weight-stream reads of the 880 MB
    /// token_embd matrix vs K separate matvec calls.
    fn launch_lm_head_q5k_batched(&self, in_ptr: *mut c_void, out_ptr: *mut c_void,
                                   p: u32) -> Result<(), String>
    {
        debug_assert!(self.token_embd.dtype == GgmlType::Q5_K);
        debug_assert!(p >= 1 && p <= 4, "batched lm_head supports K=1..4");
        let in_dim = self.token_embd.in_dim;     // hidden (5376 for 31B)
        let out_dim = self.token_embd.out_dim;   // vocab (262144)
        // 1) Quantize K input rows → BlockQ8 [K, in_dim/32] in self.xq8.
        self.launch_quantize_q8(in_ptr, self.xq8.raw_ptr(), in_dim, p)?;
        // 2) Batched matvec. Grid = ceil(out_dim / ROWS=2) — matches
        //    the K=1 dp4a layout for fair per-row work.
        let f = self.m_mv_q5k_dp4a_batched.function("matvec_q5_k_dp4a_batched_f32")?;
        let mut wp = self.token_embd.data.raw_ptr();
        let mut xp = self.xq8.raw_ptr();
        let mut yp = out_ptr;
        let mut ia = in_dim; let mut oa = out_dim; let mut nr = p;
        let mut args: [*mut c_void; 6] = [
            &mut wp as *mut _ as *mut c_void, &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void, &mut nr as *mut _ as *mut c_void];
        let grid = (out_dim + 1) / 2;   // ROWS=2 in the kernel
        unsafe { f.launch((grid, 1, 1), (64, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    /// Batched per-(token,head) int8 quantization of a prefill K or V
    /// tensor straight into the decode KV cache — grid (n_kv, P).
    fn launch_kv_quant_prefill(&self, m: &Module, src: *mut c_void, dst_q: *mut c_void,
                               dst_s: *mut c_void, n_kv: u32, head_dim: u32, p: usize)
        -> Result<(), String>
    {
        let f = m.function("kv_quant_prefill_f32")?;
        let mut sa=src; let mut dq=dst_q; let mut ds=dst_s;
        let mut nk=n_kv; let mut hd=head_dim;
        let mut args: [*mut c_void; 5] = [
            &mut sa as *mut _ as *mut c_void, &mut dq as *mut _ as *mut c_void,
            &mut ds as *mut _ as *mut c_void, &mut nk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void];
        unsafe { f.launch((n_kv, p as u32, 1),(256,1,1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_rope_prefill(&self, m: &Module, x: *mut c_void, n_heads: u32, head_dim: u32,
                           kind: AttnKind, p: usize) -> Result<(), String>
    {
        let f = m.function("rope_prefill_f32")?;
        let (cos, sin, rd) = match kind {
            AttnKind::Sliding => (self.rope_cos_swa.raw_ptr(), self.rope_sin_swa.raw_ptr(),
                                  self.rope_dim_swa as u32),
            AttnKind::Full    => (self.rope_cos_full.raw_ptr(), self.rope_sin_full.raw_ptr(),
                                  self.rope_dim_full as u32),
        };
        let block: u32 = 64;
        let grid_x = ((rd / 2) + block - 1) / block;
        let mut xa=x; let mut ca=cos; let mut sa=sin;
        let mut hd=head_dim; let mut rdv=rd; let mut nh=n_heads;
        let mut args: [*mut c_void; 6] = [
            &mut xa as *mut _ as *mut c_void, &mut ca as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut rdv as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void];
        unsafe { f.launch((grid_x, n_heads, p as u32),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    /// `rope_apply_batched_f32`: same half-split rotation as the decode
    /// `rope.cpp`, but with an explicit `base_pos` so the K rows at
    /// `[base_pos, base_pos+p)` rotate at their absolute sequence
    /// positions. Used by `verify_forward` when we batch process
    /// K candidate tokens starting from a non-zero state.pos.
    // ===== Offset-variant launchers used by verify_forward =====
    //
    // The three kernels below (kv_quant_prefill_offset_f32, rope_apply_
    // batched_offset_f32, attn_step_q8_batched_offset_f32) read base_pos
    // from a device-resident uint32 (`self.v_base_pos`) instead of taking
    // it as a launch-time kernel argument. This lets verify_forward run
    // either inline OR as a captured HIP graph that's replayed for every
    // round (different base_pos per call) without re-capturing.

    fn launch_kv_quant_prefill_offset(&self, src: *mut c_void,
                                       dst_q_base: *mut c_void, dst_s_base: *mut c_void,
                                       n_kv: u32, head_dim: u32, p: usize)
        -> Result<(), String>
    {
        let f = self.m_kvq_pf.function("kv_quant_prefill_offset_f32")?;
        let mut sa = src; let mut dqb = dst_q_base; let mut dsb = dst_s_base;
        let mut bp = self.v_base_pos.raw_ptr();
        let mut nk = n_kv; let mut hd = head_dim;
        let mut args: [*mut c_void; 6] = [
            &mut sa  as *mut _ as *mut c_void, &mut dqb as *mut _ as *mut c_void,
            &mut dsb as *mut _ as *mut c_void, &mut bp  as *mut _ as *mut c_void,
            &mut nk  as *mut _ as *mut c_void, &mut hd  as *mut _ as *mut c_void];
        unsafe { f.launch((n_kv, p as u32, 1),(256,1,1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_rope_batched_offset(&self, x: *mut c_void, n_heads: u32, head_dim: u32,
                                   kind: AttnKind, p: usize) -> Result<(), String>
    {
        let f = self.m_rope_b.function("rope_apply_batched_offset_f32")?;
        let (cos, sin, rd) = match kind {
            AttnKind::Sliding => (self.rope_cos_swa.raw_ptr(), self.rope_sin_swa.raw_ptr(),
                                  self.rope_dim_swa as u32),
            AttnKind::Full    => (self.rope_cos_full.raw_ptr(), self.rope_sin_full.raw_ptr(),
                                  self.rope_dim_full as u32),
        };
        let block: u32 = 64;
        let grid_x = ((rd / 2) + block - 1) / block;
        let mut xa = x; let mut ca = cos; let mut sa = sin;
        let mut hd = head_dim; let mut rdv = rd; let mut nh = n_heads;
        let mut bp = self.v_base_pos.raw_ptr();
        let mut args: [*mut c_void; 7] = [
            &mut xa  as *mut _ as *mut c_void, &mut ca  as *mut _ as *mut c_void,
            &mut sa  as *mut _ as *mut c_void, &mut hd  as *mut _ as *mut c_void,
            &mut rdv as *mut _ as *mut c_void, &mut nh  as *mut _ as *mut c_void,
            &mut bp  as *mut _ as *mut c_void];
        unsafe { f.launch((grid_x, n_heads, p as u32),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_attn_step_q8_batched_offset(&self,
        q: *mut c_void,
        k_cache: *mut c_void, k_scale: *mut c_void,
        v_cache: *mut c_void, v_scale: *mut c_void,
        out: *mut c_void,
        n_kv: u32, head_dim: u32,
        max_base_pos: u32, n_q_rows: u32,
        window: u32) -> Result<(), String>
    {
        let f = self.m_attn_step_q8_b.function("attn_step_q8_batched_offset_f32")?;
        let n_heads = self.n_heads as u32;
        let block: u32 = 256;
        // LDS sized to the worst-case window — `max_base_pos` is a host-
        // side upper bound on the base_pos value we'll see during this
        // capture's lifetime (passed in so the LDS size is captured
        // correctly; if a future caller exceeds it the kernel would OOB
        // its scores buffer).
        let max_win = if window > 0 { window.min(max_base_pos + n_q_rows) }
                      else { max_base_pos + n_q_rows };
        let smem = head_dim + (max_win + block) * 4;
        let scaling: f32 = 1.0f32;

        let mut qa = q; let mut kca = k_cache; let mut ksa = k_scale;
        let mut vca = v_cache; let mut vsa = v_scale; let mut oa = out;
        let mut nh = n_heads; let mut nkv = n_kv; let mut hd = head_dim;
        let mut bp = self.v_base_pos.raw_ptr();
        let mut nq = n_q_rows; let mut wn = window; let mut sc = scaling;
        let mut args: [*mut c_void; 13] = [
            &mut qa  as *mut _ as *mut c_void, &mut kca as *mut _ as *mut c_void,
            &mut ksa as *mut _ as *mut c_void, &mut vca as *mut _ as *mut c_void,
            &mut vsa as *mut _ as *mut c_void, &mut oa  as *mut _ as *mut c_void,
            &mut nh  as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd  as *mut _ as *mut c_void, &mut bp  as *mut _ as *mut c_void,
            &mut nq  as *mut _ as *mut c_void, &mut wn  as *mut _ as *mut c_void,
            &mut sc  as *mut _ as *mut c_void];
        unsafe { f.launch((n_heads, n_q_rows, 1),(block,1,1), smem,
                           Some(&self.stream), &mut args) }
    }

    fn launch_attn_prefill(&self, m: &Module, q: *mut c_void, k: *mut c_void, v: *mut c_void,
                           out: *mut c_void, n_kv: u32, head_dim: u32, window: u32, p: usize)
        -> Result<(), String>
    {
        // Flash-attention prefill: BQ=8 queries/workgroup (one wavefront
        // each), BK=8-key tiles staged in LDS. Must match the kernel's
        // #define BQ / BK.
        const BQ: u32 = 8;
        const BK: u32 = 8;
        let f = m.function("attn_prefill_flash_f32")?;
        let block: u32 = 64 * BQ;
        let smem = 2 * BK * head_dim * 4;
        let mut qa=q; let mut ka=k; let mut va=v; let mut oa=out;
        let mut nh=self.n_heads as u32; let mut nkv=n_kv; let mut hd=head_dim;
        let mut wn=window; let mut sc=1.0f32; let mut pr=p as u32; let mut bp=0u32;
        let mut args: [*mut c_void; 11] = [
            &mut qa as *mut _ as *mut c_void, &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut wn as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void, &mut pr as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void];
        unsafe { f.launch((self.n_heads as u32, (p as u32 + BQ - 1) / BQ, 1),
                          (block,1,1), smem, Some(&self.stream), &mut args) }
    }

    /// Permute the per-layer-embedding tensor from token-major
    /// `[P][n_layer][np]` to layer-major `[n_layer][P][np]`.
    fn launch_permute_ple(&self, m: &Module, src: *mut c_void, dst: *mut c_void,
                          p: u32, n_layer: u32, np: u32) -> Result<(), String>
    {
        let f = m.function("permute_ple_f32")?;
        let block: u32 = 256;
        let mut sa=src; let mut da=dst; let mut pa=p; let mut nl=n_layer; let mut npa=np;
        let mut args: [*mut c_void; 5] = [
            &mut sa as *mut _ as *mut c_void, &mut da as *mut _ as *mut c_void,
            &mut pa as *mut _ as *mut c_void, &mut nl as *mut _ as *mut c_void,
            &mut npa as *mut _ as *mut c_void];
        unsafe { f.launch(((np + block - 1) / block, p, n_layer), (block,1,1),
                          0, Some(&self.stream), &mut args) }
    }

    /// One transformer block, in place on `hidden_a`. All position-
    /// dependent work (rope, KV write, attention) reads `d_pos`, so the
    /// chain is identical for every decode step.
    fn block_forward(&self, b: &GpuGemma4Block, li: usize, state: &Gemma4GpuState)
        -> Result<(), String>
    {
        let h = self.hidden as u32;
        let head_dim = b.head_dim;
        let n_kv = b.n_kv;

        // KV-sharing: a `kv_donor` layer computes only Q and attends
        // against the donor's cache. Otherwise it owns `caches[li]`.
        let own_kv = &state.caches[li];
        let attn_kv = match b.kv_donor {
            Some(d) => &state.caches[d],
            None    => own_kv,
        };
        // SuperQuant routing — when state was built via
        // new_with_superquant, we read/write through the per-layer
        // SuperQuantKvCache instead of own_kv/attn_kv. Computed once
        // for use below.
        let sq_caches = state.superquant.as_ref();
        let sq_own = sq_caches.map(|v| &v[li]);
        let sq_attn = sq_caches.map(|v| match b.kv_donor {
            Some(d) => &v[d],
            None    => &v[li],
        });

        // --- Attention ---
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), b.attn_norm.raw_ptr(),
                            self.normed.raw_ptr(), h)?;
        self.prof_lap("a_norm");
        // Quantize the shared post-norm activation ONCE — Q, K, V all
        // read the same `normed`. (launch_matvec would re-quantize per
        // call; that's 2-3 redundant quantize launches per layer.)
        let need_kv = b.kv_donor.is_none();
        let qkv_repacked = b.attn_q.repacked
            && (!need_kv || (b.attn_k.repacked
                && b.attn_v.as_ref().map(|v| v.repacked).unwrap_or(true)));
        if qkv_repacked {
            self.launch_quantize_q8(self.normed.raw_ptr(), self.xq8.raw_ptr(),
                                    b.attn_q.in_dim, 1)?;
            self.launch_matvec_xq8(&b.attn_q, self.xq8.raw_ptr(), self.q_buf.raw_ptr())?;
        } else {
            self.launch_matvec(&b.attn_q, self.normed.raw_ptr(), self.q_buf.raw_ptr())?;
        }
        self.launch_rmsnorm_mh(self.q_buf.raw_ptr(), b.attn_q_norm.raw_ptr(),
                               self.q_buf.raw_ptr(), self.n_heads as u32, head_dim as u32)?;
        self.launch_rope(self.q_buf.raw_ptr(), self.n_heads as u32, head_dim as u32,
                         b.kind)?;
        self.prof_lap("a_q_proj");
        // K/V: computed and written to the cache only on KV-owning layers.
        if need_kv {
            if qkv_repacked {
                self.launch_matvec_xq8(&b.attn_k, self.xq8.raw_ptr(), self.k_proj.raw_ptr())?;
            } else {
                self.launch_matvec(&b.attn_k, self.normed.raw_ptr(), self.k_proj.raw_ptr())?;
            }
            let v_src = match &b.attn_v {
                Some(wv) => {
                    if qkv_repacked {
                        self.launch_matvec_xq8(wv, self.xq8.raw_ptr(), self.v_norm.raw_ptr())?;
                    } else {
                        self.launch_matvec(wv, self.normed.raw_ptr(), self.v_norm.raw_ptr())?;
                    }
                    self.v_norm.raw_ptr()  // temp holding the raw V projection
                }
                None => self.k_proj.raw_ptr(),  // full layers: V is the K projection
            };
            // K: per-head weighted norm + RoPE.
            self.launch_rmsnorm_mh(self.k_proj.raw_ptr(), b.attn_k_norm.raw_ptr(),
                                   self.k_norm.raw_ptr(), n_kv as u32, head_dim as u32)?;
            self.launch_rope(self.k_norm.raw_ptr(), n_kv as u32, head_dim as u32,
                             b.kind)?;
            // V: per-head plain RMSNorm (ones weight). Reads v_src, writes v_norm.
            self.launch_rmsnorm_mh(v_src, self.ones.raw_ptr(), self.v_norm.raw_ptr(),
                                   n_kv as u32, head_dim as u32)?;
            self.prof_lap("a_kv_proj");
            // Quantize (k, v) and append at d_pos. SuperQuant uses its
            // own internal pos (warm_count); standard int8 uses d_pos.
            if let Some(sq) = sq_own {
                sq.write_step(&self.kernel_cache,
                              self.k_norm.raw_ptr(), self.v_norm.raw_ptr())?;
            } else {
                self.launch_kv_write_q8(self.k_norm.raw_ptr(), own_kv.k.raw_ptr(),
                                        own_kv.ks.raw_ptr(), n_kv as u32, head_dim as u32)?;
                self.launch_kv_write_q8(self.v_norm.raw_ptr(), own_kv.v.raw_ptr(),
                                        own_kv.vs.raw_ptr(), n_kv as u32, head_dim as u32)?;
            }
            self.prof_lap("a_kv_write");
        }
        let window = match b.kind {
            AttnKind::Sliding => self.sliding_window as u32,
            AttnKind::Full    => 0,
        };
        if let Some(sq) = sq_attn {
            // SuperQuant attention; `window` arg is dropped (sliding-window
            // not yet supported under SuperQuant — see commit log).
            //
            // Default to the rotated-space (rs) variant — same output as
            // the naive path but cuts cold-tier latency 3-5× by skipping
            // the per-position iRHT. Opt out with REINSTINCT_KV_SUPERQUANT_NAIVE=1
            // for A/B comparison.
            let _ = window;
            // Default: wave-parallel rotated-space (_wp). Opt out:
            //   REINSTINCT_KV_SUPERQUANT_RS=1    → single-wave rotated-space
            //   REINSTINCT_KV_SUPERQUANT_NAIVE=1 → naive per-position iRHT
            // All three produce the same output (orthonormal rotation +
            // wave-disjoint position dispatch); they differ in cold-tier
            // latency.
            if std::env::var_os("REINSTINCT_KV_SUPERQUANT_NAIVE").is_some() {
                self.launch_attn_superquant(self.q_buf.raw_ptr(), sq,
                                            self.attn_concat.raw_ptr(),
                                            n_kv as u32, head_dim as u32)?;
            } else if std::env::var_os("REINSTINCT_KV_SUPERQUANT_RS").is_some() {
                self.launch_attn_superquant_rs(self.q_buf.raw_ptr(), sq,
                                               self.attn_concat.raw_ptr(),
                                               n_kv as u32, head_dim as u32)?;
            } else {
                self.launch_attn_superquant_wp(self.q_buf.raw_ptr(), sq,
                                               self.attn_concat.raw_ptr(),
                                               n_kv as u32, head_dim as u32)?;
            }
        } else {
            self.launch_attn_q8(self.q_buf.raw_ptr(), attn_kv.k.raw_ptr(), attn_kv.ks.raw_ptr(),
                                attn_kv.v.raw_ptr(), attn_kv.vs.raw_ptr(), self.attn_concat.raw_ptr(),
                                n_kv as u32, head_dim as u32, window)?;
        }
        self.prof_lap("a_kernel");
        // Output projection, fused post-norm + residual.
        self.launch_matvec(&b.attn_output, self.attn_concat.raw_ptr(),
                           self.hidden_b.raw_ptr())?;
        self.launch_rmsnorm_add(self.hidden_b.raw_ptr(), b.post_attn_norm.raw_ptr(),
                                self.hidden_a.raw_ptr(), h)?;
        self.prof_lap("a_out_proj");

        // --- FFN --- (dense GeGLU, or the dual shared-MLP + MoE branch)
        match &b.moe {
            None => {
                self.launch_rmsnorm(self.hidden_a.raw_ptr(), b.ffn_norm.raw_ptr(),
                                    self.normed.raw_ptr(), h)?;
                // Quantize the shared post-norm ONCE for ffn_gate + ffn_up.
                if b.ffn_gate.repacked && b.ffn_up.repacked {
                    self.launch_quantize_q8(self.normed.raw_ptr(), self.xq8.raw_ptr(),
                                            b.ffn_gate.in_dim, 1)?;
                    self.launch_matvec_xq8(&b.ffn_gate, self.xq8.raw_ptr(), self.ffn_a.raw_ptr())?;
                    self.launch_matvec_xq8(&b.ffn_up,   self.xq8.raw_ptr(), self.ffn_b.raw_ptr())?;
                } else {
                    self.launch_matvec(&b.ffn_gate, self.normed.raw_ptr(), self.ffn_a.raw_ptr())?;
                    self.launch_matvec(&b.ffn_up,   self.normed.raw_ptr(), self.ffn_b.raw_ptr())?;
                }
                self.launch_geglu(self.ffn_a.raw_ptr(), self.ffn_b.raw_ptr(),
                                  self.ffn_a.raw_ptr(), b.ffn_gate.out_dim as u32)?;
                self.launch_matvec(&b.ffn_down, self.ffn_a.raw_ptr(), self.hidden_b.raw_ptr())?;
                // If there's no PLE residual after this, fold the per-layer
                // output scale into the final rmsnorm_add (saves one launch).
                let fold_scale = self.ple.is_none() || b.ple.is_none();
                if fold_scale {
                    self.launch_rmsnorm_add_scale(self.hidden_b.raw_ptr(),
                        b.post_ffw_norm.raw_ptr(), self.hidden_a.raw_ptr(),
                        h, b.layer_output_scale)?;
                } else {
                    self.launch_rmsnorm_add(self.hidden_b.raw_ptr(), b.post_ffw_norm.raw_ptr(),
                                            self.hidden_a.raw_ptr(), h)?;
                }
            }
            Some(mw) => self.moe_ffn(b, mw)?,
        }

        // --- Per-Layer Embedding (E4B) --- gated residual after the FFN.
        if let (Some(_), Some(pb)) = (&self.ple, &b.ple) {
            let np = self.n_embd_per_layer as u32;
            // gate = inp_gate · hidden_a  (hidden_a kept as the residual)
            self.launch_matvec(&pb.inp_gate, self.hidden_a.raw_ptr(),
                               self.ple_gate.raw_ptr())?;
            // gate = gelu(gate) ⊙ this layer's per-layer embedding slice
            let slice = unsafe {
                (self.ple_raw.raw_ptr() as *mut f32)
                    .add(li * self.n_embd_per_layer) as *mut c_void
            };
            self.launch_geglu(self.ple_gate.raw_ptr(), slice,
                              self.ple_gate.raw_ptr(), np)?;
            // project back up, fused post-norm + residual add + the
            // per-layer output scale (this PLE residual is the final
            // hidden_a writer in this layer).
            self.launch_matvec(&pb.proj, self.ple_gate.raw_ptr(), self.ple_tmp.raw_ptr())?;
            self.launch_rmsnorm_add_scale(self.ple_tmp.raw_ptr(), pb.post_norm.raw_ptr(),
                                          self.hidden_a.raw_ptr(), h, b.layer_output_scale)?;
        }
        // (For paths without PLE, the per-layer output scale was folded
        // into the dense or MoE branch's final rmsnorm_add above.)
        Ok(())
    }

    /// Dual FFN for a MoE layer: a shared dense MLP plus a 128-expert
    /// top-8 routed branch, summed, then the shared post-norm + residual.
    /// `hidden_a` holds attn_out on entry and the post-FFN result on exit.
    fn moe_ffn(&self, b: &GpuGemma4Block, mw: &MoeBlock) -> Result<(), String> {
        let h = self.hidden as u32;
        let ff_exp = self.expert_ff as u32;

        // --- Shared MLP --- → cur_mlp (kept live across the MoE branch).
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), b.ffn_norm.raw_ptr(),
                            self.normed.raw_ptr(), h)?;
        // Quantize the shared post-norm ONCE for ffn_gate + ffn_up.
        if b.ffn_gate.repacked && b.ffn_up.repacked {
            self.launch_quantize_q8(self.normed.raw_ptr(), self.xq8.raw_ptr(),
                                    b.ffn_gate.in_dim, 1)?;
            self.launch_matvec_xq8(&b.ffn_gate, self.xq8.raw_ptr(), self.ffn_a.raw_ptr())?;
            self.launch_matvec_xq8(&b.ffn_up,   self.xq8.raw_ptr(), self.ffn_b.raw_ptr())?;
        } else {
            self.launch_matvec(&b.ffn_gate, self.normed.raw_ptr(), self.ffn_a.raw_ptr())?;
            self.launch_matvec(&b.ffn_up,   self.normed.raw_ptr(), self.ffn_b.raw_ptr())?;
        }
        self.launch_geglu(self.ffn_a.raw_ptr(), self.ffn_b.raw_ptr(),
                          self.ffn_a.raw_ptr(), b.ffn_gate.out_dim as u32)?;
        self.launch_matvec(&b.ffn_down, self.ffn_a.raw_ptr(), self.hidden_b.raw_ptr())?;
        self.launch_rmsnorm(self.hidden_b.raw_ptr(), mw.post_ffw_norm_1.raw_ptr(),
                            self.cur_mlp.raw_ptr(), h)?;
        self.prof_lap("shared_mlp");

        // --- Router --- on attn_out: plain RMSNorm scaled by gate_inp_s,
        // then by 1/√hidden, then the F32 projection to expert logits.
        // gate_inp_s is pre-scaled by 1/sqrt(hidden) at load — the
        // rmsnorm folds the router input scaling, so no separate
        // launch_scale call is needed.
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), mw.gate_inp_s.raw_ptr(),
                            self.normed.raw_ptr(), h)?;
        self.launch_matvec(&mw.gate_inp, self.normed.raw_ptr(), self.moe_logits.raw_ptr())?;
        self.prof_lap("router_matvec");
        self.launch_moe_topk(1)?;
        self.prof_lap("router_topk");

        // --- Routed experts --- fully device-resident: the expert ids
        // from moe_topk stay on device, and one launch per stage covers
        // all n_expert_used experts (grid.y = expert slot). No host
        // round-trip → the whole forward is a pure kernel chain.
        // Fused: xq8 = quantize(rmsnorm(hidden_a) * pre_ffw_norm_2).
        // Saves one launch + the round-trip of normalized values through
        // HBM (was rmsnorm → moe_in → quantize → xq8).
        self.launch_rmsnorm_q8(self.hidden_a.raw_ptr(), mw.pre_ffw_norm_2.raw_ptr(),
                               self.xq8.raw_ptr(), h)?;
        self.launch_moe_matvec(mw.gate_up_exps.dtype, mw.gate_up_exps.repacked,
                               mw.gate_up_exps.data.raw_ptr(),
                               self.xq8.raw_ptr(), self.expert_gu.raw_ptr(), h, 2 * ff_exp,
                               mw.gate_up_exps.bytes_per_expert as u32,
                               /*xq_tok_stride*/ 0,
                               /*xq_slot_stride*/ 0,
                               /*n_tok*/ 1)?;
        // Fused: xq8_experts = quantize(geglu(expert_gu)).
        // Saves one launch + the HBM round-trip of expert_act through fp32.
        self.launch_moe_geglu_q8(self.expert_gu.raw_ptr(), self.xq8_experts.raw_ptr(),
                                 self.n_expert_used)?;
        self.prof_lap("expert_gate_up");
        self.launch_moe_down(mw.down_exps.dtype, mw.down_exps.repacked,
                             mw.down_exps.data.raw_ptr(),
                             self.xq8_experts.raw_ptr(), self.expert_outs.raw_ptr(), ff_exp, h,
                             mw.down_exps.bytes_per_expert as u32,
                             /*xq_tok_stride*/ 0,
                             /*xq_slot_stride*/ ff_exp / 32,
                             /*n_tok*/ 1)?;
        self.prof_lap("expert_down");
        self.launch_moe_combine(self.expert_outs.raw_ptr(), mw.down_exps_s.raw_ptr(),
                                self.moe_acc.raw_ptr(), 1)?;
        // Fused: cur_mlp += rmsnorm(moe_acc) * post_ffw_norm_2
        self.launch_rmsnorm_add(self.moe_acc.raw_ptr(), mw.post_ffw_norm_2.raw_ptr(),
                                self.cur_mlp.raw_ptr(), h)?;
        // Final rmsnorm_add: if no PLE residual follows, fold the
        // per-layer output scale into this kernel (saves one launch).
        let fold_scale = self.ple.is_none() || b.ple.is_none();
        if fold_scale {
            self.launch_rmsnorm_add_scale(self.cur_mlp.raw_ptr(),
                b.post_ffw_norm.raw_ptr(), self.hidden_a.raw_ptr(),
                h, b.layer_output_scale)?;
        } else {
            self.launch_rmsnorm_add(self.cur_mlp.raw_ptr(), b.post_ffw_norm.raw_ptr(),
                                    self.hidden_a.raw_ptr(), h)?;
        }
        self.prof_lap("moe_combine");
        Ok(())
    }
}

