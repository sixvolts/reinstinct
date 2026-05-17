//! Bridge between the CPU oracle (`cpu::qwen3_5::Qwen35F32Model`) and a
//! growing GPU forward path. This first cut owns *just* the weights and
//! kernels needed to validate the chaining/lifetime/integration story:
//!
//!   embed_lookup(token, token_embd)  →  hidden_a
//!   rmsnorm(hidden_a, output_norm)   →  hidden_b
//!   matvec(hidden_b, output_proj)    →  logits
//!
//! The middle step is not physically meaningful on its own (`output_norm`
//! belongs at the *end* of forward, not after the embedding) but the
//! chain still has a well-defined CPU oracle, so it's a tight integration
//! test for everything we need before we add real layers:
//!   - persistent device buffers for resident weights
//!   - persistent Module/Function handles
//!   - device pointers flowing between launches with no host round-trips
//!   - a final D2H copy to compare against CPU
//!
//! As we add kernels (attention, GDN, FFN, RoPE, ...), this struct grows
//! method-by-method and eventually becomes the real GPU forward path.

use std::ffi::c_void;

#[cfg(test)]
use crate::cpu::qwen3_5::Qwen35F32Model;
use crate::model::qwen3_5::Qwen35Model;
use crate::gguf::{GgufFile, GgmlType};
use crate::hip::{self, DeviceBuf, Event, Graph, GraphExec, Module, Stream};
use crate::hip::sys::HipStreamCaptureMode;
use crate::hip::rocblas::{Handle as RocblasHandle, RocblasOp};

/// Per-stage GPU timing breakdown for one `forward_token` call,
/// measured with HIP events (so each `*_ms` is genuine GPU time
/// on `self.stream`, not host wall-clock).
#[derive(Debug, Default, Clone)]
pub struct GpuForwardTrace {
    pub embed_ms:        f32,
    pub block_ms:        Vec<f32>,   // one entry per layer, schedule order
    pub output_norm_ms:  f32,
    pub output_proj_ms:  f32,
    pub total_ms:        f32,        // sum from before embed to after output_proj
}
use super::KernelCache;

const EMBED_LOOKUP_SOURCE:      &str = include_str!("../../kernels/embed_lookup.cpp");
const RMSNORM_SOURCE:           &str = include_str!("../../kernels/rmsnorm.cpp");
const MATVEC_SOURCE:            &str = include_str!("../../kernels/matvec.cpp");
const SWIGLU_SOURCE:            &str = include_str!("../../kernels/swiglu.cpp");
const RMSNORM_MULTIHEAD_SOURCE: &str = include_str!("../../kernels/rmsnorm_multihead.cpp");
const SPLIT_Q_GATE_SOURCE:      &str = include_str!("../../kernels/split_q_gate.cpp");
const SIGMOID_MUL_SOURCE:       &str = include_str!("../../kernels/sigmoid_mul.cpp");
const ROPE_SOURCE:              &str = include_str!("../../kernels/rope.cpp");
const ATTN_STEP_SOURCE:         &str = include_str!("../../kernels/attn_step.cpp");
const ADD_INPLACE_SOURCE:       &str = include_str!("../../kernels/add_inplace.cpp");
const CONV1D_STEP_SOURCE:           &str = include_str!("../../kernels/conv1d_step.cpp");
const SILU_INPLACE_SOURCE:          &str = include_str!("../../kernels/silu_inplace.cpp");
const L2NORM_MULTIHEAD_SOURCE:      &str = include_str!("../../kernels/l2norm_multihead.cpp");
const GDN_DECAY_BETA_SOURCE:        &str = include_str!("../../kernels/gdn_decay_beta.cpp");
const GDN_RECURRENT_STEP_SOURCE:    &str = include_str!("../../kernels/gdn_recurrent_step.cpp");
const GDN_RECURRENT_STEP_LDS_SOURCE:&str = include_str!("../../kernels/gdn_recurrent_step_lds.cpp");
const GDN_RECURRENT_STEP_FUSED_SOURCE: &str = include_str!("../../kernels/gdn_recurrent_step_fused.cpp");
const CONV1D_STEP_SILU_SOURCE:      &str = include_str!("../../kernels/conv1d_step_silu.cpp");
const L2NORM_QK_SOURCE:             &str = include_str!("../../kernels/l2norm_qk.cpp");
const RMSNORM_GATED_MULTIHEAD_SOURCE: &str = include_str!("../../kernels/rmsnorm_gated_multihead.cpp");

const MATVEC_Q8_0_SOURCE:   &str = include_str!("../../kernels/matvec_q8_0.cpp");
const MATVEC_Q4_K_SOURCE:   &str = include_str!("../../kernels/matvec_q4_k.cpp");
const MATVEC_Q5_K_SOURCE:   &str = include_str!("../../kernels/matvec_q5_k.cpp");
const MATVEC_Q6_K_SOURCE:   &str = include_str!("../../kernels/matvec_q6_k.cpp");
const MATVEC_IQ4_XS_SOURCE: &str = include_str!("../../kernels/matvec_iq4_xs.cpp");
const MATVEC_F16_SOURCE:    &str = include_str!("../../kernels/matvec_f16.cpp");
const EMBED_LOOKUP_Q6_K_SOURCE: &str = include_str!("../../kernels/embed_lookup_q6_k.cpp");
const EMBED_LOOKUP_Q4_K_SOURCE: &str = include_str!("../../kernels/embed_lookup_q4_k.cpp");

const CVT_F32_F16_SOURCE:       &str = include_str!("../../kernels/cvt_f32_f16.cpp");
const DEQUANT_Q4_K_F16_SOURCE:  &str = include_str!("../../kernels/dequant_q4_k_f16.cpp");
const DEQUANT_Q5_K_F16_SOURCE:  &str = include_str!("../../kernels/dequant_q5_k_f16.cpp");
const DEQUANT_Q6_K_F16_SOURCE:  &str = include_str!("../../kernels/dequant_q6_k_f16.cpp");
const DEQUANT_Q8_0_F16_SOURCE:  &str = include_str!("../../kernels/dequant_q8_0_f16.cpp");
const DEQUANT_IQ4_XS_F16_SOURCE:&str = include_str!("../../kernels/dequant_iq4_xs_f16.cpp");
const ROPE_BATCHED_SOURCE:      &str = include_str!("../../kernels/rope_batched.cpp");
const ATTN_STEP_BATCHED_SOURCE: &str = include_str!("../../kernels/attn_step_batched.cpp");

const MATVEC_F32_WAVE64_SOURCE:    &str = include_str!("../../kernels/matvec_f32_wave64.cpp");
const MATVEC_Q4_K_WAVE64_SOURCE:   &str = include_str!("../../kernels/matvec_q4_k_wave64.cpp");
const MATVEC_Q5_K_WAVE64_SOURCE:   &str = include_str!("../../kernels/matvec_q5_k_wave64.cpp");
const MATVEC_Q6_K_WAVE64_SOURCE:   &str = include_str!("../../kernels/matvec_q6_k_wave64.cpp");
const MATVEC_Q8_0_WAVE64_SOURCE:   &str = include_str!("../../kernels/matvec_q8_0_wave64.cpp");
const MATVEC_IQ4_XS_WAVE64_SOURCE: &str = include_str!("../../kernels/matvec_iq4_xs_wave64.cpp");
const MATVEC_F16_WAVE64_SOURCE:    &str = include_str!("../../kernels/matvec_f16_wave64.cpp");

/// A weight tensor used as the W matrix in a `y = W·x` matvec, resident on
/// device. Holds the raw on-disk byte stream + on-disk dtype, so the
/// dispatcher can pick the right fused dequant+GEMV kernel per type.
///
/// Shape convention follows GGUF: `shape = [in_dim, out_dim]`, flat layout
/// `w[j * in_dim + i]` (row j is one output row of length in_dim).
pub struct GpuMatvecTensor {
    pub data:    DeviceBuf<u8>,
    pub dtype:   GgmlType,
    pub in_dim:  u32,
    pub out_dim: u32,
}

impl GpuMatvecTensor {
    /// Load the named tensor from `gguf` straight to device memory in its
    /// on-disk form. Verifies the tensor is 2D and computes (in_dim, out_dim).
    pub fn from_gguf(gguf: &GgufFile, name: &str) -> Result<Self, String> {
        let info = gguf.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
        let bytes = gguf.tensor_data(name)
            .map_err(|e| format!("read {name}: {e}"))?
            .ok_or_else(|| format!("tensor {name} has no data"))?;
        let shape = info.shape();
        if shape.len() != 2 {
            return Err(format!("tensor {name}: expected 2D, got {shape:?}"));
        }
        let in_dim  = shape[0] as u32;
        let out_dim = shape[1] as u32;
        Ok(Self {
            data: DeviceBuf::from_slice(bytes)?,
            dtype: info.ggml_type,
            in_dim, out_dim,
        })
    }
}

/// Load an fp32 tensor straight from GGUF to device.
fn load_fp32_tensor(gguf: &GgufFile, name: &str) -> Result<DeviceBuf<f32>, String> {
    let info = gguf.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
    if info.ggml_type != GgmlType::F32 {
        return Err(format!("tensor {name}: expected F32, got {:?}", info.ggml_type));
    }
    let bytes = gguf.tensor_data(name)
        .map_err(|e| format!("read {name}: {e}"))?
        .ok_or_else(|| format!("tensor {name} has no data"))?;
    let floats: &[f32] = bytemuck::cast_slice(bytes);
    DeviceBuf::from_slice(floats)
}

/// FFN weights for a single transformer block, resident on device. Matvec
/// weights are kept in their on-disk quantized form; the matvec dispatcher
/// picks the right kernel per dtype.
pub struct GpuFfnWeights {
    pub gate: GpuMatvecTensor,   // [hidden, ffn]
    pub up:   GpuMatvecTensor,   // [hidden, ffn]
    pub down: GpuMatvecTensor,   // [ffn,    hidden]
}

impl GpuFfnWeights {
    pub fn from_gguf(gguf: &GgufFile, layer: u32) -> Result<Self, String> {
        Ok(Self {
            gate: GpuMatvecTensor::from_gguf(gguf, &format!("blk.{layer}.ffn_gate.weight"))?,
            up:   GpuMatvecTensor::from_gguf(gguf, &format!("blk.{layer}.ffn_up.weight"))?,
            down: GpuMatvecTensor::from_gguf(gguf, &format!("blk.{layer}.ffn_down.weight"))?,
        })
    }
}

/// All weights for one full-attention transformer block on the GPU.
/// Bundles the attention sub-layer, the post-attention norm, and the
/// FFN sub-layer in the same lifetime.
pub struct GpuFullAttnBlock {
    pub attn:       GpuFullAttnWeights,
    pub post_norm:  DeviceBuf<f32>,    // [hidden] — pre-FFN RMSNorm weight
    pub ffn:        GpuFfnWeights,
}

impl GpuFullAttnBlock {
    pub fn from_gguf(gguf: &GgufFile, layer: u32) -> Result<Self, String> {
        Ok(Self {
            attn:      GpuFullAttnWeights::from_gguf(gguf, layer)?,
            post_norm: load_fp32_tensor(gguf, &format!("blk.{layer}.post_attention_norm.weight"))?,
            ffn:       GpuFfnWeights::from_gguf(gguf, layer)?,
        })
    }
}

/// Full-attention block weights for a single transformer block.
pub struct GpuFullAttnWeights {
    pub attn_norm:   DeviceBuf<f32>,    // [hidden]
    pub attn_q:      GpuMatvecTensor,   // [hidden, 2 * q_dim]   (Q | gate concat)
    pub attn_k:      GpuMatvecTensor,   // [hidden, kv_dim]
    pub attn_v:      GpuMatvecTensor,   // [hidden, kv_dim]
    pub attn_q_norm: DeviceBuf<f32>,    // [head_dim]            (per-head)
    pub attn_k_norm: DeviceBuf<f32>,    // [head_dim]
    pub attn_output: GpuMatvecTensor,   // [q_dim, hidden]
}

impl GpuFullAttnWeights {
    pub fn from_gguf(gguf: &GgufFile, layer: u32) -> Result<Self, String> {
        let pre = format!("blk.{layer}.");
        Ok(Self {
            attn_norm:   load_fp32_tensor(gguf, &format!("{pre}attn_norm.weight"))?,
            attn_q:      GpuMatvecTensor::from_gguf(gguf, &format!("{pre}attn_q.weight"))?,
            attn_k:      GpuMatvecTensor::from_gguf(gguf, &format!("{pre}attn_k.weight"))?,
            attn_v:      GpuMatvecTensor::from_gguf(gguf, &format!("{pre}attn_v.weight"))?,
            attn_q_norm: load_fp32_tensor(gguf, &format!("{pre}attn_q_norm.weight"))?,
            attn_k_norm: load_fp32_tensor(gguf, &format!("{pre}attn_k_norm.weight"))?,
            attn_output: GpuMatvecTensor::from_gguf(gguf, &format!("{pre}attn_output.weight"))?,
        })
    }
}

/// Linear-attention (GDN) block weights, resident on device.
pub struct GpuLinAttnWeights {
    pub attn_norm:   DeviceBuf<f32>,    // [hidden]
    pub attn_qkv:    GpuMatvecTensor,   // [hidden, conv_dim]
    pub attn_gate:   GpuMatvecTensor,   // [hidden, value_dim]
    pub ssm_alpha:   GpuMatvecTensor,   // [hidden, n_heads]
    pub ssm_beta:    GpuMatvecTensor,   // [hidden, n_heads]
    pub ssm_a:       DeviceBuf<f32>,    // [n_heads]   (already -exp(A_log))
    pub ssm_dt_bias: DeviceBuf<f32>,    // [n_heads]
    pub ssm_conv1d:  DeviceBuf<f32>,    // [conv_dim, kernel]
    pub ssm_norm:    DeviceBuf<f32>,    // [head_dim]
    pub ssm_out:     GpuMatvecTensor,   // [value_dim, hidden]
}

impl GpuLinAttnWeights {
    pub fn from_gguf(gguf: &GgufFile, layer: u32) -> Result<Self, String> {
        let pre = format!("blk.{layer}.");
        Ok(Self {
            attn_norm:   load_fp32_tensor(gguf, &format!("{pre}attn_norm.weight"))?,
            attn_qkv:    GpuMatvecTensor::from_gguf(gguf, &format!("{pre}attn_qkv.weight"))?,
            attn_gate:   GpuMatvecTensor::from_gguf(gguf, &format!("{pre}attn_gate.weight"))?,
            ssm_alpha:   GpuMatvecTensor::from_gguf(gguf, &format!("{pre}ssm_alpha.weight"))?,
            ssm_beta:    GpuMatvecTensor::from_gguf(gguf, &format!("{pre}ssm_beta.weight"))?,
            ssm_a:       load_fp32_tensor(gguf, &format!("{pre}ssm_a"))?,
            ssm_dt_bias: load_fp32_tensor(gguf, &format!("{pre}ssm_dt.bias"))?,
            ssm_conv1d:  load_fp32_tensor(gguf, &format!("{pre}ssm_conv1d.weight"))?,
            ssm_norm:    load_fp32_tensor(gguf, &format!("{pre}ssm_norm.weight"))?,
            ssm_out:     GpuMatvecTensor::from_gguf(gguf, &format!("{pre}ssm_out.weight"))?,
        })
    }
}

/// All weights for one linear-attention transformer block on the GPU
/// (GDN attention + post-norm + FFN).
pub struct GpuLinAttnBlock {
    pub attn:      GpuLinAttnWeights,
    pub post_norm: DeviceBuf<f32>,
    pub ffn:       GpuFfnWeights,
}

impl GpuLinAttnBlock {
    pub fn from_gguf(gguf: &GgufFile, layer: u32) -> Result<Self, String> {
        Ok(Self {
            attn:      GpuLinAttnWeights::from_gguf(gguf, layer)?,
            post_norm: load_fp32_tensor(gguf, &format!("blk.{layer}.post_attention_norm.weight"))?,
            ffn:       GpuFfnWeights::from_gguf(gguf, layer)?,
        })
    }
}

/// Per-GDN-block recurrent + Conv1D state, resident on device.
pub struct GpuLinAttnState {
    pub recurrent: DeviceBuf<f32>,    // [n_heads, head_dim, head_dim]
    pub conv_hist: DeviceBuf<f32>,    // [conv_dim, kernel-1]
    pub n_heads:     usize,
    pub head_dim:    usize,
    pub conv_dim:    usize,
    pub conv_kernel: usize,
}

impl GpuLinAttnState {
    pub fn new(n_heads: usize, head_dim: usize, conv_dim: usize, conv_kernel: usize)
        -> Result<Self, String>
    {
        let recurrent = DeviceBuf::new(n_heads * head_dim * head_dim)?;
        let conv_hist = DeviceBuf::new(conv_dim * (conv_kernel - 1))?;
        // Zero-initialise: hipMalloc returns uninitialised; populate from
        // host zeros so the recurrent matrix and conv history start clean.
        let zeros_r = vec![0.0f32; recurrent.len()];
        recurrent.copy_from_host(&zeros_r)?;
        let zeros_c = vec![0.0f32; conv_hist.len()];
        conv_hist.copy_from_host(&zeros_c)?;
        Ok(Self { recurrent, conv_hist, n_heads, head_dim, conv_dim, conv_kernel })
    }

    pub fn reset(&mut self) -> Result<(), String> {
        let zeros_r = vec![0.0f32; self.recurrent.len()];
        self.recurrent.copy_from_host(&zeros_r)?;
        let zeros_c = vec![0.0f32; self.conv_hist.len()];
        self.conv_hist.copy_from_host(&zeros_c)?;
        Ok(())
    }
}

/// One transformer block's weights, dispatched on block kind. Owned by
/// `GpuQwen35` (one per layer); the inner bundle holds all weights for
/// that block's attention sub-layer + post-norm + FFN.
pub enum GpuBlock {
    Full(GpuFullAttnBlock),
    Linear(GpuLinAttnBlock),
}

impl GpuBlock {
    pub fn from_gguf(gguf: &GgufFile, layer: u32, kind: crate::model::qwen3_5::BlockKind)
        -> Result<Self, String>
    {
        use crate::model::qwen3_5::BlockKind;
        Ok(match kind {
            BlockKind::FullAttention   => GpuBlock::Full(GpuFullAttnBlock::from_gguf(gguf, layer)?),
            BlockKind::LinearAttention => GpuBlock::Linear(GpuLinAttnBlock::from_gguf(gguf, layer)?),
        })
    }
}

/// One transformer block's mutable state. KV cache for full attention,
/// recurrent + conv state for GDN.
pub enum GpuBlockState {
    Full(GpuKvCache),
    Linear(GpuLinAttnState),
}

impl GpuBlockState {
    pub fn reset(&mut self) -> Result<(), String> {
        match self {
            GpuBlockState::Full(kv) => { kv.reset(); Ok(()) }
            GpuBlockState::Linear(s) => s.reset(),
        }
    }
}

/// Per-token mutable state for a Qwen 3.5 forward pass: one block-state
/// per layer plus a position counter (mostly diagnostic — each block-state
/// keeps its own position).
pub struct Qwen35GpuState {
    pub block_states: Vec<GpuBlockState>,
    pub pos: usize,
}

impl Qwen35GpuState {
    pub fn new(model: &Qwen35Model, max_seq: usize) -> Result<Self, String> {
        use crate::model::qwen3_5::BlockKind;
        let cfg = &model.config;
        let conv_dim = 3 * cfg.gdn_value_dim as usize;
        let mut block_states = Vec::with_capacity(model.block_kinds.len());
        for &kind in &model.block_kinds {
            block_states.push(match kind {
                BlockKind::FullAttention => GpuBlockState::Full(GpuKvCache::new(
                    max_seq,
                    cfg.attn_n_kv_heads as usize,
                    cfg.attn_head_dim as usize,
                )?),
                BlockKind::LinearAttention => GpuBlockState::Linear(GpuLinAttnState::new(
                    cfg.gdn_n_heads     as usize,
                    cfg.gdn_head_dim    as usize,
                    conv_dim,
                    cfg.gdn_conv_kernel as usize,
                )?),
            });
        }
        Ok(Self { block_states, pos: 0 })
    }

    pub fn reset(&mut self) -> Result<(), String> {
        for s in &mut self.block_states { s.reset()?; }
        self.pos = 0;
        Ok(())
    }
}

/// Per-block KV cache resident on device.
pub struct GpuKvCache {
    pub k: DeviceBuf<f32>,     // [max_seq, n_kv_heads, head_dim]
    pub v: DeviceBuf<f32>,     // [max_seq, n_kv_heads, head_dim]
    pub max_seq: usize,
    pub kv_dim: usize,         // n_kv_heads * head_dim — bytes per slot
    pub len:    usize,         // populated positions [0, len)
}

impl GpuKvCache {
    pub fn new(max_seq: usize, n_kv_heads: usize, head_dim: usize) -> Result<Self, String> {
        let kv_dim = n_kv_heads * head_dim;
        Ok(Self {
            k: DeviceBuf::new(max_seq * kv_dim)?,
            v: DeviceBuf::new(max_seq * kv_dim)?,
            max_seq, kv_dim, len: 0,
        })
    }
    pub fn reset(&mut self) { self.len = 0; }
}

pub struct GpuQwen35 {
    // Resident weights.
    token_embd: GpuMatvecTensor,           // [hidden, vocab] (GGUF shape order)
    output_norm: DeviceBuf<f32>,           // [hidden]
    /// `None` when `tied_embeddings` — `output_proj` reuses `token_embd`.
    output_proj: Option<GpuMatvecTensor>,  // [hidden, vocab]

    // Per-call activation scratch (persistent across calls; overwritten each call).
    hidden_a:    DeviceBuf<f32>,   // [hidden]
    hidden_b:    DeviceBuf<f32>,   // [hidden]
    ffn_a:       DeviceBuf<f32>,   // [ffn]
    ffn_b:       DeviceBuf<f32>,   // [ffn]
    q_raw:       DeviceBuf<f32>,   // [2 * q_dim]
    q_buf:       DeviceBuf<f32>,   // [q_dim]
    gate_buf:    DeviceBuf<f32>,   // [q_dim]
    k_raw:       DeviceBuf<f32>,   // [kv_dim]
    v_raw:       DeviceBuf<f32>,   // [kv_dim]
    k_norm:      DeviceBuf<f32>,   // [kv_dim]
    attn_concat: DeviceBuf<f32>,   // [q_dim]
    logits:      DeviceBuf<f32>,   // [vocab]

    // GDN scratch buffers.
    gdn_qkv:      DeviceBuf<f32>,  // [conv_dim]          mixed_qkv projection
    gdn_conv_out: DeviceBuf<f32>,  // [conv_dim]          conv1d output (post-silu)
    gdn_z:        DeviceBuf<f32>,  // [value_dim]         attn_gate projection
    gdn_a:        DeviceBuf<f32>,  // [n_heads]           ssm_alpha projection
    gdn_b:        DeviceBuf<f32>,  // [n_heads]           ssm_beta projection
    gdn_q:        DeviceBuf<f32>,  // [value_dim]         L2-normed Q (scaled)
    gdn_k:        DeviceBuf<f32>,  // [value_dim]         L2-normed K
    gdn_decay:    DeviceBuf<f32>,  // [n_heads]
    gdn_beta:     DeviceBuf<f32>,  // [n_heads]
    gdn_core_out: DeviceBuf<f32>,  // [value_dim]         core attn out / normed_out
    gdn_delta:    DeviceBuf<f32>,  // [n_heads, head_dim] cross-thread delta for LDS recurrent kernel

    // RoPE tables resident on device.
    rope_cos: DeviceBuf<f32>,      // [max_seq, rotary_dim]
    rope_sin: DeviceBuf<f32>,      // [max_seq, rotary_dim]

    // Compiled kernel modules — keep alive for the lifetime of self.
    embed_module:            Module,
    rmsnorm_module:          Module,
    matvec_module:           Module,
    swiglu_module:           Module,
    rmsnorm_multihead_module: Module,
    split_q_gate_module:     Module,
    sigmoid_mul_module:      Module,
    rope_module:             Module,
    attn_step_module:        Module,
    add_inplace_module:      Module,
    conv1d_step_module:           Module,
    silu_inplace_module:          Module,
    l2norm_multihead_module:      Module,
    gdn_decay_beta_module:        Module,
    gdn_recurrent_step_module:    Module,
    gdn_recurrent_step_lds_module: Module,
    gdn_recurrent_step_fused_module: Module,
    conv1d_step_silu_module:      Module,
    l2norm_qk_module:             Module,
    rmsnorm_gated_multihead_module: Module,

    matvec_q8_0_module:    Module,
    matvec_q4_k_module:    Module,
    matvec_q5_k_module:    Module,
    matvec_q6_k_module:    Module,
    matvec_iq4_xs_module:  Module,
    matvec_f16_module:     Module,
    embed_lookup_q6_k_module: Module,
    embed_lookup_q4_k_module: Module,
    matvec_f32_wave64_module:    Module,
    matvec_q4_k_wave64_module:   Module,
    matvec_q5_k_wave64_module:   Module,
    matvec_q6_k_wave64_module:   Module,
    matvec_q8_0_wave64_module:   Module,
    matvec_iq4_xs_wave64_module: Module,
    matvec_f16_wave64_module:    Module,

    /// Per-layer transformer block weights, in schedule order.
    blocks: Vec<GpuBlock>,

    /// Stream all kernel launches and async memcpys flow through. Owning
    /// one stream lets us capture the whole forward chain into a HIP graph.
    stream: Stream,

    // --- Batched prefill machinery ---
    rocblas:           RocblasHandle,
    cvt_module:        Module,
    dequant_q4_k_module:   Module,
    dequant_q5_k_module:   Module,
    dequant_q6_k_module:   Module,
    dequant_q8_0_module:   Module,
    dequant_iq4_xs_module: Module,
    rope_batched_module:   Module,
    attn_step_batched_module: Module,

    // Dimensions.
    hidden:     usize,
    ffn:        usize,
    vocab:      usize,
    n_heads:    usize,
    n_kv_heads: usize,
    head_dim:   usize,
    rotary_dim: usize,
    // GDN dims.
    gdn_value_dim:   usize,
    gdn_conv_dim:    usize,
    gdn_n_heads:     usize,
    gdn_head_dim:    usize,
    gdn_conv_kernel: usize,
    rms_eps:    f32,
    #[allow(dead_code)]
    max_seq:    usize,
}

impl GpuQwen35 {
    pub fn new(model: &Qwen35Model, gguf: &GgufFile, cache: &KernelCache, max_seq: usize)
        -> Result<Self, String>
    {
        let cfg = &model.config;
        let hidden     = cfg.hidden_size      as usize;
        let ffn        = cfg.ffn_size         as usize;
        let vocab      = cfg.vocab_size       as usize;
        let n_heads    = cfg.attn_n_heads     as usize;
        let n_kv_heads = cfg.attn_n_kv_heads  as usize;
        let head_dim   = cfg.attn_head_dim    as usize;
        let rotary_dim = cfg.rope_dim_count   as usize;
        let q_dim  = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        // GDN dims.
        let gdn_value_dim   = cfg.gdn_value_dim   as usize;
        let gdn_n_heads     = cfg.gdn_n_heads     as usize;
        let gdn_head_dim    = cfg.gdn_head_dim    as usize;
        let gdn_conv_kernel = cfg.gdn_conv_kernel as usize;
        // conv_dim = 2 * key_dim + value_dim; in Qwen 3.5 0.8B key=value, so 3*value_dim.
        let gdn_conv_dim    = 3 * gdn_value_dim;

        let token_embd  = GpuMatvecTensor::from_gguf(gguf, "token_embd.weight")?;
        let output_norm = load_fp32_tensor(gguf, "output_norm.weight")?;
        let output_proj = if cfg.tied_embeddings {
            None
        } else {
            Some(GpuMatvecTensor::from_gguf(gguf, "output.weight")?)
        };

        let hidden_a    = DeviceBuf::new(hidden)?;
        let hidden_b    = DeviceBuf::new(hidden)?;
        let ffn_a       = DeviceBuf::new(ffn)?;
        let ffn_b       = DeviceBuf::new(ffn)?;
        let q_raw       = DeviceBuf::new(2 * q_dim)?;
        let q_buf       = DeviceBuf::new(q_dim)?;
        let gate_buf    = DeviceBuf::new(q_dim)?;
        let k_raw       = DeviceBuf::new(kv_dim)?;
        let v_raw       = DeviceBuf::new(kv_dim)?;
        let k_norm      = DeviceBuf::new(kv_dim)?;
        let attn_concat = DeviceBuf::new(q_dim)?;
        let logits      = DeviceBuf::new(vocab)?;

        let gdn_qkv      = DeviceBuf::new(gdn_conv_dim)?;
        let gdn_conv_out = DeviceBuf::new(gdn_conv_dim)?;
        let gdn_z        = DeviceBuf::new(gdn_value_dim)?;
        let gdn_a        = DeviceBuf::new(gdn_n_heads)?;
        let gdn_b        = DeviceBuf::new(gdn_n_heads)?;
        let gdn_q        = DeviceBuf::new(gdn_value_dim)?;
        let gdn_k        = DeviceBuf::new(gdn_value_dim)?;
        let gdn_decay    = DeviceBuf::new(gdn_n_heads)?;
        let gdn_beta     = DeviceBuf::new(gdn_n_heads)?;
        let gdn_core_out = DeviceBuf::new(gdn_value_dim)?;
        let gdn_delta    = DeviceBuf::new(gdn_n_heads * gdn_head_dim)?;

        // Build RoPE tables host-side once and upload.
        let rope = crate::cpu::rope::RopeCache::new(rotary_dim, max_seq, cfg.rope_freq_base);
        let mut cos = vec![0.0f32; max_seq * rotary_dim];
        let mut sin = vec![0.0f32; max_seq * rotary_dim];
        for pos in 0..max_seq {
            let (c, s) = rope.get(pos);
            cos[pos * rotary_dim..(pos + 1) * rotary_dim].copy_from_slice(c);
            sin[pos * rotary_dim..(pos + 1) * rotary_dim].copy_from_slice(s);
        }
        let rope_cos = DeviceBuf::from_slice(&cos)?;
        let rope_sin = DeviceBuf::from_slice(&sin)?;

        let embed_hsaco             = cache.compile("embed_lookup",      EMBED_LOOKUP_SOURCE)?;
        let rmsnorm_hsaco           = cache.compile("rmsnorm",           RMSNORM_SOURCE)?;
        let matvec_hsaco            = cache.compile("matvec",            MATVEC_SOURCE)?;
        let swiglu_hsaco            = cache.compile("swiglu",            SWIGLU_SOURCE)?;
        let rmsnorm_multihead_hsaco = cache.compile("rmsnorm_multihead", RMSNORM_MULTIHEAD_SOURCE)?;
        let split_q_gate_hsaco      = cache.compile("split_q_gate",      SPLIT_Q_GATE_SOURCE)?;
        let sigmoid_mul_hsaco       = cache.compile("sigmoid_mul",       SIGMOID_MUL_SOURCE)?;
        let rope_hsaco              = cache.compile("rope",              ROPE_SOURCE)?;
        let attn_step_hsaco         = cache.compile("attn_step",         ATTN_STEP_SOURCE)?;
        let add_inplace_hsaco       = cache.compile("add_inplace",       ADD_INPLACE_SOURCE)?;
        let conv1d_step_hsaco            = cache.compile("conv1d_step",       CONV1D_STEP_SOURCE)?;
        let silu_inplace_hsaco           = cache.compile("silu_inplace",      SILU_INPLACE_SOURCE)?;
        let l2norm_multihead_hsaco       = cache.compile("l2norm_multihead",  L2NORM_MULTIHEAD_SOURCE)?;
        let gdn_decay_beta_hsaco         = cache.compile("gdn_decay_beta",    GDN_DECAY_BETA_SOURCE)?;
        let gdn_recurrent_step_hsaco     = cache.compile("gdn_recurrent_step", GDN_RECURRENT_STEP_SOURCE)?;
        let gdn_recurrent_step_lds_hsaco = cache.compile("gdn_recurrent_step_lds", GDN_RECURRENT_STEP_LDS_SOURCE)?;
        let gdn_recurrent_step_fused_hsaco = cache.compile("gdn_recurrent_step_fused", GDN_RECURRENT_STEP_FUSED_SOURCE)?;
        let conv1d_step_silu_hsaco       = cache.compile("conv1d_step_silu", CONV1D_STEP_SILU_SOURCE)?;
        let l2norm_qk_hsaco              = cache.compile("l2norm_qk",        L2NORM_QK_SOURCE)?;
        let rmsnorm_gated_multihead_hsaco = cache.compile("rmsnorm_gated_multihead", RMSNORM_GATED_MULTIHEAD_SOURCE)?;
        let matvec_q8_0_hsaco   = cache.compile("matvec_q8_0",   MATVEC_Q8_0_SOURCE)?;
        let matvec_q4_k_hsaco   = cache.compile("matvec_q4_k",   MATVEC_Q4_K_SOURCE)?;
        let matvec_q5_k_hsaco   = cache.compile("matvec_q5_k",   MATVEC_Q5_K_SOURCE)?;
        let matvec_q6_k_hsaco   = cache.compile("matvec_q6_k",   MATVEC_Q6_K_SOURCE)?;
        let matvec_iq4_xs_hsaco = cache.compile("matvec_iq4_xs", MATVEC_IQ4_XS_SOURCE)?;
        let matvec_f16_hsaco    = cache.compile("matvec_f16",    MATVEC_F16_SOURCE)?;
        let embed_lookup_q6_k_hsaco = cache.compile("embed_lookup_q6_k", EMBED_LOOKUP_Q6_K_SOURCE)?;
        let embed_lookup_q4_k_hsaco = cache.compile("embed_lookup_q4_k", EMBED_LOOKUP_Q4_K_SOURCE)?;
        let matvec_f32_wave64_hsaco    = cache.compile("matvec_f32_wave64",    MATVEC_F32_WAVE64_SOURCE)?;
        let matvec_q4_k_wave64_hsaco   = cache.compile("matvec_q4_k_wave64",   MATVEC_Q4_K_WAVE64_SOURCE)?;
        let matvec_q5_k_wave64_hsaco   = cache.compile("matvec_q5_k_wave64",   MATVEC_Q5_K_WAVE64_SOURCE)?;
        let matvec_q6_k_wave64_hsaco   = cache.compile("matvec_q6_k_wave64",   MATVEC_Q6_K_WAVE64_SOURCE)?;
        let matvec_q8_0_wave64_hsaco   = cache.compile("matvec_q8_0_wave64",   MATVEC_Q8_0_WAVE64_SOURCE)?;
        let matvec_iq4_xs_wave64_hsaco = cache.compile("matvec_iq4_xs_wave64", MATVEC_IQ4_XS_WAVE64_SOURCE)?;
        let matvec_f16_wave64_hsaco    = cache.compile("matvec_f16_wave64",    MATVEC_F16_WAVE64_SOURCE)?;

        // Load every per-layer block's weights from GGUF.
        let mut blocks = Vec::with_capacity(model.block_kinds.len());
        for (i, &kind) in model.block_kinds.iter().enumerate() {
            blocks.push(GpuBlock::from_gguf(gguf, i as u32, kind)?);
        }

        // The single stream all launches flow through.
        let stream = Stream::new()?;
        // rocBLAS handle for batched-prefill GEMMs, bound to our stream.
        let rocblas_handle = RocblasHandle::new()?;
        rocblas_handle.set_stream(&stream)?;

        Ok(Self {
            token_embd, output_norm, output_proj,
            hidden_a, hidden_b, ffn_a, ffn_b,
            q_raw, q_buf, gate_buf, k_raw, v_raw, k_norm, attn_concat, logits,
            rope_cos, rope_sin,
            embed_module:             Module::load(&embed_hsaco)?,
            rmsnorm_module:           Module::load(&rmsnorm_hsaco)?,
            matvec_module:            Module::load(&matvec_hsaco)?,
            swiglu_module:            Module::load(&swiglu_hsaco)?,
            rmsnorm_multihead_module: Module::load(&rmsnorm_multihead_hsaco)?,
            split_q_gate_module:      Module::load(&split_q_gate_hsaco)?,
            sigmoid_mul_module:       Module::load(&sigmoid_mul_hsaco)?,
            rope_module:              Module::load(&rope_hsaco)?,
            attn_step_module:         Module::load(&attn_step_hsaco)?,
            add_inplace_module:       Module::load(&add_inplace_hsaco)?,
            conv1d_step_module:           Module::load(&conv1d_step_hsaco)?,
            silu_inplace_module:          Module::load(&silu_inplace_hsaco)?,
            l2norm_multihead_module:      Module::load(&l2norm_multihead_hsaco)?,
            gdn_decay_beta_module:        Module::load(&gdn_decay_beta_hsaco)?,
            gdn_recurrent_step_module:    Module::load(&gdn_recurrent_step_hsaco)?,
            gdn_recurrent_step_lds_module: Module::load(&gdn_recurrent_step_lds_hsaco)?,
            gdn_recurrent_step_fused_module: Module::load(&gdn_recurrent_step_fused_hsaco)?,
            conv1d_step_silu_module:      Module::load(&conv1d_step_silu_hsaco)?,
            l2norm_qk_module:             Module::load(&l2norm_qk_hsaco)?,
            rmsnorm_gated_multihead_module: Module::load(&rmsnorm_gated_multihead_hsaco)?,
            matvec_q8_0_module:   Module::load(&matvec_q8_0_hsaco)?,
            matvec_q4_k_module:   Module::load(&matvec_q4_k_hsaco)?,
            matvec_q5_k_module:   Module::load(&matvec_q5_k_hsaco)?,
            matvec_q6_k_module:   Module::load(&matvec_q6_k_hsaco)?,
            matvec_iq4_xs_module: Module::load(&matvec_iq4_xs_hsaco)?,
            matvec_f16_module:    Module::load(&matvec_f16_hsaco)?,
            embed_lookup_q6_k_module: Module::load(&embed_lookup_q6_k_hsaco)?,
            embed_lookup_q4_k_module: Module::load(&embed_lookup_q4_k_hsaco)?,
            rocblas:                  rocblas_handle,
            cvt_module:               Module::load(&cache.compile("cvt_f32_f16", CVT_F32_F16_SOURCE)?)?,
            dequant_q4_k_module:      Module::load(&cache.compile("dequant_q4_k_f16", DEQUANT_Q4_K_F16_SOURCE)?)?,
            dequant_q5_k_module:      Module::load(&cache.compile("dequant_q5_k_f16", DEQUANT_Q5_K_F16_SOURCE)?)?,
            dequant_q6_k_module:      Module::load(&cache.compile("dequant_q6_k_f16", DEQUANT_Q6_K_F16_SOURCE)?)?,
            dequant_q8_0_module:      Module::load(&cache.compile("dequant_q8_0_f16", DEQUANT_Q8_0_F16_SOURCE)?)?,
            dequant_iq4_xs_module:    Module::load(&cache.compile("dequant_iq4_xs_f16", DEQUANT_IQ4_XS_F16_SOURCE)?)?,
            rope_batched_module:      Module::load(&cache.compile("rope_batched", ROPE_BATCHED_SOURCE)?)?,
            attn_step_batched_module: Module::load(&cache.compile("attn_step_batched", ATTN_STEP_BATCHED_SOURCE)?)?,
            matvec_f32_wave64_module:    Module::load(&matvec_f32_wave64_hsaco)?,
            matvec_q4_k_wave64_module:   Module::load(&matvec_q4_k_wave64_hsaco)?,
            matvec_q5_k_wave64_module:   Module::load(&matvec_q5_k_wave64_hsaco)?,
            matvec_q6_k_wave64_module:   Module::load(&matvec_q6_k_wave64_hsaco)?,
            matvec_q8_0_wave64_module:   Module::load(&matvec_q8_0_wave64_hsaco)?,
            matvec_iq4_xs_wave64_module: Module::load(&matvec_iq4_xs_wave64_hsaco)?,
            matvec_f16_wave64_module:    Module::load(&matvec_f16_wave64_hsaco)?,
            blocks,
            stream,
            hidden, ffn, vocab, n_heads, n_kv_heads, head_dim, rotary_dim,
            gdn_value_dim, gdn_conv_dim, gdn_n_heads, gdn_head_dim, gdn_conv_kernel,
            gdn_qkv, gdn_conv_out, gdn_z, gdn_a, gdn_b, gdn_q, gdn_k,
            gdn_decay, gdn_beta, gdn_core_out, gdn_delta,
            rms_eps: cfg.rms_norm_eps,
            max_seq,
        })
    }

    /// q_dim = n_heads * head_dim
    pub fn q_dim(&self) -> usize { self.n_heads * self.head_dim }
    /// kv_dim = n_kv_heads * head_dim
    pub fn kv_dim(&self) -> usize { self.n_kv_heads * self.head_dim }

    /// The matvec tensor used for the final output projection. Same as
    /// `output_proj` if separate; falls back to `token_embd` if tied.
    fn output_proj_tensor(&self) -> &GpuMatvecTensor {
        self.output_proj.as_ref().unwrap_or(&self.token_embd)
    }

    // ---- Per-op launchers ---------------------------------------------------
    //
    // These take raw device pointers so callers can chain them without each
    // op needing to know about DeviceBuf<T>. They never allocate or sync —
    // sync is the caller's responsibility before reading results back.

    fn launch_embed_lookup(&self, table: *mut c_void, out: *mut c_void,
                           token: u32, n: u32) -> Result<(), String>
    {
        let f = self.embed_module.function("embed_lookup_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut t = table; let mut o = out; let mut row = token; let mut nn = n;
        let mut args: [*mut c_void; 4] = [
            &mut t   as *mut _ as *mut c_void,
            &mut o   as *mut _ as *mut c_void,
            &mut row as *mut _ as *mut c_void,
            &mut nn  as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_embed_lookup_q6_k(&self, table: *mut c_void, out: *mut c_void,
                                token: u32, hidden: u32) -> Result<(), String>
    {
        let f = self.embed_lookup_q6_k_module.function("embed_lookup_q6_k_f32")?;
        // One HIP block per Q6_K super-block (256 weights), 256 threads each.
        let block: u32 = 256;
        let grid = hidden / 256;
        let mut t = table; let mut o = out; let mut row = token; let mut h = hidden;
        let mut args: [*mut c_void; 4] = [
            &mut t   as *mut _ as *mut c_void,
            &mut o   as *mut _ as *mut c_void,
            &mut row as *mut _ as *mut c_void,
            &mut h   as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_embed_lookup_q4_k(&self, table: *mut c_void, out: *mut c_void,
                                token: u32, hidden: u32) -> Result<(), String>
    {
        let f = self.embed_lookup_q4_k_module.function("embed_lookup_q4_k_f32")?;
        // One HIP block per Q4_K super-block (256 weights), 256 threads each.
        let block: u32 = 256;
        let grid = hidden / 256;
        let mut t = table; let mut o = out; let mut row = token; let mut h = hidden;
        let mut args: [*mut c_void; 4] = [
            &mut t   as *mut _ as *mut c_void,
            &mut o   as *mut _ as *mut c_void,
            &mut row as *mut _ as *mut c_void,
            &mut h   as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    /// Gather one row from `table` (shape `[hidden, vocab]` in GGUF order)
    /// and write the dequantised fp32 row into `out`. Dispatches by the
    /// table's on-disk dtype.
    fn launch_embed_lookup_dispatch(&self, table: &GpuMatvecTensor, out: *mut c_void,
                                    token: u32) -> Result<(), String>
    {
        let hidden = table.in_dim;  // first dim of [hidden, vocab]
        match table.dtype {
            GgmlType::F32  => self.launch_embed_lookup(table.data.raw_ptr(), out, token, hidden),
            GgmlType::Q6_K => self.launch_embed_lookup_q6_k(table.data.raw_ptr(), out, token, hidden),
            GgmlType::Q4_K => self.launch_embed_lookup_q4_k(table.data.raw_ptr(), out, token, hidden),
            other => Err(format!("embed_lookup: no kernel for {:?}", other)),
        }
    }

    fn launch_rmsnorm(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void,
                      n: u32, eps: f32) -> Result<(), String>
    {
        let f = self.rmsnorm_module.function("rmsnorm_f32")?;
        let block: u32 = 256;
        let mut xa = x; let mut wa = w; let mut ya = y;
        let mut na = n; let mut ea = eps;
        let mut args: [*mut c_void; 5] = [
            &mut xa as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void,
        ];
        let smem = block * std::mem::size_of::<f32>() as u32;
        unsafe { f.launch((1, 1, 1), (block, 1, 1), smem, Some(&self.stream), &mut args) }
    }

    fn launch_matvec(&self, w: *mut c_void, x: *mut c_void, y: *mut c_void,
                     in_dim: u32, out_dim: u32) -> Result<(), String>
    {
        let f = self.matvec_module.function("matvec_f32")?;
        let block: u32 = 256;
        let mut wa = w; let mut xa = x; let mut ya = y;
        let mut ia = in_dim; let mut oa = out_dim;
        let mut args: [*mut c_void; 5] = [
            &mut wa as *mut _ as *mut c_void,
            &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
        ];
        let smem = block * std::mem::size_of::<f32>() as u32;
        unsafe { f.launch((out_dim, 1, 1), (block, 1, 1), smem, Some(&self.stream), &mut args) }
    }

    /// Per-quant-type matvec launchers — same signature as launch_matvec.
    /// All five fused dequant+GEMV kernels share the (W bytes, x f32, y f32,
    /// in_dim, out_dim) interface.
    fn launch_matvec_q_kernel(&self, module: &Module, kname: &str,
                              w: *mut c_void, x: *mut c_void, y: *mut c_void,
                              in_dim: u32, out_dim: u32) -> Result<(), String>
    {
        let f = module.function(kname)?;
        let block: u32 = 256;
        let mut wa = w; let mut xa = x; let mut ya = y;
        let mut ia = in_dim; let mut oa = out_dim;
        let mut args: [*mut c_void; 5] = [
            &mut wa as *mut _ as *mut c_void,
            &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
        ];
        let smem = block * std::mem::size_of::<f32>() as u32;
        unsafe { f.launch((out_dim, 1, 1), (block, 1, 1), smem, Some(&self.stream), &mut args) }
    }

    /// Wave-cooperative launcher: 64 threads (one wavefront) per output
    /// row, no shared memory, reduction via __shfl_xor inside the kernel.
    fn launch_matvec_wave64(&self, module: &Module, kname: &str,
                            w: *mut c_void, x: *mut c_void, y: *mut c_void,
                            in_dim: u32, out_dim: u32) -> Result<(), String>
    {
        let f = module.function(kname)?;
        let block: u32 = 64;
        let mut wa = w; let mut xa = x; let mut ya = y;
        let mut ia = in_dim; let mut oa = out_dim;
        let mut args: [*mut c_void; 5] = [
            &mut wa as *mut _ as *mut c_void,
            &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((out_dim, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    /// Dispatch a matvec to the right kernel based on the weight's on-disk
    /// dtype. Output `y` always lands as fp32.
    fn launch_matvec_dispatch(&self, w: &GpuMatvecTensor,
                              x: *mut c_void, y: *mut c_void) -> Result<(), String>
    {
        let in_d  = w.in_dim;
        let out_d = w.out_dim;
        let wp    = w.data.raw_ptr();
        match w.dtype {
            GgmlType::F32    => self.launch_matvec_wave64(&self.matvec_f32_wave64_module,
                                    "matvec_f32_wave64", wp, x, y, in_d, out_d),
            GgmlType::Q8_0   => self.launch_matvec_wave64(&self.matvec_q8_0_wave64_module,
                                    "matvec_q8_0_wave64_f32", wp, x, y, in_d, out_d),
            GgmlType::Q4_K   => self.launch_matvec_wave64(&self.matvec_q4_k_wave64_module,
                                    "matvec_q4_k_wave64_f32", wp, x, y, in_d, out_d),
            GgmlType::Q5_K   => self.launch_matvec_wave64(&self.matvec_q5_k_wave64_module,
                                    "matvec_q5_k_wave64_f32", wp, x, y, in_d, out_d),
            GgmlType::Q6_K   => self.launch_matvec_wave64(&self.matvec_q6_k_wave64_module,
                                    "matvec_q6_k_wave64_f32", wp, x, y, in_d, out_d),
            GgmlType::IQ4_XS => self.launch_matvec_wave64(&self.matvec_iq4_xs_wave64_module,
                                    "matvec_iq4_xs_wave64_f32", wp, x, y, in_d, out_d),
            GgmlType::F16    => self.launch_matvec_wave64(&self.matvec_f16_wave64_module,
                                    "matvec_f16_wave64_f32", wp, x, y, in_d, out_d),
            other => Err(format!("matvec dispatch: no kernel for {:?}", other)),
        }
    }

    fn launch_swiglu(&self, gate: *mut c_void, up: *mut c_void, out: *mut c_void,
                     n: u32) -> Result<(), String>
    {
        let f = self.swiglu_module.function("swiglu_mul_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut ga = gate; let mut ua = up; let mut oa = out; let mut na = n;
        let mut args: [*mut c_void; 4] = [
            &mut ga as *mut _ as *mut c_void,
            &mut ua as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_rmsnorm_multihead(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void,
                                n_heads: u32, head_dim: u32, eps: f32) -> Result<(), String>
    {
        let f = self.rmsnorm_multihead_module.function("rmsnorm_multihead_f32")?;
        let block: u32 = 256;
        let mut xa = x; let mut wa = w; let mut ya = y;
        let mut nh = n_heads; let mut hd = head_dim; let mut ea = eps;
        let mut args: [*mut c_void; 6] = [
            &mut xa as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void,
        ];
        let smem = block * std::mem::size_of::<f32>() as u32;
        unsafe { f.launch((n_heads, 1, 1), (block, 1, 1), smem, Some(&self.stream), &mut args) }
    }

    fn launch_split_q_gate(&self, q_raw: *mut c_void, q: *mut c_void, gate: *mut c_void,
                           n_heads: u32, head_dim: u32) -> Result<(), String>
    {
        let f = self.split_q_gate_module.function("split_q_gate_f32")?;
        let block: u32 = 256;
        let total = n_heads * head_dim;
        let grid = (total + block - 1) / block;
        let mut qra = q_raw; let mut qa = q; let mut ga = gate;
        let mut nh = n_heads; let mut hd = head_dim;
        let mut args: [*mut c_void; 5] = [
            &mut qra as *mut _ as *mut c_void,
            &mut qa  as *mut _ as *mut c_void,
            &mut ga  as *mut _ as *mut c_void,
            &mut nh  as *mut _ as *mut c_void,
            &mut hd  as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_sigmoid_mul(&self, x: *mut c_void, gate: *mut c_void, n: u32)
        -> Result<(), String>
    {
        let f = self.sigmoid_mul_module.function("sigmoid_mul_inplace_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut xa = x; let mut ga = gate; let mut na = n;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void,
            &mut ga as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_rope(&self, x: *mut c_void, n_heads: u32, pos: u32) -> Result<(), String> {
        let f = self.rope_module.function("rope_apply_f32")?;
        let half = (self.rotary_dim / 2) as u32;
        let block: u32 = 64;
        let grid_x = (half + block - 1) / block;
        let mut xa = x;
        let mut ca = self.rope_cos.raw_ptr();
        let mut sa = self.rope_sin.raw_ptr();
        let mut hd = self.head_dim   as u32;
        let mut rd = self.rotary_dim as u32;
        let mut nh = n_heads;
        let mut p  = pos;
        let mut args: [*mut c_void; 7] = [
            &mut xa as *mut _ as *mut c_void,
            &mut ca as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut rd as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut p  as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid_x, n_heads, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_conv1d_step(&self, x_new: *mut c_void, w: *mut c_void, hist: *mut c_void,
                          y: *mut c_void, n_channels: u32, kernel_size: u32)
        -> Result<(), String>
    {
        let f = self.conv1d_step_module.function("conv1d_step_f32")?;
        let block: u32 = 256;
        let grid = (n_channels + block - 1) / block;
        let mut xa = x_new; let mut wa = w; let mut ha = hist; let mut ya = y;
        let mut nc = n_channels; let mut ks = kernel_size;
        let mut args: [*mut c_void; 6] = [
            &mut xa as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void,
            &mut ha as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut ks as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_silu_inplace(&self, x: *mut c_void, n: u32) -> Result<(), String> {
        let f = self.silu_inplace_module.function("silu_inplace_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut xa = x; let mut na = n;
        let mut args: [*mut c_void; 2] = [
            &mut xa as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_l2norm_multihead(&self, x: *mut c_void, y: *mut c_void,
                                n_heads: u32, head_dim: u32, eps: f32, scale: f32)
        -> Result<(), String>
    {
        let f = self.l2norm_multihead_module.function("l2norm_multihead_f32")?;
        let block: u32 = 128;
        let mut xa = x; let mut ya = y; let mut nh = n_heads; let mut hd = head_dim;
        let mut ea = eps; let mut sa = scale;
        let mut args: [*mut c_void; 6] = [
            &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void,
        ];
        let smem = block * std::mem::size_of::<f32>() as u32;
        unsafe { f.launch((n_heads, 1, 1), (block, 1, 1), smem, Some(&self.stream), &mut args) }
    }

    fn launch_gdn_decay_beta(&self, a: *mut c_void, b: *mut c_void,
                             ssm_a: *mut c_void, dt_bias: *mut c_void,
                             decay: *mut c_void, beta: *mut c_void, n_heads: u32)
        -> Result<(), String>
    {
        let f = self.gdn_decay_beta_module.function("gdn_decay_beta_f32")?;
        let block: u32 = 64;
        let grid = (n_heads + block - 1) / block;
        let mut aa = a; let mut bb = b; let mut sa = ssm_a; let mut da = dt_bias;
        let mut dca = decay; let mut beta_a = beta; let mut nh = n_heads;
        let mut args: [*mut c_void; 7] = [
            &mut aa     as *mut _ as *mut c_void,
            &mut bb     as *mut _ as *mut c_void,
            &mut sa     as *mut _ as *mut c_void,
            &mut da     as *mut _ as *mut c_void,
            &mut dca    as *mut _ as *mut c_void,
            &mut beta_a as *mut _ as *mut c_void,
            &mut nh     as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_gdn_recurrent_step(&self,
        q: *mut c_void, k: *mut c_void, v: *mut c_void,
        decay: *mut c_void, beta: *mut c_void,
        state: *mut c_void, out: *mut c_void,
        n_heads: u32, head_dim: u32) -> Result<(), String>
    {
        let f = self.gdn_recurrent_step_module.function("gdn_recurrent_step_f32")?;
        let block: u32 = head_dim;
        let smem = 4 * head_dim * std::mem::size_of::<f32>() as u32;
        let mut qa = q; let mut ka = k; let mut va = v;
        let mut da = decay; let mut ba = beta;
        let mut sa = state; let mut oa = out;
        let mut nh = n_heads; let mut hd = head_dim;
        let mut args: [*mut c_void; 9] = [
            &mut qa as *mut _ as *mut c_void,
            &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void,
            &mut da as *mut _ as *mut c_void,
            &mut ba as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((n_heads, 1, 1), (block, 1, 1), smem, Some(&self.stream), &mut args) }
    }

    fn launch_gdn_recurrent_step_lds(&self,
        q: *mut c_void, k: *mut c_void, v: *mut c_void,
        decay: *mut c_void, beta: *mut c_void,
        state: *mut c_void, out: *mut c_void, delta_scratch: *mut c_void,
        n_heads: u32, head_dim: u32) -> Result<(), String>
    {
        let f = self.gdn_recurrent_step_lds_module.function("gdn_recurrent_step_lds_f32")?;
        let block: u32 = head_dim;
        // Dynamic LDS = state matrix only (head_dim * head_dim floats).
        let smem = head_dim * head_dim * std::mem::size_of::<f32>() as u32;
        let mut qa = q; let mut ka = k; let mut va = v;
        let mut da = decay; let mut ba = beta;
        let mut sa = state; let mut oa = out; let mut dla = delta_scratch;
        let mut nh = n_heads; let mut hd = head_dim;
        let mut args: [*mut c_void; 10] = [
            &mut qa  as *mut _ as *mut c_void,
            &mut ka  as *mut _ as *mut c_void,
            &mut va  as *mut _ as *mut c_void,
            &mut da  as *mut _ as *mut c_void,
            &mut ba  as *mut _ as *mut c_void,
            &mut sa  as *mut _ as *mut c_void,
            &mut oa  as *mut _ as *mut c_void,
            &mut dla as *mut _ as *mut c_void,
            &mut nh  as *mut _ as *mut c_void,
            &mut hd  as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((n_heads, 1, 1), (block, 1, 1), smem, Some(&self.stream), &mut args) }
    }

    fn launch_conv1d_step_silu(&self, x_new: *mut c_void, w: *mut c_void, hist: *mut c_void,
                               y: *mut c_void, n_channels: u32, kernel_size: u32)
        -> Result<(), String>
    {
        let f = self.conv1d_step_silu_module.function("conv1d_step_silu_f32")?;
        let block: u32 = 256;
        let grid = (n_channels + block - 1) / block;
        let mut xa = x_new; let mut wa = w; let mut ha = hist; let mut ya = y;
        let mut nc = n_channels; let mut ks = kernel_size;
        let mut args: [*mut c_void; 6] = [
            &mut xa as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void,
            &mut ha as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut ks as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_l2norm_qk(&self, q_in: *mut c_void, q_out: *mut c_void,
                        k_in: *mut c_void, k_out: *mut c_void,
                        n_heads: u32, head_dim: u32, eps: f32, q_scale: f32)
        -> Result<(), String>
    {
        let f = self.l2norm_qk_module.function("l2norm_qk_f32")?;
        let block: u32 = 128;
        let mut qi = q_in; let mut qo = q_out; let mut ki = k_in; let mut ko = k_out;
        let mut nh = n_heads; let mut hd = head_dim; let mut ea = eps; let mut sc = q_scale;
        let mut args: [*mut c_void; 8] = [
            &mut qi as *mut _ as *mut c_void,
            &mut qo as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
            &mut ko as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let smem = block * std::mem::size_of::<f32>() as u32;
        // 2D grid: x = head index, y = side (0 = Q, 1 = K).
        unsafe { f.launch((n_heads, 2, 1), (block, 1, 1), smem, Some(&self.stream), &mut args) }
    }

    fn launch_gdn_recurrent_step_fused(&self,
        q: *mut c_void, k: *mut c_void, v: *mut c_void,
        a: *mut c_void, b: *mut c_void, ssm_a: *mut c_void, dt_bias: *mut c_void,
        state: *mut c_void, out: *mut c_void,
        n_heads: u32, head_dim: u32) -> Result<(), String>
    {
        let f = self.gdn_recurrent_step_fused_module.function("gdn_recurrent_step_fused_f32")?;
        let block: u32 = head_dim;
        let smem = 4 * head_dim * std::mem::size_of::<f32>() as u32;
        let mut qa = q; let mut ka = k; let mut va = v;
        let mut aa = a; let mut ba = b; let mut sma = ssm_a; let mut dta = dt_bias;
        let mut sa = state; let mut oa = out;
        let mut nh = n_heads; let mut hd = head_dim;
        let mut args: [*mut c_void; 11] = [
            &mut qa  as *mut _ as *mut c_void,
            &mut ka  as *mut _ as *mut c_void,
            &mut va  as *mut _ as *mut c_void,
            &mut aa  as *mut _ as *mut c_void,
            &mut ba  as *mut _ as *mut c_void,
            &mut sma as *mut _ as *mut c_void,
            &mut dta as *mut _ as *mut c_void,
            &mut sa  as *mut _ as *mut c_void,
            &mut oa  as *mut _ as *mut c_void,
            &mut nh  as *mut _ as *mut c_void,
            &mut hd  as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((n_heads, 1, 1), (block, 1, 1), smem, Some(&self.stream), &mut args) }
    }

    fn launch_rmsnorm_gated_multihead(&self, x: *mut c_void, z: *mut c_void, w: *mut c_void,
                                      y: *mut c_void, n_heads: u32, head_dim: u32, eps: f32)
        -> Result<(), String>
    {
        let f = self.rmsnorm_gated_multihead_module.function("rmsnorm_gated_multihead_f32")?;
        let block: u32 = 128;
        let mut xa = x; let mut za = z; let mut wa = w; let mut ya = y;
        let mut nh = n_heads; let mut hd = head_dim; let mut ea = eps;
        let mut args: [*mut c_void; 7] = [
            &mut xa as *mut _ as *mut c_void,
            &mut za as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void,
        ];
        let smem = block * std::mem::size_of::<f32>() as u32;
        unsafe { f.launch((n_heads, 1, 1), (block, 1, 1), smem, Some(&self.stream), &mut args) }
    }

    fn launch_add_inplace(&self, x: *mut c_void, y: *mut c_void, n: u32) -> Result<(), String> {
        let f = self.add_inplace_module.function("add_inplace_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut xa = x; let mut ya = y; let mut na = n;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_attn_step(&self, q: *mut c_void, k_cache: *mut c_void, v_cache: *mut c_void,
                        out: *mut c_void, total_len: u32, scaling: f32) -> Result<(), String>
    {
        let f = self.attn_step_module.function("attn_step_f32")?;
        let block: u32 = 256;
        let grid: u32 = self.n_heads as u32;
        let head_dim = self.head_dim as u32;
        let smem = ((head_dim + total_len) + block) * std::mem::size_of::<f32>() as u32;
        let mut qa = q; let mut ka = k_cache; let mut va = v_cache; let mut oa = out;
        let mut nh = self.n_heads as u32;
        let mut nkv = self.n_kv_heads as u32;
        let mut hd = head_dim;
        let mut tl = total_len;
        let mut sc = scaling;
        let mut args: [*mut c_void; 9] = [
            &mut qa as *mut _ as *mut c_void,
            &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut tl as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), smem, Some(&self.stream), &mut args) }
    }

    /// embed → output_norm → output_proj. Returns vocab-length logits.
    /// Composition is artificial (norm doesn't belong here in real
    /// forward), but every kernel and every device pointer in the
    /// pipeline is exercised.
    pub fn embed_norm_proj(&self, token: u32) -> Result<Vec<f32>, String> {
        self.launch_embed_lookup_dispatch(&self.token_embd, self.hidden_a.raw_ptr(), token)?;
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), self.output_norm.raw_ptr(),
                            self.hidden_b.raw_ptr(), self.hidden as u32, self.rms_eps)?;
        self.launch_matvec_dispatch(self.output_proj_tensor(),
                                    self.hidden_b.raw_ptr(), self.logits.raw_ptr())?;
        self.stream.synchronize()?;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// Device-pointer attention step. Reads `input_ptr`, writes the
    /// attention sub-layer output (post-projection, pre-residual) to
    /// `output_ptr`. `input_ptr` is read-only here — must NOT alias
    /// `output_ptr`. No H2D/D2H/sync.
    fn step_full_attention(&self,
        input_ptr: *mut c_void, output_ptr: *mut c_void,
        weights: &GpuFullAttnWeights, kv_cache: &mut GpuKvCache,
    ) -> Result<(), String>
    {
        assert!(kv_cache.len < kv_cache.max_seq, "KV cache full");
        let h_dim  = self.hidden as u32;
        let q_dim  = self.q_dim()  as u32;
        let pos = kv_cache.len;
        let scaling = (self.head_dim as f32).powf(-0.5);

        // normed → output_ptr (output_ptr serves dual duty: normed first,
        //                      then final attn output overwrites it)
        self.launch_rmsnorm(input_ptr, weights.attn_norm.raw_ptr(),
                            output_ptr, h_dim, self.rms_eps)?;
        self.launch_matvec_dispatch(&weights.attn_q, output_ptr, self.q_raw.raw_ptr())?;
        self.launch_matvec_dispatch(&weights.attn_k, output_ptr, self.k_raw.raw_ptr())?;
        self.launch_matvec_dispatch(&weights.attn_v, output_ptr, self.v_raw.raw_ptr())?;
        self.launch_split_q_gate(self.q_raw.raw_ptr(), self.q_buf.raw_ptr(),
                                 self.gate_buf.raw_ptr(),
                                 self.n_heads as u32, self.head_dim as u32)?;
        self.launch_rmsnorm_multihead(self.q_buf.raw_ptr(), weights.attn_q_norm.raw_ptr(),
                                      self.q_buf.raw_ptr(),
                                      self.n_heads as u32, self.head_dim as u32, self.rms_eps)?;
        self.launch_rope(self.q_buf.raw_ptr(), self.n_heads as u32, pos as u32)?;
        self.launch_rmsnorm_multihead(self.k_raw.raw_ptr(), weights.attn_k_norm.raw_ptr(),
                                      self.k_norm.raw_ptr(),
                                      self.n_kv_heads as u32, self.head_dim as u32, self.rms_eps)?;
        self.launch_rope(self.k_norm.raw_ptr(), self.n_kv_heads as u32, pos as u32)?;
        // Async D2D push on the same stream — no host sync needed; ordering
        // against preceding kernel launches is preserved by stream semantics.
        kv_cache.k.copy_from_device_at_async(&self.k_norm, pos * kv_cache.kv_dim, &self.stream)?;
        kv_cache.v.copy_from_device_at_async(&self.v_raw,  pos * kv_cache.kv_dim, &self.stream)?;
        let total_len = (pos + 1) as u32;
        self.launch_attn_step(self.q_buf.raw_ptr(),
                              kv_cache.k.raw_ptr(), kv_cache.v.raw_ptr(),
                              self.attn_concat.raw_ptr(), total_len, scaling)?;
        self.launch_sigmoid_mul(self.attn_concat.raw_ptr(), self.gate_buf.raw_ptr(), q_dim)?;
        self.launch_matvec_dispatch(&weights.attn_output, self.attn_concat.raw_ptr(), output_ptr)?;
        kv_cache.len += 1;
        Ok(())
    }

    /// Device-pointer FFN step. `input_ptr == output_ptr` is allowed
    /// (gate/up matvecs run before down writes back). No H2D/D2H/sync.
    fn step_swiglu_ffn(&self,
        input_ptr: *mut c_void, output_ptr: *mut c_void,
        weights: &GpuFfnWeights,
    ) -> Result<(), String>
    {
        let f = self.ffn as u32;
        self.launch_matvec_dispatch(&weights.gate, input_ptr, self.ffn_a.raw_ptr())?;
        self.launch_matvec_dispatch(&weights.up,   input_ptr, self.ffn_b.raw_ptr())?;
        self.launch_swiglu(self.ffn_a.raw_ptr(), self.ffn_b.raw_ptr(),
                           self.ffn_a.raw_ptr(), f)?;
        self.launch_matvec_dispatch(&weights.down, self.ffn_a.raw_ptr(), output_ptr)?;
        Ok(())
    }

    /// On-device "full transformer block" composer: takes a hidden_io
    /// buffer (mutated in place by both residual sums) and a scratch
    /// buffer (overwritten three times — first as attn_out, then as
    /// post-norm output, then as ffn_out). No H2D / D2H / sync.
    fn step_full_attention_block_dev(&self,
        hidden_io: *mut c_void, scratch: *mut c_void,
        weights: &GpuFullAttnBlock, kv_cache: &mut GpuKvCache,
    ) -> Result<(), String>
    {
        let h = self.hidden as u32;
        self.step_full_attention(hidden_io, scratch, &weights.attn, kv_cache)?;
        self.launch_add_inplace(hidden_io, scratch, h)?;
        self.launch_rmsnorm(hidden_io, weights.post_norm.raw_ptr(), scratch, h, self.rms_eps)?;
        self.step_swiglu_ffn(scratch, scratch, &weights.ffn)?;
        self.launch_add_inplace(hidden_io, scratch, h)?;
        Ok(())
    }

    /// On-device "linear (GDN) transformer block" composer.
    fn step_linear_attention_block_dev(&self,
        hidden_io: *mut c_void, scratch: *mut c_void,
        weights: &GpuLinAttnBlock, state: &mut GpuLinAttnState,
    ) -> Result<(), String>
    {
        let h = self.hidden as u32;
        self.step_linear_attention(hidden_io, scratch, &weights.attn, state)?;
        self.launch_add_inplace(hidden_io, scratch, h)?;
        self.launch_rmsnorm(hidden_io, weights.post_norm.raw_ptr(), scratch, h, self.rms_eps)?;
        self.step_swiglu_ffn(scratch, scratch, &weights.ffn)?;
        self.launch_add_inplace(hidden_io, scratch, h)?;
        Ok(())
    }

    /// End-to-end forward pass for one decode token. Mirrors
    /// `cpu::qwen3_5::Qwen35F32Model::forward_token`.
    ///
    ///   embed_lookup(token) → hidden_a
    ///   for each block in schedule:
    ///       block_step(hidden_a, hidden_b, w, state)
    ///   output_norm(hidden_a) → hidden_b
    ///   output_proj(hidden_b) → logits
    ///   sync, D2H logits
    ///
    /// State advances by one position per block.
    pub fn forward_token(&self, token: u32, state: &mut Qwen35GpuState)
        -> Result<Vec<f32>, String>
    {
        self.enqueue_forward_token(token, state)?;
        self.stream.synchronize()?;
        state.pos += 1;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// Diagnostic: run one decode step where every kernel inside one
    /// chosen linear-attention block is bracketed with HIP events.
    /// Returns (logits, list of (name, ms) pairs) for the block at
    /// `traced_block_idx`. Other blocks run normally.
    pub fn forward_token_traced_gdn(&self, token: u32, state: &mut Qwen35GpuState,
                                    traced_block_idx: usize)
        -> Result<(Vec<f32>, Vec<(&'static str, f32)>), String>
    {
        assert_eq!(state.block_states.len(), self.blocks.len());
        let h_dim     = self.hidden        as u32;
        let value_dim = self.gdn_value_dim as u32;
        let conv_dim  = self.gdn_conv_dim  as u32;
        let n_heads   = self.gdn_n_heads   as u32;
        let head_dim  = self.gdn_head_dim  as u32;
        let q_scale   = (self.gdn_head_dim as f32).powf(-0.5);

        // Embed lookup → hidden_a
        self.launch_embed_lookup_dispatch(&self.token_embd, self.hidden_a.raw_ptr(), token)?;

        // Walk blocks, but for `traced_block_idx` (which must be a Linear
        // block) we expand the chain manually with events between kernels.
        let mut traced_events: Vec<(&'static str, Event, Event)> = Vec::new();
        for (i, (block, st)) in self.blocks.iter().zip(state.block_states.iter_mut()).enumerate() {
            if i != traced_block_idx {
                match (block, st) {
                    (GpuBlock::Full(w), GpuBlockState::Full(kv)) => {
                        self.step_full_attention_block_dev(
                            self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), w, kv)?;
                    }
                    (GpuBlock::Linear(w), GpuBlockState::Linear(s)) => {
                        self.step_linear_attention_block_dev(
                            self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), w, s)?;
                    }
                    _ => return Err("block kind mismatch".into()),
                }
                continue;
            }

            // Traced block — must be Linear.
            let (w, lstate) = match (block, st) {
                (GpuBlock::Linear(w), GpuBlockState::Linear(s)) => (w, s),
                _ => return Err("traced block must be LinearAttention".into()),
            };

            // Helper: wrap a closure in HIP events and append to the trace.
            macro_rules! traced {
                ($name:expr, $body:expr) => {{
                    let s = Event::new()?;  s.record(&self.stream)?;
                    $body?;
                    let e = Event::new()?;  e.record(&self.stream)?;
                    traced_events.push(($name, s, e));
                }};
            }

            // attn pre-norm (output_ptr = hidden_b serves as scratch)
            traced!("attn_norm", self.launch_rmsnorm(self.hidden_a.raw_ptr(),
                w.attn.attn_norm.raw_ptr(), self.hidden_b.raw_ptr(), h_dim, self.rms_eps));
            traced!("matvec_attn_qkv", self.launch_matvec_dispatch(&w.attn.attn_qkv,
                self.hidden_b.raw_ptr(), self.gdn_qkv.raw_ptr()));
            traced!("matvec_attn_gate", self.launch_matvec_dispatch(&w.attn.attn_gate,
                self.hidden_b.raw_ptr(), self.gdn_z.raw_ptr()));
            traced!("matvec_ssm_alpha", self.launch_matvec_dispatch(&w.attn.ssm_alpha,
                self.hidden_b.raw_ptr(), self.gdn_a.raw_ptr()));
            traced!("matvec_ssm_beta", self.launch_matvec_dispatch(&w.attn.ssm_beta,
                self.hidden_b.raw_ptr(), self.gdn_b.raw_ptr()));
            traced!("conv1d_step_silu", self.launch_conv1d_step_silu(self.gdn_qkv.raw_ptr(),
                w.attn.ssm_conv1d.raw_ptr(), lstate.conv_hist.raw_ptr(),
                self.gdn_conv_out.raw_ptr(), conv_dim, self.gdn_conv_kernel as u32));
            let conv_out_ptr = self.gdn_conv_out.raw_ptr() as *mut f32;
            let q_in_ptr = unsafe { conv_out_ptr.add(0)                      } as *mut c_void;
            let k_in_ptr = unsafe { conv_out_ptr.add(self.gdn_value_dim)     } as *mut c_void;
            let v_in_ptr = unsafe { conv_out_ptr.add(2 * self.gdn_value_dim) } as *mut c_void;
            traced!("l2norm_qk", self.launch_l2norm_qk(q_in_ptr, self.gdn_q.raw_ptr(),
                k_in_ptr, self.gdn_k.raw_ptr(), n_heads, head_dim, 1e-6, q_scale));
            traced!("recurrent_step_fused", self.launch_gdn_recurrent_step_fused(
                self.gdn_q.raw_ptr(), self.gdn_k.raw_ptr(), v_in_ptr,
                self.gdn_a.raw_ptr(), self.gdn_b.raw_ptr(),
                w.attn.ssm_a.raw_ptr(), w.attn.ssm_dt_bias.raw_ptr(),
                lstate.recurrent.raw_ptr(), self.gdn_core_out.raw_ptr(),
                n_heads, head_dim));
            traced!("rmsnorm_gated", self.launch_rmsnorm_gated_multihead(
                self.gdn_core_out.raw_ptr(), self.gdn_z.raw_ptr(),
                w.attn.ssm_norm.raw_ptr(), self.gdn_core_out.raw_ptr(),
                n_heads, head_dim, self.rms_eps));
            traced!("matvec_ssm_out", self.launch_matvec_dispatch(&w.attn.ssm_out,
                self.gdn_core_out.raw_ptr(), self.hidden_b.raw_ptr()));
            // Post-block residual + ffn (untraced)
            self.launch_add_inplace(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), h_dim)?;
            self.launch_rmsnorm(self.hidden_a.raw_ptr(), w.post_norm.raw_ptr(),
                                self.hidden_b.raw_ptr(), h_dim, self.rms_eps)?;
            self.step_swiglu_ffn(self.hidden_b.raw_ptr(), self.hidden_b.raw_ptr(), &w.ffn)?;
            self.launch_add_inplace(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), h_dim)?;
        }

        // Output norm + projection
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), self.output_norm.raw_ptr(),
                            self.hidden_b.raw_ptr(), h_dim, self.rms_eps)?;
        self.launch_matvec_dispatch(self.output_proj_tensor(),
                                    self.hidden_b.raw_ptr(), self.logits.raw_ptr())?;
        self.stream.synchronize()?;
        state.pos += 1;

        let mut trace = Vec::with_capacity(traced_events.len());
        for (name, s, e) in &traced_events {
            trace.push((*name, Event::elapsed_time(s, e)?));
        }
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok((out, trace))
    }

    /// Like `forward_token` but records per-stage GPU times via HIP
    /// events. Adds ~N+3 events per call, plus one elapsed_time query
    /// per stage at the end — small overhead but not free, so reserve
    /// for diagnostics, not the inner loop.
    pub fn forward_token_traced(&self, token: u32, state: &mut Qwen35GpuState)
        -> Result<(Vec<f32>, GpuForwardTrace), String>
    {
        assert_eq!(state.block_states.len(), self.blocks.len());
        let n_blocks = self.blocks.len();
        // Checkpoints: e0 before embed, e1 after embed = before block 0,
        // e[i+2] after block i, e[n+2] after output_norm, e[n+3] after output_proj.
        let events: Vec<Event> = (0..n_blocks + 4)
            .map(|_| Event::new())
            .collect::<Result<Vec<_>, _>>()?;

        events[0].record(&self.stream)?;
        self.launch_embed_lookup_dispatch(&self.token_embd, self.hidden_a.raw_ptr(), token)?;
        events[1].record(&self.stream)?;

        for (i, (block, st)) in self.blocks.iter().zip(state.block_states.iter_mut()).enumerate() {
            match (block, st) {
                (GpuBlock::Full(w), GpuBlockState::Full(kv)) => {
                    self.step_full_attention_block_dev(
                        self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), w, kv)?;
                }
                (GpuBlock::Linear(w), GpuBlockState::Linear(s)) => {
                    self.step_linear_attention_block_dev(
                        self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), w, s)?;
                }
                _ => return Err("block kind mismatch".into()),
            }
            events[i + 2].record(&self.stream)?;
        }

        self.launch_rmsnorm(self.hidden_a.raw_ptr(), self.output_norm.raw_ptr(),
                            self.hidden_b.raw_ptr(), self.hidden as u32, self.rms_eps)?;
        events[n_blocks + 2].record(&self.stream)?;

        self.launch_matvec_dispatch(self.output_proj_tensor(),
                                    self.hidden_b.raw_ptr(), self.logits.raw_ptr())?;
        events[n_blocks + 3].record(&self.stream)?;

        // Sync on the *last* event (finishes the chain) before reading.
        events[n_blocks + 3].synchronize()?;
        state.pos += 1;

        let mut block_ms = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            block_ms.push(Event::elapsed_time(&events[i + 1], &events[i + 2])?);
        }
        let trace = GpuForwardTrace {
            embed_ms:       Event::elapsed_time(&events[0],            &events[1])?,
            block_ms,
            output_norm_ms: Event::elapsed_time(&events[n_blocks + 1], &events[n_blocks + 2])?,
            output_proj_ms: Event::elapsed_time(&events[n_blocks + 2], &events[n_blocks + 3])?,
            total_ms:       Event::elapsed_time(&events[0],            &events[n_blocks + 3])?,
        };

        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok((out, trace))
    }

    /// On-device portion of `forward_token`: every kernel launch and
    /// every async memcpy is enqueued to `self.stream` with no host
    /// syncs in between. Used both for direct execution (followed by
    /// stream-sync + D2H of `self.logits`) and for HIP graph capture.
    fn enqueue_forward_token(&self, token: u32, state: &mut Qwen35GpuState)
        -> Result<(), String>
    {
        assert_eq!(state.block_states.len(), self.blocks.len());
        self.launch_embed_lookup_dispatch(&self.token_embd, self.hidden_a.raw_ptr(), token)?;
        for (block, st) in self.blocks.iter().zip(state.block_states.iter_mut()) {
            match (block, st) {
                (GpuBlock::Full(w), GpuBlockState::Full(kv)) => {
                    self.step_full_attention_block_dev(
                        self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), w, kv)?;
                }
                (GpuBlock::Linear(w), GpuBlockState::Linear(s)) => {
                    self.step_linear_attention_block_dev(
                        self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), w, s)?;
                }
                _ => return Err("block kind mismatch between weights and state".into()),
            }
        }
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), self.output_norm.raw_ptr(),
                            self.hidden_b.raw_ptr(), self.hidden as u32, self.rms_eps)?;
        self.launch_matvec_dispatch(self.output_proj_tensor(),
                                    self.hidden_b.raw_ptr(), self.logits.raw_ptr())?;
        Ok(())
    }

    /// Capture the on-device portion of `forward_token(token, state)`
    /// into a HIP graph and return an executable handle.
    ///
    /// **The captured graph encodes the specific `token` and the
    /// state's position at capture time** — re-launching it advances
    /// the recorded slot, not whatever position `state` currently has.
    /// In particular, scalar kernel args (token id, RoPE pos, attn
    /// total_len) and KV cache write offsets are baked in.
    ///
    /// For benchmarking single-step decode latency this is fine;
    /// production multi-token decode needs parametric capture
    /// (`hipGraphExecKernelNodeSetParams` + memcpy node updates),
    /// which lives in a follow-up.
    pub fn capture_forward_graph(&self, token: u32, state: &mut Qwen35GpuState)
        -> Result<GraphExec, String>
    {
        Graph::begin_capture(&self.stream, HipStreamCaptureMode::Global)?;
        if let Err(e) = self.enqueue_forward_token(token, state) {
            // Make a best-effort attempt to leave the stream in a sane state.
            let _ = Graph::end_capture(&self.stream);
            return Err(e);
        }
        let graph = Graph::end_capture(&self.stream)?;
        let exec = graph.instantiate()?;
        // graph (the topology) is free to drop once instantiated; the
        // GraphExec keeps the executable copy.
        drop(graph);
        Ok(exec)
    }

    /// Launch a previously-captured forward graph and return logits.
    /// `state.pos` is bumped by 1 — but mutating internal positions
    /// inside the graph is the caller's contract (see capture warning).
    pub fn forward_token_via_graph(&self, exec: &GraphExec, state: &mut Qwen35GpuState)
        -> Result<Vec<f32>, String>
    {
        exec.launch(&self.stream)?;
        self.stream.synchronize()?;
        state.pos += 1;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// Run `forward_token` over each input token in order; return the
    /// logits at the last position. Mirrors
    /// `cpu::qwen3_5::Qwen35F32Model::forward_tokens`.
    pub fn forward_tokens(&self, tokens: &[u32], state: &mut Qwen35GpuState)
        -> Result<Vec<f32>, String>
    {
        assert!(!tokens.is_empty(), "forward_tokens needs at least one token");
        let mut last = Vec::new();
        for &t in tokens {
            last = self.forward_token(t, state)?;
        }
        Ok(last)
    }

    // ===== Batched prefill =================================================

    fn launch_cvt(&self, kname: &str, src: *mut c_void, dst: *mut c_void, n: u32)
        -> Result<(), String>
    {
        let f = self.cvt_module.function(kname)?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut s = src; let mut d = dst; let mut na = n;
        let mut args: [*mut c_void; 3] = [
            &mut s as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    /// Bulk-dequant a quantized weight tensor to a fresh fp16 buffer.
    fn dequant_weight(&self, w: &GpuMatvecTensor) -> Result<DeviceBuf<u16>, String> {
        let n = (w.in_dim as usize) * (w.out_dim as usize);
        let out: DeviceBuf<u16> = DeviceBuf::new(n)?;
        let (module, kname, wpb, threads): (&Module, &str, usize, u32) = match w.dtype {
            GgmlType::Q4_K   => (&self.dequant_q4_k_module,   "dequant_q4_k_f16",   256, 256),
            GgmlType::Q5_K   => (&self.dequant_q5_k_module,   "dequant_q5_k_f16",   256, 256),
            GgmlType::Q6_K   => (&self.dequant_q6_k_module,   "dequant_q6_k_f16",   256, 256),
            GgmlType::Q8_0   => (&self.dequant_q8_0_module,   "dequant_q8_0_f16",    32,  32),
            GgmlType::IQ4_XS => (&self.dequant_iq4_xs_module, "dequant_iq4_xs_f16", 256, 256),
            other => return Err(format!("dequant_weight: unsupported {other:?}")),
        };
        let n_blocks = (n / wpb) as u32;
        let f = module.function(kname)?;
        let mut w_ptr = w.data.raw_ptr();
        let mut o_ptr = out.raw_ptr();
        let mut nb = n_blocks;
        let mut args: [*mut c_void; 3] = [
            &mut w_ptr as *mut _ as *mut c_void,
            &mut o_ptr as *mut _ as *mut c_void,
            &mut nb    as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((n_blocks, 1, 1), (threads, 1, 1), 0, Some(&self.stream), &mut args)?; }
        Ok(out)
    }

    /// Batched matmul: `Y[N, out] = X[N, in] · Wᵀ`. Dequant W→fp16,
    /// X→fp16, fp32-accumulate GEMM, Y→fp32. All on `self.stream`.
    fn bmm(&self, w: &GpuMatvecTensor, x_f32: *mut c_void, n_rows: usize,
           y_f32: *mut c_void) -> Result<(), String>
    {
        let in_d = w.in_dim as usize;
        let out_d = w.out_dim as usize;

        // W → fp16 (F16 weights are already fp16: use raw bytes directly).
        let dq: Option<DeviceBuf<u16>>;
        let w_ptr: *mut c_void;
        if w.dtype == GgmlType::F16 {
            w_ptr = w.data.raw_ptr();
            dq = None;
        } else {
            let b = self.dequant_weight(w)?;
            w_ptr = b.raw_ptr();
            dq = Some(b);
        }
        // X → fp16.
        let x_f16: DeviceBuf<u16> = DeviceBuf::new(n_rows * in_d)?;
        self.launch_cvt("cvt_f32_to_f16", x_f32, x_f16.raw_ptr(), (n_rows * in_d) as u32)?;
        // GEMM. rocBLAS handle shares self.stream, so it serialises after
        // the dequant + cvt launches above — no explicit sync needed.
        let y_f16: DeviceBuf<u16> = DeviceBuf::new(n_rows * out_d)?;
        unsafe {
            self.rocblas.gemm_f16_f32acc(
                RocblasOp::Transpose, RocblasOp::None,
                out_d as i32, n_rows as i32, in_d as i32,
                1.0,
                w_ptr as *const c_void, in_d as i32,
                x_f16.as_ptr() as *const c_void, in_d as i32,
                0.0,
                y_f16.as_ptr() as *mut c_void, out_d as i32,
            )?;
        }
        self.launch_cvt("cvt_f16_to_f32", y_f16.raw_ptr(), y_f32, (n_rows * out_d) as u32)?;
        drop(dq);  // keep the dequant buffer alive across the GEMM
        Ok(())
    }

    fn launch_rope_batched(&self, x: *mut c_void, n_heads: u32, n_rows: u32, base_pos: u32)
        -> Result<(), String>
    {
        let f = self.rope_batched_module.function("rope_apply_batched_f32")?;
        let half = (self.rotary_dim / 2) as u32;
        let block: u32 = 64;
        let grid_x = (half + block - 1) / block;
        let mut xa = x;
        let mut ca = self.rope_cos.raw_ptr();
        let mut sa = self.rope_sin.raw_ptr();
        let mut hd = self.head_dim as u32;
        let mut rd = self.rotary_dim as u32;
        let mut nh = n_heads;
        let mut bp = base_pos;
        let mut args: [*mut c_void; 7] = [
            &mut xa as *mut _ as *mut c_void,
            &mut ca as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut rd as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid_x, n_heads, n_rows), (block, 1, 1), 0,
                          Some(&self.stream), &mut args) }
    }

    fn launch_attn_step_batched(&self, q: *mut c_void, k_cache: *mut c_void,
                                v_cache: *mut c_void, out: *mut c_void,
                                base_pos: u32, n_rows: u32, scaling: f32)
        -> Result<(), String>
    {
        let f = self.attn_step_batched_module.function("attn_step_batched_f32")?;
        let block: u32 = 256;
        let head_dim = self.head_dim as u32;
        let max_total = base_pos + n_rows;
        let smem = ((head_dim + max_total) + block) * std::mem::size_of::<f32>() as u32;
        let mut qa = q; let mut ka = k_cache; let mut va = v_cache; let mut oa = out;
        let mut nh = self.n_heads as u32;
        let mut nkv = self.n_kv_heads as u32;
        let mut hd = head_dim;
        let mut bp = base_pos;
        let mut nr = n_rows;
        let mut sc = scaling;
        let mut args: [*mut c_void; 10] = [
            &mut qa as *mut _ as *mut c_void,
            &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((self.n_heads as u32, n_rows, 1), (block, 1, 1),
                          smem, Some(&self.stream), &mut args) }
    }

    /// Batched prefill: process all `tokens` in one pass, advancing each
    /// block's state, and return the logits at the LAST position.
    ///
    /// Mirrors `forward_tokens` but batches every matmul into a single
    /// rocBLAS GEMM (weight read once, reused across N rows). The GDN
    /// recurrent + conv steps stay sequential per position — that's an
    /// inherent data dependency — but their projections are batched.
    pub fn forward_tokens_batched(&self, tokens: &[u32], state: &mut Qwen35GpuState)
        -> Result<Vec<f32>, String>
    {
        assert!(!tokens.is_empty(), "forward_tokens_batched needs ≥1 token");
        let n = tokens.len();
        let h     = self.hidden;
        let q_dim = self.q_dim();
        let kv_dim = self.kv_dim();
        let vdim  = self.gdn_value_dim;
        let cdim  = self.gdn_conv_dim;
        let scaling = (self.head_dim as f32).powf(-0.5);

        // Per-call batched activation buffers.
        let ba: DeviceBuf<f32> = DeviceBuf::new(n * h)?;        // running hidden
        let bb: DeviceBuf<f32> = DeviceBuf::new(n * h)?;        // scratch
        let bnorm: DeviceBuf<f32> = DeviceBuf::new(n * h)?;     // normed scratch

        // 1) Embed all tokens into ba (one row each).
        for (r, &tok) in tokens.iter().enumerate() {
            let row_ptr = unsafe { (ba.raw_ptr() as *mut f32).add(r * h) } as *mut c_void;
            self.launch_embed_lookup_dispatch(&self.token_embd, row_ptr, tok)?;
        }

        // 2) Every block.
        for (block, st) in self.blocks.iter().zip(state.block_states.iter_mut()) {
            match (block, st) {
                (GpuBlock::Full(w), GpuBlockState::Full(kv)) => {
                    self.batched_full_block(&ba, &bb, &bnorm, w, kv, n, scaling)?;
                }
                (GpuBlock::Linear(w), GpuBlockState::Linear(s)) => {
                    self.batched_linear_block(&ba, &bb, &bnorm, w, s, n)?;
                }
                _ => return Err("block kind mismatch".into()),
            }
        }
        let _ = (q_dim, kv_dim, vdim, cdim);

        // 3) Output norm + projection on the LAST row only.
        let last_in = unsafe { (ba.raw_ptr() as *mut f32).add((n - 1) * h) } as *mut c_void;
        self.launch_rmsnorm(last_in, self.output_norm.raw_ptr(),
                            self.hidden_b.raw_ptr(), h as u32, self.rms_eps)?;
        self.launch_matvec_dispatch(self.output_proj_tensor(),
                                    self.hidden_b.raw_ptr(), self.logits.raw_ptr())?;
        self.stream.synchronize()?;
        state.pos += n;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// One full-attention block over a batch of `n` rows. `ba` is the
    /// running hidden (mutated in place); `bb` / `bnorm` are scratch.
    fn batched_full_block(&self, ba: &DeviceBuf<f32>, bb: &DeviceBuf<f32>,
                          bnorm: &DeviceBuf<f32>, w: &GpuFullAttnBlock,
                          kv: &mut GpuKvCache, n: usize, scaling: f32)
        -> Result<(), String>
    {
        let h = self.hidden;
        let q_dim = self.q_dim();
        let kv_dim = self.kv_dim();
        let base_pos = kv.len;
        assert!(base_pos + n <= kv.max_seq, "KV cache overflow in batched prefill");

        // pre-norm → bnorm  (n independent rmsnorms via the multihead kernel)
        self.launch_rmsnorm_multihead(ba.raw_ptr(), w.attn.attn_norm.raw_ptr(),
                                      bnorm.raw_ptr(), n as u32, h as u32, self.rms_eps)?;

        // QKV projections, batched.
        let q_raw: DeviceBuf<f32> = DeviceBuf::new(n * 2 * q_dim)?;
        let k_raw: DeviceBuf<f32> = DeviceBuf::new(n * kv_dim)?;
        let v_raw: DeviceBuf<f32> = DeviceBuf::new(n * kv_dim)?;
        self.bmm(&w.attn.attn_q, bnorm.raw_ptr(), n, q_raw.raw_ptr())?;
        self.bmm(&w.attn.attn_k, bnorm.raw_ptr(), n, k_raw.raw_ptr())?;
        self.bmm(&w.attn.attn_v, bnorm.raw_ptr(), n, v_raw.raw_ptr())?;

        // split q_raw → q, gate. The split kernel walks n_heads*head_dim
        // elements; passing n*n_heads covers all rows.
        let q_buf:   DeviceBuf<f32> = DeviceBuf::new(n * q_dim)?;
        let gate:    DeviceBuf<f32> = DeviceBuf::new(n * q_dim)?;
        self.launch_split_q_gate(q_raw.raw_ptr(), q_buf.raw_ptr(), gate.raw_ptr(),
                                 (n * self.n_heads) as u32, self.head_dim as u32)?;
        // per-head Q-norm (n*n_heads independent heads).
        self.launch_rmsnorm_multihead(q_buf.raw_ptr(), w.attn.attn_q_norm.raw_ptr(),
                                      q_buf.raw_ptr(),
                                      (n * self.n_heads) as u32, self.head_dim as u32,
                                      self.rms_eps)?;
        self.launch_rope_batched(q_buf.raw_ptr(), self.n_heads as u32, n as u32, base_pos as u32)?;
        // per-kv-head K-norm.
        let k_norm: DeviceBuf<f32> = DeviceBuf::new(n * kv_dim)?;
        self.launch_rmsnorm_multihead(k_raw.raw_ptr(), w.attn.attn_k_norm.raw_ptr(),
                                      k_norm.raw_ptr(),
                                      (n * self.n_kv_heads) as u32, self.head_dim as u32,
                                      self.rms_eps)?;
        self.launch_rope_batched(k_norm.raw_ptr(), self.n_kv_heads as u32, n as u32, base_pos as u32)?;

        // Push all N (k, v) into the cache at slots [base_pos, base_pos+n).
        kv.k.copy_from_device_at_async(&k_norm, base_pos * kv_dim, &self.stream)?;
        kv.v.copy_from_device_at_async(&v_raw,  base_pos * kv_dim, &self.stream)?;

        // Batched causal attention → attn_concat.
        let attn: DeviceBuf<f32> = DeviceBuf::new(n * q_dim)?;
        self.launch_attn_step_batched(q_buf.raw_ptr(), kv.k.raw_ptr(), kv.v.raw_ptr(),
                                      attn.raw_ptr(), base_pos as u32, n as u32, scaling)?;
        // output gate + projection.
        self.launch_sigmoid_mul(attn.raw_ptr(), gate.raw_ptr(), (n * q_dim) as u32)?;
        self.bmm(&w.attn.attn_output, attn.raw_ptr(), n, bb.raw_ptr())?;
        self.launch_add_inplace(ba.raw_ptr(), bb.raw_ptr(), (n * h) as u32)?;

        // FFN sub-layer.
        self.launch_rmsnorm_multihead(ba.raw_ptr(), w.post_norm.raw_ptr(),
                                      bnorm.raw_ptr(), n as u32, h as u32, self.rms_eps)?;
        self.batched_ffn(bnorm, bb, &w.ffn, n)?;
        self.launch_add_inplace(ba.raw_ptr(), bb.raw_ptr(), (n * h) as u32)?;

        kv.len += n;
        Ok(())
    }

    /// Batched SwiGLU FFN: `out_bb = down(silu(gate(in)) * up(in))`.
    fn batched_ffn(&self, input: &DeviceBuf<f32>, out_bb: &DeviceBuf<f32>,
                   ffn: &GpuFfnWeights, n: usize) -> Result<(), String>
    {
        let f = self.ffn;
        let gate: DeviceBuf<f32> = DeviceBuf::new(n * f)?;
        let up:   DeviceBuf<f32> = DeviceBuf::new(n * f)?;
        self.bmm(&ffn.gate, input.raw_ptr(), n, gate.raw_ptr())?;
        self.bmm(&ffn.up,   input.raw_ptr(), n, up.raw_ptr())?;
        self.launch_swiglu(gate.raw_ptr(), up.raw_ptr(), gate.raw_ptr(), (n * f) as u32)?;
        self.bmm(&ffn.down, gate.raw_ptr(), n, out_bb.raw_ptr())?;
        Ok(())
    }

    /// One GDN block over a batch of `n` rows: projections batched, the
    /// conv1d + recurrent state updates looped sequentially per row
    /// (inherent recurrence — position r depends on r-1).
    fn batched_linear_block(&self, ba: &DeviceBuf<f32>, bb: &DeviceBuf<f32>,
                            bnorm: &DeviceBuf<f32>, w: &GpuLinAttnBlock,
                            st: &mut GpuLinAttnState, n: usize)
        -> Result<(), String>
    {
        let h = self.hidden;
        let vdim = self.gdn_value_dim;
        let cdim = self.gdn_conv_dim;
        let nh   = self.gdn_n_heads as u32;
        let hd   = self.gdn_head_dim as u32;
        let q_scale = (self.gdn_head_dim as f32).powf(-0.5);

        // pre-norm.
        self.launch_rmsnorm_multihead(ba.raw_ptr(), w.attn.attn_norm.raw_ptr(),
                                      bnorm.raw_ptr(), n as u32, h as u32, self.rms_eps)?;

        // Four projections, batched.
        let qkv: DeviceBuf<f32> = DeviceBuf::new(n * cdim)?;
        let z:   DeviceBuf<f32> = DeviceBuf::new(n * vdim)?;
        let a:   DeviceBuf<f32> = DeviceBuf::new(n * self.gdn_n_heads)?;
        let b:   DeviceBuf<f32> = DeviceBuf::new(n * self.gdn_n_heads)?;
        self.bmm(&w.attn.attn_qkv,  bnorm.raw_ptr(), n, qkv.raw_ptr())?;
        self.bmm(&w.attn.attn_gate, bnorm.raw_ptr(), n, z.raw_ptr())?;
        self.bmm(&w.attn.ssm_alpha, bnorm.raw_ptr(), n, a.raw_ptr())?;
        self.bmm(&w.attn.ssm_beta,  bnorm.raw_ptr(), n, b.raw_ptr())?;

        // conv1d + SiLU, sequential per row (conv history threads through).
        let conv_out: DeviceBuf<f32> = DeviceBuf::new(n * cdim)?;
        for r in 0..n {
            let in_ptr  = unsafe { (qkv.raw_ptr()      as *mut f32).add(r * cdim) } as *mut c_void;
            let out_ptr = unsafe { (conv_out.raw_ptr() as *mut f32).add(r * cdim) } as *mut c_void;
            self.launch_conv1d_step_silu(in_ptr, w.attn.ssm_conv1d.raw_ptr(),
                                         st.conv_hist.raw_ptr(), out_ptr,
                                         cdim as u32, self.gdn_conv_kernel as u32)?;
        }

        // Per-row: L2-norm Q/K, recurrent step, gated rmsnorm.
        let core: DeviceBuf<f32> = DeviceBuf::new(n * vdim)?;
        let q_buf: DeviceBuf<f32> = DeviceBuf::new(vdim)?;
        let k_buf: DeviceBuf<f32> = DeviceBuf::new(vdim)?;
        for r in 0..n {
            let conv_ptr = conv_out.raw_ptr() as *mut f32;
            let q_in = unsafe { conv_ptr.add(r * cdim)            } as *mut c_void;
            let k_in = unsafe { conv_ptr.add(r * cdim + vdim)     } as *mut c_void;
            let v_in = unsafe { conv_ptr.add(r * cdim + 2 * vdim) } as *mut c_void;
            self.launch_l2norm_qk(q_in, q_buf.raw_ptr(), k_in, k_buf.raw_ptr(),
                                  nh, hd, 1e-6, q_scale)?;
            let a_row = unsafe { (a.raw_ptr() as *mut f32).add(r * self.gdn_n_heads) } as *mut c_void;
            let b_row = unsafe { (b.raw_ptr() as *mut f32).add(r * self.gdn_n_heads) } as *mut c_void;
            let core_row = unsafe { (core.raw_ptr() as *mut f32).add(r * vdim) } as *mut c_void;
            self.launch_gdn_recurrent_step_fused(q_buf.raw_ptr(), k_buf.raw_ptr(), v_in,
                                                 a_row, b_row,
                                                 w.attn.ssm_a.raw_ptr(),
                                                 w.attn.ssm_dt_bias.raw_ptr(),
                                                 st.recurrent.raw_ptr(), core_row,
                                                 nh, hd)?;
            let z_row = unsafe { (z.raw_ptr() as *mut f32).add(r * vdim) } as *mut c_void;
            self.launch_rmsnorm_gated_multihead(core_row, z_row, w.attn.ssm_norm.raw_ptr(),
                                                core_row, nh, hd, self.rms_eps)?;
        }

        // ssm_out projection, batched.
        self.bmm(&w.attn.ssm_out, core.raw_ptr(), n, bb.raw_ptr())?;
        self.launch_add_inplace(ba.raw_ptr(), bb.raw_ptr(), (n * h) as u32)?;

        // FFN sub-layer.
        self.launch_rmsnorm_multihead(ba.raw_ptr(), w.post_norm.raw_ptr(),
                                      bnorm.raw_ptr(), n as u32, h as u32, self.rms_eps)?;
        self.batched_ffn(bnorm, bb, &w.ffn, n)?;
        self.launch_add_inplace(ba.raw_ptr(), bb.raw_ptr(), (n * h) as u32)?;
        Ok(())
    }

    /// One full transformer block (full-attention variant): pre-norm +
    /// attention + residual + pre-norm + FFN + residual. Mirrors
    /// `cpu::qwen3_5::full_attention_block`.
    ///
    /// Internal buffers used as scratch:
    ///   hidden_a — running hidden state (in/out)
    ///   hidden_b — first attn output, then post-norm output, then ffn output
    pub fn apply_full_attention_block(&self,
        input: &[f32],
        weights: &GpuFullAttnBlock,
        kv_cache: &mut GpuKvCache,
    ) -> Result<Vec<f32>, String>
    {
        assert_eq!(input.len(), self.hidden);
        let h_dim = self.hidden as u32;

        // H2D the input.
        self.hidden_a.copy_from_host(input)?;

        // Sub-layer 1: attention with pre-norm + residual.
        self.step_full_attention(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(),
                                 &weights.attn, kv_cache)?;
        self.launch_add_inplace(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), h_dim)?;

        // Sub-layer 2: FFN with pre-norm + residual.
        // post-norm rewrites hidden_b (now serving as `normed`).
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), weights.post_norm.raw_ptr(),
                            self.hidden_b.raw_ptr(), h_dim, self.rms_eps)?;
        // FFN reads hidden_b, writes hidden_b (alias OK — gate/up read
        // happens before down writes within the stream).
        self.step_swiglu_ffn(self.hidden_b.raw_ptr(), self.hidden_b.raw_ptr(),
                             &weights.ffn)?;
        self.launch_add_inplace(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), h_dim)?;

        self.stream.synchronize()?;
        let mut out = vec![0.0f32; self.hidden];
        self.hidden_a.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// Device-pointer linear-attention (GDN) step. Mirrors
    /// `cpu::qwen3_5::linear_attention_step`. Reads `input_ptr` (preserved),
    /// writes the post-projection output to `output_ptr`. Updates the
    /// recurrent + conv state in `state`.
    fn step_linear_attention(&self,
        input_ptr: *mut c_void, output_ptr: *mut c_void,
        weights: &GpuLinAttnWeights, state: &mut GpuLinAttnState,
    ) -> Result<(), String>
    {
        let h_dim     = self.hidden        as u32;
        let conv_dim  = self.gdn_conv_dim  as u32;
        let n_heads   = self.gdn_n_heads   as u32;
        let head_dim  = self.gdn_head_dim  as u32;
        let q_scale   = (self.gdn_head_dim as f32).powf(-0.5);

        // 1) normed = rmsnorm(input, attn_norm) → output_ptr (used as scratch)
        self.launch_rmsnorm(input_ptr, weights.attn_norm.raw_ptr(),
                            output_ptr, h_dim, self.rms_eps)?;

        // 2) Four projections off normed.
        self.launch_matvec_dispatch(&weights.attn_qkv,  output_ptr, self.gdn_qkv.raw_ptr())?;
        self.launch_matvec_dispatch(&weights.attn_gate, output_ptr, self.gdn_z.raw_ptr())?;
        self.launch_matvec_dispatch(&weights.ssm_alpha, output_ptr, self.gdn_a.raw_ptr())?;
        self.launch_matvec_dispatch(&weights.ssm_beta,  output_ptr, self.gdn_b.raw_ptr())?;

        // 3) Causal Conv1D with SiLU fused into the output write.
        self.launch_conv1d_step_silu(self.gdn_qkv.raw_ptr(), weights.ssm_conv1d.raw_ptr(),
                                     state.conv_hist.raw_ptr(), self.gdn_conv_out.raw_ptr(),
                                     conv_dim, self.gdn_conv_kernel as u32)?;

        // 4) conv_out is laid out [Q | K | V], each [n_heads * head_dim] = value_dim.
        //    Slice by pointer arithmetic — the data is contiguous.
        let conv_out_ptr = self.gdn_conv_out.raw_ptr() as *mut f32;
        let q_in_ptr = unsafe { conv_out_ptr.add(0)                      } as *mut c_void;
        let k_in_ptr = unsafe { conv_out_ptr.add(self.gdn_value_dim)     } as *mut c_void;
        let v_in_ptr = unsafe { conv_out_ptr.add(2 * self.gdn_value_dim) } as *mut c_void;

        // 5) Per-head L2-norm of Q (scale 1/√head_dim) and K (scale 1),
        //    fused into one 2D-grid launch.
        self.launch_l2norm_qk(q_in_ptr, self.gdn_q.raw_ptr(),
                              k_in_ptr, self.gdn_k.raw_ptr(),
                              n_heads, head_dim, 1e-6, q_scale)?;

        // 6+7) Recurrent gated delta-rule update — decay/beta computed
        //      inside the kernel from a/b/ssm_a/dt_bias.
        self.launch_gdn_recurrent_step_fused(self.gdn_q.raw_ptr(), self.gdn_k.raw_ptr(), v_in_ptr,
                                             self.gdn_a.raw_ptr(), self.gdn_b.raw_ptr(),
                                             weights.ssm_a.raw_ptr(), weights.ssm_dt_bias.raw_ptr(),
                                             state.recurrent.raw_ptr(),
                                             self.gdn_core_out.raw_ptr(),
                                             n_heads, head_dim)?;

        // 8) Per-head gated RMSNorm: core_out *= w * silu(z), in place.
        self.launch_rmsnorm_gated_multihead(self.gdn_core_out.raw_ptr(), self.gdn_z.raw_ptr(),
                                             weights.ssm_norm.raw_ptr(),
                                             self.gdn_core_out.raw_ptr(),
                                             n_heads, head_dim, self.rms_eps)?;

        // 9) Project back to hidden.
        self.launch_matvec_dispatch(&weights.ssm_out, self.gdn_core_out.raw_ptr(), output_ptr)?;
        Ok(())
    }

    /// Run one decode step of the linear-attention (GDN) sub-layer.
    /// `input` and the returned vector are hidden-sized.
    pub fn apply_linear_attention(&self,
        input: &[f32],
        weights: &GpuLinAttnWeights,
        state: &mut GpuLinAttnState,
    ) -> Result<Vec<f32>, String>
    {
        assert_eq!(input.len(), self.hidden);
        self.hidden_a.copy_from_host(input)?;
        self.step_linear_attention(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(),
                                    weights, state)?;
        self.stream.synchronize()?;
        let mut out = vec![0.0f32; self.hidden];
        self.hidden_b.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// One full transformer block (linear-attention variant): GDN +
    /// residual + post-norm + FFN + residual. Mirrors
    /// `cpu::qwen3_5::linear_attention_block`.
    pub fn apply_linear_attention_block(&self,
        input: &[f32],
        weights: &GpuLinAttnBlock,
        state: &mut GpuLinAttnState,
    ) -> Result<Vec<f32>, String>
    {
        assert_eq!(input.len(), self.hidden);
        let h_dim = self.hidden as u32;

        self.hidden_a.copy_from_host(input)?;
        self.step_linear_attention(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(),
                                    &weights.attn, state)?;
        self.launch_add_inplace(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), h_dim)?;
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), weights.post_norm.raw_ptr(),
                            self.hidden_b.raw_ptr(), h_dim, self.rms_eps)?;
        self.step_swiglu_ffn(self.hidden_b.raw_ptr(), self.hidden_b.raw_ptr(), &weights.ffn)?;
        self.launch_add_inplace(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), h_dim)?;
        self.stream.synchronize()?;
        let mut out = vec![0.0f32; self.hidden];
        self.hidden_a.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// Run one decode step of the full-attention block (matches
    /// `cpu::qwen3_5::full_attention_step`).
    ///
    ///   normed     = rmsnorm(input, attn_norm)
    ///   q_raw,k,v  = matvec(normed, {attn_q (2× width), attn_k, attn_v})
    ///   q, gate    = split per-head q_raw into Q + gate
    ///   q          = rmsnorm_per_head(q,      attn_q_norm); rope(q, pos)
    ///   k          = rmsnorm_per_head(k_raw,  attn_k_norm); rope(k, pos)
    ///   kv_cache.push(k, v) at position `cache_len`
    ///   attn       = attn_step(q, K_cache[0..len+1], V_cache[0..len+1])
    ///   attn      *= sigmoid(gate)
    ///   out        = matvec(attn, attn_output)
    ///
    /// Returns the hidden-sized output as a Vec<f32>. Increments
    /// `kv_cache.len` by 1.
    pub fn apply_full_attention(&self,
        input: &[f32],
        weights: &GpuFullAttnWeights,
        kv_cache: &mut GpuKvCache,
    ) -> Result<Vec<f32>, String>
    {
        assert_eq!(input.len(), self.hidden, "input must be hidden-sized");
        self.hidden_a.copy_from_host(input)?;
        self.step_full_attention(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(),
                                 weights, kv_cache)?;
        self.stream.synchronize()?;
        let mut out = vec![0.0f32; self.hidden];
        self.hidden_b.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// Run a SwiGLU FFN block on `input` with the given block's weights.
    ///
    ///   gate = matvec(input, gate_w, hidden→ffn)
    ///   up   = matvec(input, up_w,   hidden→ffn)
    ///   mid  = silu(gate) * up                  (in-place into ffn_a)
    ///   out  = matvec(mid,  down_w, ffn→hidden)
    ///
    /// Returns the hidden-sized FFN output as a host Vec<f32>. `input`
    /// is copied H2D into the internal hidden_a scratch.
    pub fn apply_swiglu_ffn(&self, input: &[f32], weights: &GpuFfnWeights)
        -> Result<Vec<f32>, String>
    {
        assert_eq!(input.len(), self.hidden, "input must be hidden-sized");
        self.hidden_a.copy_from_host(input)?;
        self.step_swiglu_ffn(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), weights)?;
        self.stream.synchronize()?;
        let mut out = vec![0.0f32; self.hidden];
        self.hidden_b.copy_to_host(&mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::gguf::GgufFile;

    fn fixture_path() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("REINSTINCT_GGUF_FIXTURE") {
            return Some(PathBuf::from(p));
        }
        let home = std::env::var_os("HOME")?;
        let p = PathBuf::from(home).join("models/qwen-3.5-0.8B/Qwen3.5-0.8B-UD-Q4_K_XL.gguf");
        p.exists().then_some(p)
    }

    #[test]
    fn embed_norm_proj_matches_cpu_chain() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP device"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip: no GGUF fixture"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache: {e}"); return }
        };

        let g = GgufFile::open(&path).expect("open gguf");
        let m = Qwen35F32Model::load(&g).expect("load model");
        let cfg = &m.model.config;
        let hidden = cfg.hidden_size as usize;
        let vocab  = cfg.vocab_size as usize;

        let gpu = GpuQwen35::new(&m.model, &g, &cache, 32).expect("new GpuQwen35");

        // Test on a couple of tokens including EOS and a mid-vocab.
        for &token in &[cfg.eos_token_id, 100u32, 50_000u32] {
            // CPU oracle: embed → output_norm → output_proj.
            let off = token as usize * hidden;
            let embed = &m.weights.token_embd[off..off + hidden];
            let mut normed = vec![0.0f32; hidden];
            crate::cpu::ops::rmsnorm(embed, &m.weights.output_norm, cfg.rms_norm_eps, &mut normed);
            let proj_w = if cfg.tied_embeddings {
                m.weights.token_embd.as_slice()
            } else {
                m.weights.output.as_ref().unwrap().as_slice()
            };
            let mut cpu_logits = vec![0.0f32; vocab];
            crate::cpu::ops::matvec(&normed, proj_w, hidden, vocab, &mut cpu_logits);

            let gpu_logits = gpu.embed_norm_proj(token).expect("gpu chain");

            // Reduction order differs in rmsnorm + matvec, so bit-equality
            // isn't expected. Use combined abs+rel tolerance: a value
            // passes if it's within EITHER bound. Small logits get the
            // abs bound (avoiding catastrophic cancellation in the rel
            // metric); large logits get the rel bound.
            const ABS_TOL: f32 = 5.0e-4;
            const REL_TOL: f32 = 1.0e-3;
            let mut max_abs = 0.0f32;
            let mut worst_violation = 0.0f32;
            let mut worst_at = 0usize;
            for i in 0..vocab {
                let d = (gpu_logits[i] - cpu_logits[i]).abs();
                if d > max_abs { max_abs = d; }
                let allowed = ABS_TOL.max(REL_TOL * cpu_logits[i].abs());
                let v = d - allowed;
                if v > worst_violation { worst_violation = v; worst_at = i; }
            }
            eprintln!("token {token}: max_abs={max_abs:.3e}, worst violation {:.3e} at idx {}",
                worst_violation, worst_at);
            assert!(worst_violation <= 0.0,
                "token {token}: idx {worst_at} gpu={} cpu={} diff exceeds abs={ABS_TOL:.1e} or rel={REL_TOL:.1e}",
                gpu_logits[worst_at], cpu_logits[worst_at]);
        }
    }

    #[test]
    fn swiglu_ffn_matches_cpu_for_real_block() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP device"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip: no GGUF fixture"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache: {e}"); return }
        };

        let g = GgufFile::open(&path).unwrap();
        let m = Qwen35F32Model::load(&g).unwrap();
        let cfg = &m.model.config;
        let h = cfg.hidden_size as usize;
        let f = cfg.ffn_size as usize;

        // Pull FFN weights from a couple of real blocks: one linear-attn (e.g. block 0)
        // and one full-attn (e.g. block 3 — pattern is L,L,L,F,L,L,L,F,...).
        // The FFN layout is identical between block kinds, so this also confirms
        // we're reading the same tensors per block.
        use crate::cpu::qwen3_5::BlockWeights;
        let block_indices = [0usize, 3];
        for &block_idx in &block_indices {
            let (gate_w, up_w, down_w) = match &m.weights.blocks[block_idx] {
                BlockWeights::LinearAttention(w) => (&w.ffn_gate, &w.ffn_up, &w.ffn_down),
                BlockWeights::FullAttention(w)   => (&w.ffn_gate, &w.ffn_up, &w.ffn_down),
            };

            // Build a deterministic, realistic-magnitude pre-norm input
            // (RMSNorm output has rms ≈ 1 by construction, so values O(1) are fair).
            let mut s: u64 = 0xDEADBEEF ^ block_idx as u64;
            let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                               ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
            let input: Vec<f32> = (0..h).map(|_| rng() * 2.0).collect();

            // CPU oracle.
            let mut cpu_out = vec![0.0f32; h];
            crate::cpu::qwen3_5::swiglu_ffn(&input, gate_w, up_w, down_w, h, f, &mut cpu_out);

            // GPU.
            let weights = GpuFfnWeights::from_gguf(&g, block_idx as u32).expect("alloc ffn weights");
            let gpu = GpuQwen35::new(&m.model, &g, &cache, 32).expect("new GpuQwen35");
            let gpu_out = gpu.apply_swiglu_ffn(&input, &weights).expect("gpu ffn");

            // Compare with combined abs+rel tolerance.
            const ABS_TOL: f32 = 1.0e-3;
            const REL_TOL: f32 = 5.0e-3;
            let mut max_abs = 0.0f32;
            let mut worst_violation = 0.0f32;
            let mut worst_at = 0usize;
            for i in 0..h {
                let d = (gpu_out[i] - cpu_out[i]).abs();
                if d > max_abs { max_abs = d; }
                let allowed = ABS_TOL.max(REL_TOL * cpu_out[i].abs());
                let v = d - allowed;
                if v > worst_violation { worst_violation = v; worst_at = i; }
            }
            eprintln!("block {block_idx} ffn: max_abs={max_abs:.3e}, worst_violation={:.3e} at {worst_at}",
                worst_violation);
            assert!(worst_violation <= 0.0,
                "block {block_idx} ffn[{worst_at}]: gpu={} cpu={} diff exceeds abs={ABS_TOL:.1e} or rel={REL_TOL:.1e}",
                gpu_out[worst_at], cpu_out[worst_at]);
        }
    }

    #[test]
    fn forward_token_matches_cpu_oracle() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP device"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip: no GGUF fixture"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache: {e}"); return }
        };

        let g = GgufFile::open(&path).unwrap();
        let m = Qwen35F32Model::load(&g).unwrap();
        let cfg = &m.model.config;
        let vocab = cfg.vocab_size as usize;

        let max_seq = 16usize;
        let gpu = GpuQwen35::new(&m.model, &g, &cache, max_seq).expect("new GpuQwen35");

        // Validate against the CPU oracle on a handful of single tokens
        // we already have golden coverage for.
        for &token in &[cfg.eos_token_id, 100u32, 50_000u32] {
            let mut cpu_state = m.new_state(max_seq);
            let cpu_logits = m.forward_token(token, &mut cpu_state);

            let mut gpu_state = Qwen35GpuState::new(&m.model,max_seq).expect("new gpu state");
            let gpu_logits = gpu.forward_token(token, &mut gpu_state).expect("gpu forward");

            assert_eq!(gpu_logits.len(), vocab);

            // Top-K agreement is the most behaviorally-meaningful check —
            // tiny float drift on near-zero logits would blow up a strict
            // elementwise tolerance, but the argmax / top-K should agree.
            const ABS_TOL: f32 = 5.0e-3;
            const REL_TOL: f32 = 5.0e-3;
            let mut max_abs = 0.0f32;
            let mut worst_violation = 0.0f32;
            let mut worst_at = 0usize;
            for i in 0..vocab {
                let d = (gpu_logits[i] - cpu_logits[i]).abs();
                if d > max_abs { max_abs = d; }
                let allowed = ABS_TOL.max(REL_TOL * cpu_logits[i].abs());
                let v = d - allowed;
                if v > worst_violation { worst_violation = v; worst_at = i; }
            }

            // argmax sanity
            let cpu_argmax = (0..vocab).max_by(|&a, &b| cpu_logits[a].total_cmp(&cpu_logits[b])).unwrap();
            let gpu_argmax = (0..vocab).max_by(|&a, &b| gpu_logits[a].total_cmp(&gpu_logits[b])).unwrap();
            eprintln!("token {token}: max_abs={max_abs:.3e}, argmax cpu={cpu_argmax} gpu={gpu_argmax}, worst_violation={:.3e}",
                worst_violation);

            assert_eq!(cpu_argmax, gpu_argmax,
                "token {token}: argmax disagree (cpu={cpu_argmax} gpu={gpu_argmax})");
            assert!(worst_violation <= 0.0,
                "token {token} idx {worst_at}: gpu={} cpu={} exceeds tol",
                gpu_logits[worst_at], cpu_logits[worst_at]);
        }
    }

    #[test]
    fn forward_tokens_batched_matches_sequential() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip: no GGUF"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c, Err(e) => { eprintln!("skip: {e}"); return }
        };
        let g = GgufFile::open(&path).unwrap();
        let m = Qwen35F32Model::load(&g).unwrap();
        let vocab = m.model.config.vocab_size as usize;
        let max_seq = 32usize;
        let gpu = GpuQwen35::new(&m.model, &g, &cache, max_seq).expect("gpu");

        let prompt = [198u32, 100, 248046, 1, 2, 50_000, 7];

        // Sequential fp32 decode path (the reference).
        let mut s_seq = Qwen35GpuState::new(&m.model,max_seq).unwrap();
        let seq = gpu.forward_tokens(&prompt, &mut s_seq).expect("sequential");

        // Batched fp16-GEMM prefill path.
        let mut s_bat = Qwen35GpuState::new(&m.model,max_seq).unwrap();
        let bat = gpu.forward_tokens_batched(&prompt, &mut s_bat).expect("batched");

        assert_eq!(seq.len(), vocab);
        assert_eq!(bat.len(), vocab);

        let seq_argmax = (0..vocab).max_by(|&a, &b| seq[a].total_cmp(&seq[b])).unwrap();
        let bat_argmax = (0..vocab).max_by(|&a, &b| bat[a].total_cmp(&bat[b])).unwrap();

        // Top-5 overlap — fp16 prefill drifts from fp32 decode over 24
        // layers, so we check behavioural agreement, not bit equality.
        let mut seq_idx: Vec<usize> = (0..vocab).collect();
        seq_idx.sort_by(|&a, &b| seq[b].total_cmp(&seq[a]));
        let mut bat_idx: Vec<usize> = (0..vocab).collect();
        bat_idx.sort_by(|&a, &b| bat[b].total_cmp(&bat[a]));
        let seq_top5: std::collections::HashSet<usize> = seq_idx[..5].iter().copied().collect();
        let overlap = bat_idx[..5].iter().filter(|i| seq_top5.contains(i)).count();

        eprintln!("batched prefill: argmax seq={seq_argmax} bat={bat_argmax}, top-5 overlap {overlap}/5");
        assert_eq!(seq_argmax, bat_argmax,
            "batched argmax {bat_argmax} != sequential {seq_argmax}");
        assert!(overlap >= 4, "top-5 overlap {overlap}/5 too low — likely a real bug");
        assert_eq!(s_bat.pos, prompt.len(), "batched state didn't advance correctly");
    }

    #[test]
    fn forward_tokens_matches_repeated_forward_token_gpu() {
        // Multi-token wrapper bit-equivalence (same stream → same logits).
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache: {e}"); return }
        };
        let g = GgufFile::open(&path).unwrap();
        let m = Qwen35F32Model::load(&g).unwrap();
        let max_seq = 16usize;
        let gpu = GpuQwen35::new(&m.model, &g, &cache, max_seq).expect("gpu");

        let prompt = [198u32, 100, 248046, 1, 2];
        let mut s_one = Qwen35GpuState::new(&m.model,max_seq).unwrap();
        let logits_batch = gpu.forward_tokens(&prompt, &mut s_one).unwrap();

        let mut s_step = Qwen35GpuState::new(&m.model,max_seq).unwrap();
        let mut logits_step = Vec::new();
        for &t in &prompt {
            logits_step = gpu.forward_token(t, &mut s_step).unwrap();
        }
        for i in 0..logits_batch.len() {
            assert_eq!(logits_batch[i].to_bits(), logits_step[i].to_bits(),
                "forward_tokens vs forward_token diverge at {i}");
        }
    }

    #[test]
    fn linear_attention_step_matches_cpu_for_real_block() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP device"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip: no GGUF fixture"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache: {e}"); return }
        };

        use crate::cpu::qwen3_5::{BlockWeights, LinAttnState as CpuLinAttnState,
                                  linear_attention_step};

        let g = GgufFile::open(&path).unwrap();
        let m = Qwen35F32Model::load(&g).unwrap();
        let cfg = &m.model.config;
        let h = cfg.hidden_size as usize;

        // Block 0 is LinearAttention in Qwen 3.5.
        let block_idx = m.model.block_kinds.iter()
            .position(|k| matches!(k, crate::model::qwen3_5::BlockKind::LinearAttention))
            .expect("model has at least one LinearAttention block");
        let weights = match &m.weights.blocks[block_idx] {
            BlockWeights::LinearAttention(w) => w,
            _ => unreachable!(),
        };
        eprintln!("validating GDN step on block {block_idx}");

        let conv_dim = 3 * cfg.gdn_value_dim as usize;
        let mut cpu_state = CpuLinAttnState::new(
            cfg.gdn_n_heads as usize,
            cfg.gdn_head_dim as usize,
            cfg.gdn_head_dim as usize,
            conv_dim,
            cfg.gdn_conv_kernel as usize,
        );

        let gpu = GpuQwen35::new(&m.model, &g, &cache, 16).expect("new GpuQwen35");
        let gpu_w = GpuLinAttnWeights::from_gguf(&g, block_idx as u32).expect("upload GDN weights");
        let mut gpu_state = GpuLinAttnState::new(
            cfg.gdn_n_heads     as usize,
            cfg.gdn_head_dim    as usize,
            conv_dim,
            cfg.gdn_conv_kernel as usize,
        ).expect("alloc gpu lin state");

        let mut s: u64 = 0xCAFE_BABE_FACE;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                           ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };

        for step in 0..4 {
            let input: Vec<f32> = (0..h).map(|_| rng() * 2.0).collect();

            let mut cpu_out = vec![0.0f32; h];
            linear_attention_step(&input, weights, cfg, &mut cpu_state, &mut cpu_out);

            let gpu_out = gpu.apply_linear_attention(&input, &gpu_w, &mut gpu_state)
                .expect("gpu GDN");

            const ABS_TOL: f32 = 1.0e-3;
            const REL_TOL: f32 = 5.0e-3;
            let mut max_abs = 0.0f32;
            let mut worst_violation = 0.0f32;
            let mut worst_at = 0usize;
            for i in 0..h {
                let d = (gpu_out[i] - cpu_out[i]).abs();
                if d > max_abs { max_abs = d; }
                let allowed = ABS_TOL.max(REL_TOL * cpu_out[i].abs());
                let v = d - allowed;
                if v > worst_violation { worst_violation = v; worst_at = i; }
            }
            eprintln!("GDN step {step}: max_abs={max_abs:.3e}, worst_violation={:.3e}",
                worst_violation);
            assert!(worst_violation <= 0.0,
                "GDN step {step} idx {worst_at}: gpu={} cpu={} exceeds tol",
                gpu_out[worst_at], cpu_out[worst_at]);
        }
    }

    #[test]
    fn linear_attention_block_matches_cpu_for_real_block() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP device"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip: no GGUF fixture"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache: {e}"); return }
        };

        use crate::cpu::qwen3_5::{BlockWeights, LinAttnState as CpuLinAttnState,
                                  linear_attention_block};

        let g = GgufFile::open(&path).unwrap();
        let m = Qwen35F32Model::load(&g).unwrap();
        let cfg = &m.model.config;
        let h = cfg.hidden_size as usize;

        let block_idx = m.model.block_kinds.iter()
            .position(|k| matches!(k, crate::model::qwen3_5::BlockKind::LinearAttention))
            .expect("model has at least one LinearAttention block");
        let weights = match &m.weights.blocks[block_idx] {
            BlockWeights::LinearAttention(w) => w,
            _ => unreachable!(),
        };

        let conv_dim = 3 * cfg.gdn_value_dim as usize;
        let mut cpu_state = CpuLinAttnState::new(
            cfg.gdn_n_heads as usize,
            cfg.gdn_head_dim as usize,
            cfg.gdn_head_dim as usize,
            conv_dim,
            cfg.gdn_conv_kernel as usize,
        );

        let gpu = GpuQwen35::new(&m.model, &g, &cache, 16).expect("new GpuQwen35");
        let gpu_block = GpuLinAttnBlock::from_gguf(&g, block_idx as u32).expect("upload GDN block");
        let mut gpu_state = GpuLinAttnState::new(
            cfg.gdn_n_heads     as usize,
            cfg.gdn_head_dim    as usize,
            conv_dim,
            cfg.gdn_conv_kernel as usize,
        ).expect("alloc gpu lin state");

        let mut s: u64 = 0xC0FFEE_BABE_BEEF;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                           ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };

        for step in 0..4 {
            let input: Vec<f32> = (0..h).map(|_| rng() * 2.0).collect();

            let mut cpu_state_out = input.clone();
            linear_attention_block(&mut cpu_state_out, weights, cfg, &mut cpu_state);

            let gpu_state_out = gpu.apply_linear_attention_block(&input, &gpu_block, &mut gpu_state)
                .expect("gpu GDN block");

            const ABS_TOL: f32 = 5.0e-3;
            const REL_TOL: f32 = 5.0e-3;
            let mut max_abs = 0.0f32;
            let mut worst_violation = 0.0f32;
            let mut worst_at = 0usize;
            for i in 0..h {
                let d = (gpu_state_out[i] - cpu_state_out[i]).abs();
                if d > max_abs { max_abs = d; }
                let allowed = ABS_TOL.max(REL_TOL * cpu_state_out[i].abs());
                let v = d - allowed;
                if v > worst_violation { worst_violation = v; worst_at = i; }
            }
            eprintln!("GDN block step {step}: max_abs={max_abs:.3e}, worst_violation={:.3e}",
                worst_violation);
            assert!(worst_violation <= 0.0,
                "GDN block step {step} idx {worst_at}: gpu={} cpu={} exceeds tol",
                gpu_state_out[worst_at], cpu_state_out[worst_at]);
        }
    }

    #[test]
    fn full_attention_block_matches_cpu_for_real_block() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP device"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip: no GGUF fixture"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache: {e}"); return }
        };

        use crate::cpu::qwen3_5::{BlockWeights, LayerKvCache, full_attention_block};
        use crate::cpu::rope::RopeCache;

        let g = GgufFile::open(&path).unwrap();
        let m = Qwen35F32Model::load(&g).unwrap();
        let cfg = &m.model.config;
        let h = cfg.hidden_size as usize;

        let block_idx = m.model.block_kinds.iter()
            .position(|k| matches!(k, crate::model::qwen3_5::BlockKind::FullAttention))
            .expect("model has at least one FullAttention block");
        let weights = match &m.weights.blocks[block_idx] {
            BlockWeights::FullAttention(w) => w,
            _ => unreachable!(),
        };
        eprintln!("validating full block {block_idx} (FullAttention + FFN)");

        let max_seq = 16usize;
        let mut layer_kv = LayerKvCache::new(
            max_seq,
            cfg.attn_n_kv_heads as usize,
            cfg.attn_head_dim   as usize,
        );
        let rope = RopeCache::new(cfg.rope_dim_count as usize, max_seq, cfg.rope_freq_base);

        let gpu = GpuQwen35::new(&m.model, &g, &cache, max_seq).expect("new GpuQwen35");
        let gpu_block = GpuFullAttnBlock::from_gguf(&g, block_idx as u32).expect("upload block");
        let mut gpu_kv = GpuKvCache::new(
            max_seq,
            cfg.attn_n_kv_heads as usize,
            cfg.attn_head_dim   as usize,
        ).expect("alloc gpu kv");

        let mut s: u64 = 0xB10C_C0DE;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                           ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };

        let n_steps = 4usize;
        for step in 0..n_steps {
            let input: Vec<f32> = (0..h).map(|_| rng() * 2.0).collect();

            // CPU oracle: in-place block.
            let mut cpu_state = input.clone();
            full_attention_block(&mut cpu_state, weights, cfg, &mut layer_kv, &rope, step);

            let gpu_state = gpu.apply_full_attention_block(&input, &gpu_block, &mut gpu_kv)
                .expect("gpu block");

            const ABS_TOL: f32 = 5.0e-3;
            const REL_TOL: f32 = 5.0e-3;
            let mut max_abs = 0.0f32;
            let mut worst_violation = 0.0f32;
            let mut worst_at = 0usize;
            for i in 0..h {
                let d = (gpu_state[i] - cpu_state[i]).abs();
                if d > max_abs { max_abs = d; }
                let allowed = ABS_TOL.max(REL_TOL * cpu_state[i].abs());
                let v = d - allowed;
                if v > worst_violation { worst_violation = v; worst_at = i; }
            }
            eprintln!("block step {step}: max_abs={max_abs:.3e}, worst_violation={:.3e}",
                worst_violation);
            assert!(worst_violation <= 0.0,
                "block step {step} idx {worst_at}: gpu={} cpu={} exceeds tol",
                gpu_state[worst_at], cpu_state[worst_at]);
        }
    }

    #[test]
    fn full_attention_step_matches_cpu_for_real_block() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP device"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip: no GGUF fixture"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache: {e}"); return }
        };

        use crate::cpu::qwen3_5::{BlockWeights, LayerKvCache, full_attention_step};
        use crate::cpu::rope::RopeCache;

        let g = GgufFile::open(&path).unwrap();
        let m = Qwen35F32Model::load(&g).unwrap();
        let cfg = &m.model.config;
        let h = cfg.hidden_size as usize;

        // Find the first FullAttention block (Qwen 3.5 pattern: L,L,L,F,...).
        let block_idx = m.model.block_kinds.iter()
            .position(|k| matches!(k, crate::model::qwen3_5::BlockKind::FullAttention))
            .expect("model has at least one FullAttention block");
        let weights = match &m.weights.blocks[block_idx] {
            BlockWeights::FullAttention(w) => w,
            _ => unreachable!(),
        };
        eprintln!("validating block {block_idx} (FullAttention)");

        let max_seq = 16usize;

        // CPU side.
        let mut layer_kv = LayerKvCache::new(
            max_seq,
            cfg.attn_n_kv_heads as usize,
            cfg.attn_head_dim   as usize,
        );
        let rope = RopeCache::new(cfg.rope_dim_count as usize, max_seq, cfg.rope_freq_base);

        // GPU side.
        let gpu = GpuQwen35::new(&m.model, &g, &cache, max_seq).expect("new GpuQwen35");
        let gpu_w = GpuFullAttnWeights::from_gguf(&g, block_idx as u32).expect("upload attn weights");
        let mut gpu_kv = GpuKvCache::new(
            max_seq,
            cfg.attn_n_kv_heads as usize,
            cfg.attn_head_dim   as usize,
        ).expect("alloc gpu kv");

        // Drive both sides with the same sequence of inputs to verify KV
        // accumulates correctly. Realistic magnitudes: rmsnorm output is
        // O(1), so we feed inputs that look like post-residual hidden states.
        let mut s: u64 = 0xA77E_FACE_CAFE;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                           ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };

        let n_steps = 4usize;
        for step in 0..n_steps {
            let input: Vec<f32> = (0..h).map(|_| rng() * 2.0).collect();

            // CPU oracle.
            let mut cpu_out = vec![0.0f32; h];
            full_attention_step(&input, weights, cfg, &mut layer_kv, &rope, step, &mut cpu_out);

            // GPU.
            let gpu_out = gpu.apply_full_attention(&input, &gpu_w, &mut gpu_kv)
                .expect("gpu apply_full_attention");

            // Compare.
            const ABS_TOL: f32 = 1.0e-3;
            const REL_TOL: f32 = 5.0e-3;
            let mut max_abs = 0.0f32;
            let mut worst_violation = 0.0f32;
            let mut worst_at = 0usize;
            for i in 0..h {
                let d = (gpu_out[i] - cpu_out[i]).abs();
                if d > max_abs { max_abs = d; }
                let allowed = ABS_TOL.max(REL_TOL * cpu_out[i].abs());
                let v = d - allowed;
                if v > worst_violation { worst_violation = v; worst_at = i; }
            }
            eprintln!("step {step} (cache_len before push = {step}): max_abs={max_abs:.3e}, worst_violation={:.3e}",
                worst_violation);
            assert!(worst_violation <= 0.0,
                "step {step} idx {worst_at}: gpu={} cpu={} exceeds tol",
                gpu_out[worst_at], cpu_out[worst_at]);
            assert_eq!(gpu_kv.len, layer_kv.len(),
                "step {step}: GPU kv len {} doesn't match CPU {}", gpu_kv.len, layer_kv.len());
        }
    }

    #[test]
    fn embed_norm_proj_is_deterministic() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP device"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip: no GGUF fixture"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache: {e}"); return }
        };

        let g = GgufFile::open(&path).unwrap();
        let m = Qwen35F32Model::load(&g).unwrap();
        let gpu = GpuQwen35::new(&m.model, &g, &cache, 32).unwrap();
        let token = m.model.config.eos_token_id;

        let a = gpu.embed_norm_proj(token).unwrap();
        let b = gpu.embed_norm_proj(token).unwrap();
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(a[i].to_bits(), b[i].to_bits(),
                "non-deterministic at index {i}: {} vs {}", a[i], b[i]);
        }
    }
}
