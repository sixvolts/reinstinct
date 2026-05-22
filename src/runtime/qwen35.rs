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
const SWIGLU_SOURCE:            &str = include_str!("../../kernels/swiglu.cpp");
const RMSNORM_MULTIHEAD_SOURCE: &str = include_str!("../../kernels/rmsnorm_multihead.cpp");
const SPLIT_Q_GATE_SOURCE:      &str = include_str!("../../kernels/split_q_gate.cpp");
const SIGMOID_MUL_SOURCE:       &str = include_str!("../../kernels/sigmoid_mul.cpp");
const ROPE_SOURCE:              &str = include_str!("../../kernels/rope.cpp");
const KV_WRITE_F32_SOURCE:      &str = include_str!("../../kernels/kv_write_f32.cpp");
const ATTN_STEP_SOURCE:         &str = include_str!("../../kernels/attn_step.cpp");
const ATTN_PARTIAL_F32_SOURCE:  &str = include_str!("../../kernels/attn_partial_f32.cpp");
const ATTN_MERGE_SOURCE:        &str = include_str!("../../kernels/attn_merge.cpp");
/// Max split-K splits — bounds the partial-attention scratch.
const ATTN_MAX_SPLITS: u32 = 16;
const ADD_INPLACE_SOURCE:       &str = include_str!("../../kernels/add_inplace.cpp");
const GDN_RECURRENT_STEP_FUSED_SOURCE: &str = include_str!("../../kernels/gdn_recurrent_step_fused.cpp");
const CONV1D_STEP_SILU_SOURCE:      &str = include_str!("../../kernels/conv1d_step_silu.cpp");
const L2NORM_QK_SOURCE:             &str = include_str!("../../kernels/l2norm_qk.cpp");
const RMSNORM_GATED_MULTIHEAD_SOURCE: &str = include_str!("../../kernels/rmsnorm_gated_multihead.cpp");
// Batched variants — one launch covers all `n_rows` of a prefill,
// internally iterating with the recurrent state threaded through.
// Decode (n_rows=1) keeps using the single-row kernels above.
const CONV1D_STEP_SILU_BATCHED_SOURCE: &str =
    include_str!("../../kernels/conv1d_step_silu_batched.cpp");
const L2NORM_QK_BATCHED_SOURCE: &str =
    include_str!("../../kernels/l2norm_qk_batched.cpp");
const GDN_RECURRENT_STEP_FUSED_BATCHED_SOURCE: &str =
    include_str!("../../kernels/gdn_recurrent_step_fused_batched.cpp");
const RMSNORM_GATED_MULTIHEAD_BATCHED_SOURCE: &str =
    include_str!("../../kernels/rmsnorm_gated_multihead_batched.cpp");

const MATVEC_F16_SOURCE:    &str = include_str!("../../kernels/matvec_f16.cpp");
const MATVEC_F32_B256_SOURCE: &str = include_str!("../../kernels/matvec_f32_b256.cpp");
const EMBED_LOOKUP_Q6_K_SOURCE: &str = include_str!("../../kernels/embed_lookup_q6_k.cpp");
const EMBED_LOOKUP_Q4_K_SOURCE: &str = include_str!("../../kernels/embed_lookup_q4_k.cpp");
const EMBED_LOOKUP_Q8_0_SOURCE: &str = include_str!("../../kernels/embed_lookup_q8_0_v.cpp");

const CVT_F32_F16_SOURCE:       &str = include_str!("../../kernels/cvt_f32_f16.cpp");
const DEQUANT_Q4_K_F16_SOURCE:  &str = include_str!("../../kernels/dequant_q4_k_f16.cpp");
const DEQUANT_Q5_K_F16_SOURCE:  &str = include_str!("../../kernels/dequant_q5_k_f16.cpp");
const DEQUANT_Q6_K_F16_SOURCE:  &str = include_str!("../../kernels/dequant_q6_k_f16.cpp");
const DEQUANT_Q8_0_F16_SOURCE:  &str = include_str!("../../kernels/dequant_q8_0_f16.cpp");
const DEQUANT_IQ4_XS_F16_SOURCE:&str = include_str!("../../kernels/dequant_iq4_xs_f16.cpp");
const DEQUANT_Q4K_REPACKED_F16_SOURCE: &str =
    include_str!("../../kernels/dequant_q4k_repacked_f16.cpp");
const DEQUANT_Q5K_REPACKED_F16_SOURCE: &str =
    include_str!("../../kernels/dequant_q5k_repacked_f16.cpp");
const DEQUANT_Q6K_REPACKED_F16_SOURCE: &str =
    include_str!("../../kernels/dequant_q6k_repacked_f16.cpp");
const DEQUANT_Q8_0_REPACKED_F16_SOURCE: &str =
    include_str!("../../kernels/dequant_q8_0_repacked.cpp");
const ROPE_BATCHED_SOURCE:      &str = include_str!("../../kernels/rope_batched.cpp");
const ATTN_STEP_BATCHED_SOURCE: &str = include_str!("../../kernels/attn_prefill_flash.cpp");

const QUANTIZE_Q8_SOURCE:      &str = include_str!("../../kernels/quantize_q8.cpp");
const MOE_TOPK_SOURCE:        &str = include_str!("../../kernels/moe_topk.cpp");
const MOE_COMBINE_SOURCE:     &str = include_str!("../../kernels/moe_combine.cpp");
const MOE_MV_Q4K_REPACKED_SOURCE: &str = include_str!("../../kernels/moe_matvec_q4k_repacked.cpp");
const MOE_GATE_UP_SWIGLU_Q4K_SOURCE: &str =
    include_str!("../../kernels/moe_gate_up_swiglu_q4k_repacked.cpp");
const MOE_MV_Q5K_DOWN_SOURCE: &str = include_str!("../../kernels/moe_matvec_q5k_down.cpp");
const MOE_MV_Q6K_DOWN_SOURCE: &str = include_str!("../../kernels/moe_matvec_q6k_down.cpp");
const MOE_MV_Q5K_REPACKED_SOURCE: &str = include_str!("../../kernels/moe_matvec_q5k_repacked.cpp");
const MOE_MV_Q6K_REPACKED_SOURCE: &str = include_str!("../../kernels/moe_matvec_q6k_repacked.cpp");
const MOE_SHEXP_GATE_SOURCE:  &str = include_str!("../../kernels/moe_shexp_gate.cpp");
const MOE_EXPERT_SORT_SOURCE: &str = include_str!("../../kernels/moe_expert_sort.cpp");
const MOE_MMQ_Q4K_GROUPED_SOURCE: &str =
    include_str!("../../kernels/mmq_gemm_q4k_grouped.cpp");
const MOE_MMQ_Q5K_GROUPED_SOURCE: &str =
    include_str!("../../kernels/mmq_gemm_q5k_grouped.cpp");
const MOE_MMQ_Q6K_GROUPED_SOURCE: &str =
    include_str!("../../kernels/mmq_gemm_q6k_grouped.cpp");

/// Token-tile width of the grouped-expert GEMM — `tile_off` counts
/// `ceil(tokens_per_expert / MOE_GEMM_BN)` tiles. Must match the BN
/// the grouped-GEMM kernel is compiled with (mmq_gemm_q4k_grouped.cpp).
const MOE_GEMM_BN: u32 = 32;
const MATVEC_Q4_K_DP4A_SOURCE: &str = include_str!("../../kernels/matvec_q4_k_dp4a.cpp");
const MATVEC_Q5_K_DP4A_SOURCE: &str = include_str!("../../kernels/matvec_q5_k_dp4a.cpp");
const MATVEC_Q6_K_DP4A_SOURCE: &str = include_str!("../../kernels/matvec_q6_k_dp4a.cpp");
const MATVEC_Q8_0_DP4A_SOURCE: &str = include_str!("../../kernels/matvec_q8_0_dp4a.cpp");
const MATVEC_Q4K_REPACKED_SOURCE: &str = include_str!("../../kernels/matvec_q4k_repacked.cpp");
const MMQ_GEMM_Q4K_SOURCE: &str = include_str!("../../kernels/mmq_gemm_q4k_repacked.cpp");
const MMQ_GEMM_Q5K_SOURCE: &str = include_str!("../../kernels/mmq_gemm_q5k_repacked.cpp");
const MMQ_GEMM_Q6K_SOURCE: &str = include_str!("../../kernels/mmq_gemm_q6k_repacked.cpp");
const MMQ_GEMM_Q8_0_SOURCE: &str = include_str!("../../kernels/mmq_gemm_q8_0_repacked.cpp");
const MATVEC_Q5K_REPACKED_SOURCE: &str = include_str!("../../kernels/matvec_q5k_repacked.cpp");
const MATVEC_Q6K_REPACKED_SOURCE: &str = include_str!("../../kernels/matvec_q6k_repacked.cpp");
const MATVEC_Q8_0_REPACKED_SOURCE: &str = include_str!("../../kernels/matvec_q8_0_repacked.cpp");
const MATVEC_Q4K_REPACKED_BATCHED_SOURCE: &str =
    include_str!("../../kernels/matvec_q4k_repacked_batched.cpp");
const MATVEC_Q5K_REPACKED_BATCHED_SOURCE: &str =
    include_str!("../../kernels/matvec_q5k_repacked_batched.cpp");
const MATVEC_Q6K_REPACKED_BATCHED_SOURCE: &str =
    include_str!("../../kernels/matvec_q6k_repacked_batched.cpp");
/// Output rows per wavefront in the dp4a matvec kernels (`#define ROWS`).
const DP4A_ROWBLOCK: u32 = 2;

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
    /// True when `data` holds the repacked two-plane Q4_K layout
    /// (see `quant::q4_k::repack_for_matvec`) — routed to the
    /// `matvec_q4k_repacked` kernel rather than the dp4a dispatch.
    pub repacked: bool,
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
            repacked: false,
        })
    }

    /// Like [`from_gguf`], but a Q4_K weight is repacked at load time into
    /// the contiguous two-plane layout the `matvec_q4k_repacked` kernel
    /// streams at near-peak bandwidth. Other dtypes load unchanged. Use
    /// this for pure matvec weights — not for `token_embd`, which is also
    /// the embedding-lookup table and must keep its on-disk layout.
    pub fn from_gguf_matvec(gguf: &GgufFile, name: &str) -> Result<Self, String> {
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
        let packed = match info.ggml_type {
            GgmlType::Q4_K => Some(crate::quant::q4_k::repack_for_matvec(
                bytes, in_dim as usize, out_dim as usize)),
            GgmlType::Q5_K => Some(crate::quant::q5_k::repack_for_matvec(
                bytes, in_dim as usize, out_dim as usize)),
            GgmlType::Q6_K => Some(crate::quant::q6_k::repack_for_matvec(
                bytes, in_dim as usize, out_dim as usize)),
            GgmlType::Q8_0 => Some(crate::quant::q8_0::repack_for_matvec(
                bytes, in_dim as usize, out_dim as usize)),
            _ => None,
        };
        match packed {
            Some(p) => Ok(Self {
                data: DeviceBuf::from_slice(&p)?,
                dtype: info.ggml_type,
                in_dim, out_dim,
                repacked: true,
            }),
            None => Ok(Self {
                data: DeviceBuf::from_slice(bytes)?,
                dtype: info.ggml_type,
                in_dim, out_dim,
                repacked: false,
            }),
        }
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
    /// `repack` selects the fast contiguous Q4_K layout (production) vs
    /// the on-disk layout (fp32-consistency tests, which need the wave64
    /// path `set_dp4a(false)` exercises).
    pub fn from_gguf(gguf: &GgufFile, layer: u32, repack: bool) -> Result<Self, String> {
        let mv = |n: &str| if repack { GpuMatvecTensor::from_gguf_matvec(gguf, n) }
                           else      { GpuMatvecTensor::from_gguf(gguf, n) };
        Ok(Self {
            gate: mv(&format!("blk.{layer}.ffn_gate.weight"))?,
            up:   mv(&format!("blk.{layer}.ffn_up.weight"))?,
            down: mv(&format!("blk.{layer}.ffn_down.weight"))?,
        })
    }
}

/// A 3D expert-weight tensor `[in_dim, out_dim, n_expert]` on device.
/// K-quant slices are repacked per expert into the contiguous matvec
/// layout (`quant::q*_k::repack_for_matvec`).
pub struct GpuExpertTensor {
    pub data:    DeviceBuf<u8>,
    pub dtype:   GgmlType,
    pub in_dim:  u32,
    pub out_dim: u32,
    pub bytes_per_expert: usize,
    pub repacked: bool,
}

impl GpuExpertTensor {
    pub fn from_gguf(gguf: &GgufFile, name: &str) -> Result<Self, String> {
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
        // K-quant experts repack per slice; other dtypes load on-disk.
        let repack_one = |slice: &[u8]| -> Option<Vec<u8>> {
            match info.ggml_type {
                GgmlType::Q4_K => Some(crate::quant::q4_k::repack_for_matvec(slice, in_dim, out_dim)),
                GgmlType::Q5_K => Some(crate::quant::q5_k::repack_for_matvec(slice, in_dim, out_dim)),
                GgmlType::Q6_K => Some(crate::quant::q6_k::repack_for_matvec(slice, in_dim, out_dim)),
                _ => None,
            }
        };
        if repack_one(&bytes[..bpe]).is_some() {
            let mut packed = Vec::new();
            for e in 0..n_expert {
                packed.extend_from_slice(&repack_one(&bytes[e * bpe..(e + 1) * bpe]).unwrap());
            }
            Ok(Self {
                bytes_per_expert: packed.len() / n_expert,
                dtype: info.ggml_type,
                in_dim: in_dim as u32, out_dim: out_dim as u32,
                data: DeviceBuf::from_slice(&packed)?,
                repacked: true,
            })
        } else {
            Ok(Self {
                bytes_per_expert: bpe,
                dtype: info.ggml_type,
                in_dim: in_dim as u32, out_dim: out_dim as u32,
                data: DeviceBuf::from_slice(bytes)?,
                repacked: false,
            })
        }
    }
}

/// MoE FFN weights for one `qwen35moe` block: a 256-expert routed branch
/// plus a sigmoid-gated shared expert.
pub struct GpuMoeFfn {
    /// Router projection, F32 `[hidden, n_expert]`.
    pub gate_inp:   GpuMatvecTensor,
    pub gate_exps:  GpuExpertTensor,   // Q4_K [hidden, expert_ff, n_expert]
    pub up_exps:    GpuExpertTensor,   // Q4_K
    pub down_exps:  GpuExpertTensor,   // Q5_K [expert_ff, hidden, n_expert]
    /// Shared-expert scalar gate, F32 `[hidden]`.
    pub gate_inp_shexp: DeviceBuf<f32>,
    pub gate_shexp: GpuMatvecTensor,   // [hidden, shared_expert_ff]
    pub up_shexp:   GpuMatvecTensor,
    pub down_shexp: GpuMatvecTensor,   // [shared_expert_ff, hidden]
}

impl GpuMoeFfn {
    pub fn from_gguf(gguf: &GgufFile, layer: u32, repack: bool) -> Result<Self, String> {
        let pre = format!("blk.{layer}.");
        let mv = |n: &str| if repack { GpuMatvecTensor::from_gguf_matvec(gguf, n) }
                           else      { GpuMatvecTensor::from_gguf(gguf, n) };
        Ok(Self {
            gate_inp:   GpuMatvecTensor::from_gguf(gguf, &format!("{pre}ffn_gate_inp.weight"))?,
            gate_exps:  GpuExpertTensor::from_gguf(gguf, &format!("{pre}ffn_gate_exps.weight"))?,
            up_exps:    GpuExpertTensor::from_gguf(gguf, &format!("{pre}ffn_up_exps.weight"))?,
            down_exps:  GpuExpertTensor::from_gguf(gguf, &format!("{pre}ffn_down_exps.weight"))?,
            gate_inp_shexp: load_fp32_tensor(gguf, &format!("{pre}ffn_gate_inp_shexp.weight"))?,
            gate_shexp: mv(&format!("{pre}ffn_gate_shexp.weight"))?,
            up_shexp:   mv(&format!("{pre}ffn_up_shexp.weight"))?,
            down_shexp: mv(&format!("{pre}ffn_down_shexp.weight"))?,
        })
    }
}

/// A block's FFN — dense SwiGLU (`qwen35`) or MoE (`qwen35moe`).
pub enum BlockFfn {
    Dense(GpuFfnWeights),
    Moe(GpuMoeFfn),
}

impl BlockFfn {
    pub fn from_gguf(gguf: &GgufFile, layer: u32, repack: bool, moe: bool)
        -> Result<Self, String>
    {
        if moe {
            Ok(BlockFfn::Moe(GpuMoeFfn::from_gguf(gguf, layer, repack)?))
        } else {
            Ok(BlockFfn::Dense(GpuFfnWeights::from_gguf(gguf, layer, repack)?))
        }
    }
}

/// Model-wide MoE runtime — kernel modules + scratch buffers, built once
/// for a `qwen35moe` model and shared across all blocks.
/// Prefill MoE batch size. `step_moe_ffn_batched` processes up to this
/// many tokens in one set of launches; longer prompts are chunked. The
/// MoE scratch buffers below are sized for it. 256 keeps the scratch
/// ~30 MB while still amortising expert-weight reads across the batch.
const MOE_PREFILL_CHUNK: usize = 256;

/// Upper bound on the batch size of `forward_tokens_verify` — sizes the
/// resident `verify_hidden` stash. The MTP spec-decode verify batch is
/// `k + 1` tokens (k drafts + the certain token); `qwen-verify-check`
/// also uses it. 16 covers any sane k.
const VERIFY_MAX_TOKENS: usize = 16;

/// A free-list pool of `DeviceBuf<T>` keyed by element count. Replaces
/// per-call `hipMalloc` in the batched prefill / verify path: the first
/// pass allocates, every later pass reuses. `take` hands out a
/// `PooledBuf` that returns its buffer to the pool on drop. Pooled
/// buffers are never freed mid-run, so kernels still reading them on the
/// single engine stream stay safe without a per-call sync — which is
/// what makes the per-round spec-decode verify cheap.
struct DeviceBufPool<T> {
    free: std::cell::RefCell<std::collections::HashMap<usize, Vec<DeviceBuf<T>>>>,
}

impl<T: Copy> DeviceBufPool<T> {
    fn new() -> Self {
        Self { free: std::cell::RefCell::new(std::collections::HashMap::new()) }
    }

    /// A buffer of exactly `len` elements — reused from the pool when
    /// available, freshly allocated otherwise. Contents are unspecified
    /// (same contract as `DeviceBuf::new`).
    fn take(&self, len: usize) -> Result<PooledBuf<'_, T>, String> {
        let reused = self.free.borrow_mut().get_mut(&len).and_then(|v| v.pop());
        let buf = match reused {
            Some(b) => b,
            None    => DeviceBuf::new(len)?,
        };
        Ok(PooledBuf { buf: Some(buf), pool: self, len })
    }
}

/// A `DeviceBuf` borrowed from a `DeviceBufPool`; returns to the pool
/// when dropped. Derefs to `DeviceBuf<T>` so call sites are unchanged.
struct PooledBuf<'a, T> {
    buf:  Option<DeviceBuf<T>>,
    pool: &'a DeviceBufPool<T>,
    len:  usize,
}

impl<T> Drop for PooledBuf<'_, T> {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            self.pool.free.borrow_mut().entry(self.len).or_default().push(b);
        }
    }
}

impl<T> std::ops::Deref for PooledBuf<'_, T> {
    type Target = DeviceBuf<T>;
    fn deref(&self) -> &DeviceBuf<T> { self.buf.as_ref().unwrap() }
}

struct MoeRuntime {
    n_expert: usize,
    n_used:   usize,
    expert_ff: usize,
    shexp_ff:  usize,
    m_topk:       Module,
    m_mv_q4k:     Module,
    m_mv_q5k:     Module,
    m_mv_q6k:     Module,
    /// Fused gate+up matvec + SwiGLU (Q4_K experts) — decode fast path.
    m_gate_up_swiglu_q4k: Module,
    /// Row-packed expert DOWN matvec — all 64 lanes busy at in_dim≈512.
    m_down_q5k:   Module,
    m_down_q6k:   Module,
    m_combine:    Module,
    m_shexp_gate: Module,
    /// Expert-routing counting sort (grouped-expert GEMM prefill path).
    m_expert_sort: Module,
    /// Grouped-expert MMQ GEMM — repacked Q4_K/Q5_K/Q6_K (MoE prefill).
    m_grouped_q4k: Module,
    m_grouped_q5k: Module,
    m_grouped_q6k: Module,
    // Scratch — sized for MOE_PREFILL_CHUNK rows; decode uses row 0 only.
    logits:  DeviceBuf<f32>,   // [n_tok, n_expert] router logits
    ids:     DeviceBuf<i32>,   // [n_tok, n_used] selected expert ids
    weights: DeviceBuf<f32>,   // [n_tok, n_used] renormalised routing weights
    // Expert-routing sort scratch — see kernels/moe_expert_sort.cpp.
    sort_count:  DeviceBuf<i32>,  // [n_expert]   histogram
    sort_cursor: DeviceBuf<i32>,  // [n_expert]   scatter cursor
    sort_eoff:   DeviceBuf<i32>,  // [n_expert+1] expert entry offsets
    sort_toff:   DeviceBuf<i32>,  // [n_expert+1] expert GEMM-tile offsets
    sort_perm:   DeviceBuf<i32>,  // [n_tok, n_used] entry indices grouped by expert
    g_in:    DeviceBuf<u8>,    // [n_tok, n_used, hidden/32] gathered (sorted) gate/up input
    g_out:   DeviceBuf<f32>,  // [n_tok, n_used, hidden] sorted down-GEMM output
    ones:    DeviceBuf<f32>,   // [n_expert] = 1.0 (combine has no per-expert scale)
    xq8_in:  DeviceBuf<u8>,    // [n_tok, hidden/32] quantised block input
    xq8_exp: DeviceBuf<u8>,    // [n_tok, n_used, expert_ff/32] quantised down acts
    e_gate:  DeviceBuf<f32>,   // [n_tok, n_used, expert_ff]
    e_up:    DeviceBuf<f32>,   // [n_tok, n_used, expert_ff]
    e_out:   DeviceBuf<f32>,   // [n_tok, n_used, hidden]
    sh_gate: DeviceBuf<f32>,   // [n_tok, shexp_ff]
    sh_up:   DeviceBuf<f32>,   // [n_tok, shexp_ff]
    sh_out:  DeviceBuf<f32>,   // [n_tok, hidden]
}

impl MoeRuntime {
    fn new(moe: &crate::model::qwen3_5::MoeConfig, hidden: usize, cache: &KernelCache)
        -> Result<Self, String>
    {
        let n_expert  = moe.n_expert as usize;
        let n_used    = moe.n_expert_used as usize;
        let expert_ff = moe.expert_ff as usize;
        let shexp_ff  = moe.shared_expert_ff as usize;
        let ones = DeviceBuf::from_slice(&vec![1.0f32; n_expert])?;
        let c = MOE_PREFILL_CHUNK;
        Ok(Self {
            n_expert, n_used, expert_ff, shexp_ff,
            m_topk:       Module::load(&cache.compile("moe_topk", MOE_TOPK_SOURCE)?)?,
            m_gate_up_swiglu_q4k: Module::load(&cache.compile(
                              "moe_gate_up_swiglu_q4k_repacked", MOE_GATE_UP_SWIGLU_Q4K_SOURCE)?)?,
            m_down_q5k:   Module::load(&cache.compile(
                              "moe_matvec_q5k_down", MOE_MV_Q5K_DOWN_SOURCE)?)?,
            m_down_q6k:   Module::load(&cache.compile(
                              "moe_matvec_q6k_down", MOE_MV_Q6K_DOWN_SOURCE)?)?,
            m_mv_q4k:     Module::load(&cache.compile(
                              "moe_matvec_q4k_repacked", MOE_MV_Q4K_REPACKED_SOURCE)?)?,
            m_mv_q5k:     Module::load(&cache.compile(
                              "moe_matvec_q5k_repacked", MOE_MV_Q5K_REPACKED_SOURCE)?)?,
            m_mv_q6k:     Module::load(&cache.compile(
                              "moe_matvec_q6k_repacked", MOE_MV_Q6K_REPACKED_SOURCE)?)?,
            m_combine:    Module::load(&cache.compile("moe_combine", MOE_COMBINE_SOURCE)?)?,
            m_shexp_gate: Module::load(&cache.compile("moe_shexp_gate", MOE_SHEXP_GATE_SOURCE)?)?,
            m_expert_sort: Module::load(&cache.compile(
                              "moe_expert_sort", MOE_EXPERT_SORT_SOURCE)?)?,
            m_grouped_q4k: Module::load(&cache.compile(
                              "mmq_gemm_q4k_grouped", MOE_MMQ_Q4K_GROUPED_SOURCE)?)?,
            m_grouped_q5k: Module::load(&cache.compile(
                              "mmq_gemm_q5k_grouped", MOE_MMQ_Q5K_GROUPED_SOURCE)?)?,
            m_grouped_q6k: Module::load(&cache.compile(
                              "mmq_gemm_q6k_grouped", MOE_MMQ_Q6K_GROUPED_SOURCE)?)?,
            logits:  DeviceBuf::new(c * n_expert)?,
            ids:     DeviceBuf::new(c * n_used)?,
            weights: DeviceBuf::new(c * n_used)?,
            sort_count:  DeviceBuf::new(n_expert)?,
            sort_cursor: DeviceBuf::new(n_expert)?,
            sort_eoff:   DeviceBuf::new(n_expert + 1)?,
            sort_toff:   DeviceBuf::new(n_expert + 1)?,
            sort_perm:   DeviceBuf::new(c * n_used)?,
            g_in:    DeviceBuf::new(c * n_used * (hidden / 32).max(1) * 40)?,
            g_out:   DeviceBuf::new(c * n_used * hidden)?,
            ones,
            xq8_in:  DeviceBuf::new(c * (hidden / 32).max(1) * 40)?,
            xq8_exp: DeviceBuf::new(c * n_used * (expert_ff / 32).max(1) * 40)?,
            e_gate:  DeviceBuf::new(c * n_used * expert_ff)?,
            e_up:    DeviceBuf::new(c * n_used * expert_ff)?,
            e_out:   DeviceBuf::new(c * n_used * hidden)?,
            sh_gate: DeviceBuf::new(c * shexp_ff)?,
            sh_up:   DeviceBuf::new(c * shexp_ff)?,
            sh_out:  DeviceBuf::new(c * hidden)?,
        })
    }
}

/// All weights for one full-attention transformer block on the GPU.
/// Bundles the attention sub-layer, the post-attention norm, and the
/// FFN sub-layer in the same lifetime.
pub struct GpuFullAttnBlock {
    pub attn:       GpuFullAttnWeights,
    pub post_norm:  DeviceBuf<f32>,    // [hidden] — pre-FFN RMSNorm weight
    pub ffn:        BlockFfn,
}

impl GpuFullAttnBlock {
    pub fn from_gguf(gguf: &GgufFile, layer: u32, repack: bool, moe: bool)
        -> Result<Self, String>
    {
        Ok(Self {
            attn:      GpuFullAttnWeights::from_gguf(gguf, layer, repack)?,
            post_norm: load_fp32_tensor(gguf, &format!("blk.{layer}.post_attention_norm.weight"))?,
            ffn:       BlockFfn::from_gguf(gguf, layer, repack, moe)?,
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
    pub fn from_gguf(gguf: &GgufFile, layer: u32, repack: bool) -> Result<Self, String> {
        let pre = format!("blk.{layer}.");
        let mv = |n: &str| if repack { GpuMatvecTensor::from_gguf_matvec(gguf, n) }
                           else      { GpuMatvecTensor::from_gguf(gguf, n) };
        Ok(Self {
            attn_norm:   load_fp32_tensor(gguf, &format!("{pre}attn_norm.weight"))?,
            attn_q:      mv(&format!("{pre}attn_q.weight"))?,
            attn_k:      mv(&format!("{pre}attn_k.weight"))?,
            attn_v:      mv(&format!("{pre}attn_v.weight"))?,
            attn_q_norm: load_fp32_tensor(gguf, &format!("{pre}attn_q_norm.weight"))?,
            attn_k_norm: load_fp32_tensor(gguf, &format!("{pre}attn_k_norm.weight"))?,
            attn_output: mv(&format!("{pre}attn_output.weight"))?,
        })
    }
}

/// MTP next-N predictor head — Unsloth's Qwen 3.6 MTP layout
/// ("DeepSeek-V3 style"). One full-attention transformer block plus the
/// MTP-specific tensors. Designed to produce a +2 token candidate from
/// `(prev_hidden, embed(next_tok))`:
///
///     concat(enorm(embed_next), hnorm(prev_hidden))   ─ [2·hidden]
///       → eh_proj → hidden                            ─ [hidden]
///       → full transformer block (attn + ffn)         ─ [hidden]
///       → shared_head_norm                            ─ [hidden]
///       → tied lm_head                                ─ [vocab]
///
/// Draft forward is `GpuQwen35::mtp_draft_forward`; the K=1 accept rate
/// can be measured with `mtp_accept_probe` (the `qwen-mtp-probe` CLI) —
/// ~79-83% on the 4B / 27B MTP builds. The earlier "~5%" estimate was
/// wrong: it assumed a sequential GDN verify, but reinstinct has batched
/// GDN inner-loop kernels, so a K-token verify is a single batched pass.
pub struct GpuMtpHead {
    pub block:            GpuFullAttnBlock,
    pub eh_proj:          GpuMatvecTensor,   // [2·hidden, hidden]
    pub enorm:            DeviceBuf<f32>,    // [hidden] — RMSNorm on embed_next
    pub hnorm:            DeviceBuf<f32>,    // [hidden] — RMSNorm on prev_hidden
    pub shared_head_norm: DeviceBuf<f32>,    // [hidden] — pre-lm_head RMSNorm
}

impl GpuMtpHead {
    pub fn from_gguf(gguf: &GgufFile, layer: u32, repack: bool, moe: bool)
        -> Result<Self, String>
    {
        let pre = format!("blk.{layer}.");
        let mv = |n: &str| if repack { GpuMatvecTensor::from_gguf_matvec(gguf, n) }
                           else      { GpuMatvecTensor::from_gguf(gguf, n) };
        Ok(Self {
            block:            GpuFullAttnBlock::from_gguf(gguf, layer, repack, moe)?,
            eh_proj:          mv(&format!("{pre}nextn.eh_proj.weight"))?,
            enorm:            load_fp32_tensor(gguf, &format!("{pre}nextn.enorm.weight"))?,
            hnorm:            load_fp32_tensor(gguf, &format!("{pre}nextn.hnorm.weight"))?,
            shared_head_norm: load_fp32_tensor(gguf, &format!("{pre}nextn.shared_head_norm.weight"))?,
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
    pub fn from_gguf(gguf: &GgufFile, layer: u32, repack: bool) -> Result<Self, String> {
        let pre = format!("blk.{layer}.");
        let mv = |n: &str| if repack { GpuMatvecTensor::from_gguf_matvec(gguf, n) }
                           else      { GpuMatvecTensor::from_gguf(gguf, n) };
        Ok(Self {
            attn_norm:   load_fp32_tensor(gguf, &format!("{pre}attn_norm.weight"))?,
            attn_qkv:    mv(&format!("{pre}attn_qkv.weight"))?,
            attn_gate:   mv(&format!("{pre}attn_gate.weight"))?,
            ssm_alpha:   mv(&format!("{pre}ssm_alpha.weight"))?,
            ssm_beta:    mv(&format!("{pre}ssm_beta.weight"))?,
            ssm_a:       load_fp32_tensor(gguf, &format!("{pre}ssm_a"))?,
            ssm_dt_bias: load_fp32_tensor(gguf, &format!("{pre}ssm_dt.bias"))?,
            ssm_conv1d:  load_fp32_tensor(gguf, &format!("{pre}ssm_conv1d.weight"))?,
            ssm_norm:    load_fp32_tensor(gguf, &format!("{pre}ssm_norm.weight"))?,
            ssm_out:     mv(&format!("{pre}ssm_out.weight"))?,
        })
    }
}

/// All weights for one linear-attention transformer block on the GPU
/// (GDN attention + post-norm + FFN).
pub struct GpuLinAttnBlock {
    pub attn:      GpuLinAttnWeights,
    pub post_norm: DeviceBuf<f32>,
    pub ffn:       BlockFfn,
}

impl GpuLinAttnBlock {
    pub fn from_gguf(gguf: &GgufFile, layer: u32, repack: bool, moe: bool)
        -> Result<Self, String>
    {
        Ok(Self {
            attn:      GpuLinAttnWeights::from_gguf(gguf, layer, repack)?,
            post_norm: load_fp32_tensor(gguf, &format!("blk.{layer}.post_attention_norm.weight"))?,
            ffn:       BlockFfn::from_gguf(gguf, layer, repack, moe)?,
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
    pub fn from_gguf(gguf: &GgufFile, layer: u32, kind: crate::model::qwen3_5::BlockKind,
                     repack: bool, moe: bool) -> Result<Self, String>
    {
        use crate::model::qwen3_5::BlockKind;
        Ok(match kind {
            BlockKind::FullAttention =>
                GpuBlock::Full(GpuFullAttnBlock::from_gguf(gguf, layer, repack, moe)?),
            BlockKind::LinearAttention =>
                GpuBlock::Linear(GpuLinAttnBlock::from_gguf(gguf, layer, repack, moe)?),
            BlockKind::NextN => unreachable!(
                "NextN blocks aren't loaded into GpuBlock — they're MTP \
                 heads tracked separately on the runtime"
            ),
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
        let conv_dim = cfg.gdn_qkv_concat_dim() as usize;
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
                BlockKind::NextN => unreachable!(
                    "NextN blocks have no main-forward state (MTP drafter)"
                ),
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

/// A rollback checkpoint for `Qwen35GpuState`, used by the MTP
/// spec-decode loop. Captures the GDN recurrent + conv state (a content
/// copy — they are mutated in place and cannot otherwise be recovered)
/// plus the per-block KV lengths and the position counter. KV cache
/// *contents* are NOT copied: slots past the restored length are simply
/// overwritten by the next forward. Allocate once with `new`, then
/// `save` before a speculative verify and `restore` to roll it back.
pub struct Qwen35Snapshot {
    /// Per Linear block (block order): (recurrent copy, conv_hist copy).
    /// `None` for Full blocks — their rollback is just the KV `len`.
    gdn:    Vec<Option<(DeviceBuf<f32>, DeviceBuf<f32>)>>,
    kv_len: Vec<usize>,   // per block (block order); meaningful for Full
    pos:    usize,
}

impl Qwen35Snapshot {
    pub fn new(state: &Qwen35GpuState) -> Result<Self, String> {
        let mut gdn = Vec::with_capacity(state.block_states.len());
        for bs in &state.block_states {
            gdn.push(match bs {
                GpuBlockState::Full(_)   => None,
                GpuBlockState::Linear(s) => Some((
                    DeviceBuf::new(s.recurrent.len())?,
                    DeviceBuf::new(s.conv_hist.len())?,
                )),
            });
        }
        Ok(Self { gdn, kv_len: vec![0; state.block_states.len()], pos: state.pos })
    }

    /// Capture `state` into this snapshot (reuses the allocated buffers).
    pub fn save(&mut self, state: &Qwen35GpuState) -> Result<(), String> {
        self.pos = state.pos;
        for (i, bs) in state.block_states.iter().enumerate() {
            match bs {
                GpuBlockState::Full(kv)  => self.kv_len[i] = kv.len,
                GpuBlockState::Linear(s) => {
                    let (r, c) = self.gdn[i].as_ref()
                        .ok_or("snapshot/state block-kind mismatch")?;
                    r.copy_from_device_at(&s.recurrent, 0)?;
                    c.copy_from_device_at(&s.conv_hist, 0)?;
                }
            }
        }
        Ok(())
    }

    /// Roll `state` back to the captured checkpoint.
    pub fn restore(&self, state: &mut Qwen35GpuState) -> Result<(), String> {
        state.pos = self.pos;
        for (i, bs) in state.block_states.iter_mut().enumerate() {
            match bs {
                GpuBlockState::Full(kv)  => kv.len = self.kv_len[i],
                GpuBlockState::Linear(s) => {
                    let (r, c) = self.gdn[i].as_ref()
                        .ok_or("snapshot/state block-kind mismatch")?;
                    s.recurrent.copy_from_device_at(r, 0)?;
                    s.conv_hist.copy_from_device_at(c, 0)?;
                }
            }
        }
        Ok(())
    }
}

/// Per-call stats from `mtp_spec_generate`.
#[derive(Debug, Default, Clone, Copy)]
pub struct QwenSpecStats {
    pub rounds:   usize,   // spec-decode rounds executed
    pub drafted:  usize,   // total MTP drafts proposed
    pub accepted: usize,   // drafts that survived verification
    pub hit_eos:  bool,
}

impl QwenSpecStats {
    pub fn accept_rate(&self) -> f64 {
        if self.drafted == 0 { 0.0 }
        else { self.accepted as f64 / self.drafted as f64 }
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
    /// MTP-head scratch: [0..2h] = concat(enorm·emb | hnorm·prev), [2h..3h] = block hidden.
    mtp_scratch: DeviceBuf<f32>,   // [3 * hidden]
    /// Holds the previous MTP block-hidden across a chained draft (the
    /// `prev_hidden` for chain link i+1). See `mtp_draft_chain`.
    mtp_chain_hid: DeviceBuf<f32>, // [hidden]
    /// Per-position hidden states (pre output-norm) stashed by the last
    /// `forward_tokens_verify` — the MTP spec-decode loop reads a row as
    /// the next round's `prev_hidden`.
    verify_hidden: DeviceBuf<f32>, // [VERIFY_MAX_TOKENS * hidden]
    /// Buffer pools for the batched prefill / verify path — replace
    /// per-call `hipMalloc` so the per-round spec-decode verify is cheap.
    pool_f32: DeviceBufPool<f32>,
    pool_u8:  DeviceBufPool<u8>,
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

    // Split-K decode-attention scratch (FlashDecoding partials).
    attn_o_partial: DeviceBuf<f32>,  // [n_heads, ATTN_MAX_SPLITS, head_dim]
    attn_m_partial: DeviceBuf<f32>,  // [n_heads, ATTN_MAX_SPLITS]
    attn_l_partial: DeviceBuf<f32>,  // [n_heads, ATTN_MAX_SPLITS]
    use_old_attn:   bool,            // REINSTINCT_OLD_ATTN
    /// Decode position, device-resident — lets the rope / KV-write /
    /// attention kernels read it so the forward is graph-capturable.
    d_pos:          DeviceBuf<u32>,
    /// `REINSTINCT_MOE_PROFILE` — per-stage decode timing (sync-per-lap,
    /// accumulated across layers/steps). See `prof_lap` / `moe_prof_report`.
    moe_prof_on:    bool,
    prof_mark:      std::cell::Cell<std::time::Instant>,
    prof_buckets:   std::cell::RefCell<Vec<(&'static str, f64)>>,

    // GDN scratch buffers.
    gdn_qkv:      DeviceBuf<f32>,  // [conv_dim]          mixed_qkv projection
    gdn_conv_out: DeviceBuf<f32>,  // [conv_dim]          conv1d output (post-silu)
    gdn_z:        DeviceBuf<f32>,  // [value_dim]         attn_gate projection
    gdn_a:        DeviceBuf<f32>,  // [n_heads]           ssm_alpha projection
    gdn_b:        DeviceBuf<f32>,  // [n_heads]           ssm_beta projection
    gdn_q:        DeviceBuf<f32>,  // [value_dim]         L2-normed Q (scaled)
    gdn_k:        DeviceBuf<f32>,  // [value_dim]         L2-normed K
    gdn_core_out: DeviceBuf<f32>,  // [value_dim]         core attn out / normed_out

    // RoPE tables resident on device.
    rope_cos: DeviceBuf<f32>,      // [max_seq, rotary_dim]
    rope_sin: DeviceBuf<f32>,      // [max_seq, rotary_dim]

    // Compiled kernel modules — keep alive for the lifetime of self.
    embed_module:            Module,
    rmsnorm_module:          Module,
    swiglu_module:           Module,
    rmsnorm_multihead_module: Module,
    split_q_gate_module:     Module,
    sigmoid_mul_module:      Module,
    rope_module:             Module,
    attn_step_module:        Module,
    /// Split-K decode attention (FlashDecoding) — partial + merge.
    attn_partial_module:     Module,
    attn_merge_module:       Module,
    /// f32 KV-cache write at the device-resident decode position.
    kv_write_module:         Module,
    add_inplace_module:      Module,
    gdn_recurrent_step_fused_module: Module,
    conv1d_step_silu_module:      Module,
    l2norm_qk_module:             Module,
    rmsnorm_gated_multihead_module: Module,
    // Batched (n_rows-collapsed) variants used by forward_tokens_batched
    // to replace the per-row launch loops.
    conv1d_step_silu_batched_module: Module,
    l2norm_qk_batched_module: Module,
    gdn_recurrent_step_fused_batched_module: Module,
    rmsnorm_gated_multihead_batched_module: Module,

    matvec_f16_module:     Module,
    /// 256-thread/block fp32 matvec — wins over the wave64 path on
    /// small `out_dim` matvecs where wave64 starves the GPU. Used by
    /// the GDN `ssm_alpha` / `ssm_beta` projections (out_dim=n_heads=48).
    matvec_f32_b256_module: Module,
    embed_lookup_q6_k_module: Module,
    embed_lookup_q4_k_module: Module,
    embed_lookup_q8_0_module: Module,
    matvec_q4_k_wave64_module:   Module,
    matvec_q5_k_wave64_module:   Module,
    matvec_q6_k_wave64_module:   Module,
    matvec_q8_0_wave64_module:   Module,
    matvec_iq4_xs_wave64_module: Module,
    matvec_f16_wave64_module:    Module,

    // int8 dp4a matvec: quantize the activation once, then v_dot4_i32_i8.
    quantize_q8_module:      Module,
    matvec_q4_k_dp4a_module: Module,
    matvec_q5_k_dp4a_module: Module,
    matvec_q6_k_dp4a_module: Module,
    matvec_q8_0_dp4a_module: Module,
    matvec_q4k_repacked_module: Module,
    matvec_q5k_repacked_module: Module,
    matvec_q6k_repacked_module: Module,
    /// K=2..4 batched K-quant matvec — the spec-decode verify path.
    matvec_q4k_batched_module: Module,
    matvec_q5k_batched_module: Module,
    matvec_q6k_batched_module: Module,
    matvec_q8_0_repacked_module: Module,
    /// Scratch for the quantized activation (BlockQ8, 40 bytes per 32).
    xq8: DeviceBuf<u8>,
    /// Use the int8 dp4a matvec (vs the fp32 wave64 path). Env-set once;
    /// `set_dp4a` lets the GPU-vs-CPU consistency tests force fp32.
    dp4a_enabled: bool,

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
    dequant_q4k_repacked_module: Module,
    dequant_q5k_repacked_module: Module,
    dequant_q6k_repacked_module: Module,
    dequant_q8_0_repacked_module: Module,
    rope_batched_module:   Module,
    attn_step_batched_module: Module,
    /// 2D-tiled int8 MMQ GEMM — replaces dequant+HGEMM for repacked
    /// K-quants AND repacked Q8_0 in the dense prefill.
    mmq_q4k_module:        Module,
    mmq_q5k_module:        Module,
    mmq_q6k_module:        Module,
    mmq_q8_0_module:       Module,

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
    gdn_key_dim:     usize,
    gdn_conv_dim:    usize,
    gdn_n_heads:     usize,   // value heads (= recurrent states)
    gdn_n_k_heads:   usize,   // key/query heads
    gdn_head_dim:    usize,
    gdn_conv_kernel: usize,
    rms_eps:    f32,
    #[allow(dead_code)]
    max_seq:    usize,

    /// `Some` for a `qwen35moe` model — MoE FFN kernels + scratch.
    moe: Option<MoeRuntime>,

    /// MTP next-N predictor heads (Unsloth Qwen 3.6 MTP release). One
    /// entry per `nextn_predict_layers` — usually 1 if present.
    /// Loaded once at startup so the GGUF parses cleanly; not yet
    /// wired into any forward path. See `GpuMtpHead` for the reason.
    mtp: Vec<GpuMtpHead>,
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
        // GDN dims. Qwen 3.5 GDN is GQA: gdn_n_heads value heads,
        // gdn_n_k_heads key/query heads. The 0.8B has them equal.
        let gdn_value_dim   = cfg.gdn_value_dim   as usize;
        let gdn_n_heads     = cfg.gdn_n_heads     as usize;
        let gdn_n_k_heads   = cfg.gdn_n_k_heads   as usize;
        let gdn_head_dim    = cfg.gdn_head_dim    as usize;
        let gdn_conv_kernel = cfg.gdn_conv_kernel as usize;
        let gdn_key_dim     = cfg.gdn_key_dim()   as usize;
        // conv operates on q ‖ k ‖ v = 2 × key_dim + value_dim.
        let gdn_conv_dim    = cfg.gdn_qkv_concat_dim() as usize;
        // `xq8` holds the int8-quantised activation for every
        // launch_matvec_dispatch call — size it for the widest such
        // activation. The GDN out-projection (in = value_dim) and the
        // shared-expert down (in = shexp_ff) are wider than ffn/hidden
        // on the MoE models; omitting them overflowed xq8 → GPU fault.
        let xq8_max_in = hidden.max(ffn).max(q_dim)
            .max(gdn_value_dim).max(gdn_conv_dim)
            .max(cfg.moe.as_ref().map(|m| m.shared_expert_ff as usize).unwrap_or(0));

        let token_embd  = GpuMatvecTensor::from_gguf(gguf, "token_embd.weight")?;
        let output_norm = load_fp32_tensor(gguf, "output_norm.weight")?;
        let output_proj = if cfg.tied_embeddings {
            None
        } else {
            Some(GpuMatvecTensor::from_gguf(gguf, "output.weight")?)
        };

        let hidden_a    = DeviceBuf::new(hidden)?;
        let hidden_b    = DeviceBuf::new(hidden)?;
        let mtp_scratch = DeviceBuf::new(3 * hidden)?;
        let mtp_chain_hid = DeviceBuf::new(hidden)?;
        let verify_hidden = DeviceBuf::new(VERIFY_MAX_TOKENS * hidden)?;
        let pool_f32      = DeviceBufPool::new();
        let pool_u8       = DeviceBufPool::new();
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
        let attn_o_partial = DeviceBuf::new(n_heads * ATTN_MAX_SPLITS as usize * head_dim)?;
        let attn_m_partial = DeviceBuf::new(n_heads * ATTN_MAX_SPLITS as usize)?;
        let attn_l_partial = DeviceBuf::new(n_heads * ATTN_MAX_SPLITS as usize)?;
        let use_old_attn   = std::env::var_os("REINSTINCT_OLD_ATTN").is_some();
        let d_pos          = DeviceBuf::new(1)?;
        let moe_prof_on    = std::env::var_os("REINSTINCT_MOE_PROFILE").is_some();
        let prof_mark      = std::cell::Cell::new(std::time::Instant::now());
        let prof_buckets   = std::cell::RefCell::new(Vec::new());

        let gdn_qkv      = DeviceBuf::new(gdn_conv_dim)?;
        let gdn_conv_out = DeviceBuf::new(gdn_conv_dim)?;
        let gdn_z        = DeviceBuf::new(gdn_value_dim)?;
        let gdn_a        = DeviceBuf::new(gdn_n_heads)?;
        let gdn_b        = DeviceBuf::new(gdn_n_heads)?;
        let gdn_q        = DeviceBuf::new(gdn_key_dim)?;
        let gdn_k        = DeviceBuf::new(gdn_key_dim)?;
        let gdn_core_out = DeviceBuf::new(gdn_value_dim)?;

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
        let swiglu_hsaco            = cache.compile("swiglu",            SWIGLU_SOURCE)?;
        let rmsnorm_multihead_hsaco = cache.compile("rmsnorm_multihead", RMSNORM_MULTIHEAD_SOURCE)?;
        let split_q_gate_hsaco      = cache.compile("split_q_gate",      SPLIT_Q_GATE_SOURCE)?;
        let sigmoid_mul_hsaco       = cache.compile("sigmoid_mul",       SIGMOID_MUL_SOURCE)?;
        let rope_hsaco              = cache.compile("rope",              ROPE_SOURCE)?;
        let attn_step_hsaco         = cache.compile("attn_step",         ATTN_STEP_SOURCE)?;
        let attn_partial_hsaco      = cache.compile("attn_partial_f32",  ATTN_PARTIAL_F32_SOURCE)?;
        let attn_merge_hsaco        = cache.compile("attn_merge",        ATTN_MERGE_SOURCE)?;
        let kv_write_hsaco          = cache.compile("kv_write_f32",      KV_WRITE_F32_SOURCE)?;
        let add_inplace_hsaco       = cache.compile("add_inplace",       ADD_INPLACE_SOURCE)?;
        let gdn_recurrent_step_fused_hsaco = cache.compile("gdn_recurrent_step_fused", GDN_RECURRENT_STEP_FUSED_SOURCE)?;
        let conv1d_step_silu_hsaco       = cache.compile("conv1d_step_silu", CONV1D_STEP_SILU_SOURCE)?;
        let l2norm_qk_hsaco              = cache.compile("l2norm_qk",        L2NORM_QK_SOURCE)?;
        let rmsnorm_gated_multihead_hsaco = cache.compile("rmsnorm_gated_multihead", RMSNORM_GATED_MULTIHEAD_SOURCE)?;
        let conv1d_step_silu_batched_hsaco = cache.compile(
            "conv1d_step_silu_batched", CONV1D_STEP_SILU_BATCHED_SOURCE)?;
        let l2norm_qk_batched_hsaco = cache.compile(
            "l2norm_qk_batched", L2NORM_QK_BATCHED_SOURCE)?;
        let gdn_recurrent_step_fused_batched_hsaco = cache.compile(
            "gdn_recurrent_step_fused_batched", GDN_RECURRENT_STEP_FUSED_BATCHED_SOURCE)?;
        let rmsnorm_gated_multihead_batched_hsaco = cache.compile(
            "rmsnorm_gated_multihead_batched", RMSNORM_GATED_MULTIHEAD_BATCHED_SOURCE)?;
        let matvec_f16_hsaco    = cache.compile("matvec_f16",    MATVEC_F16_SOURCE)?;
        let matvec_f32_b256_hsaco = cache.compile("matvec_f32_b256", MATVEC_F32_B256_SOURCE)?;
        let embed_lookup_q6_k_hsaco = cache.compile("embed_lookup_q6_k", EMBED_LOOKUP_Q6_K_SOURCE)?;
        let embed_lookup_q4_k_hsaco = cache.compile("embed_lookup_q4_k", EMBED_LOOKUP_Q4_K_SOURCE)?;
        let embed_lookup_q8_0_hsaco = cache.compile("embed_lookup_q8_0_v", EMBED_LOOKUP_Q8_0_SOURCE)?;
        let matvec_q4_k_wave64_hsaco   = cache.compile("matvec_q4_k_wave64",   MATVEC_Q4_K_WAVE64_SOURCE)?;
        let matvec_q5_k_wave64_hsaco   = cache.compile("matvec_q5_k_wave64",   MATVEC_Q5_K_WAVE64_SOURCE)?;
        let matvec_q6_k_wave64_hsaco   = cache.compile("matvec_q6_k_wave64",   MATVEC_Q6_K_WAVE64_SOURCE)?;
        let matvec_q8_0_wave64_hsaco   = cache.compile("matvec_q8_0_wave64",   MATVEC_Q8_0_WAVE64_SOURCE)?;
        let matvec_iq4_xs_wave64_hsaco = cache.compile("matvec_iq4_xs_wave64", MATVEC_IQ4_XS_WAVE64_SOURCE)?;
        let matvec_f16_wave64_hsaco    = cache.compile("matvec_f16_wave64",    MATVEC_F16_WAVE64_SOURCE)?;
        let quantize_q8_hsaco      = cache.compile("quantize_q8",      QUANTIZE_Q8_SOURCE)?;
        let matvec_q4_k_dp4a_hsaco = cache.compile("matvec_q4_k_dp4a", MATVEC_Q4_K_DP4A_SOURCE)?;
        let matvec_q5_k_dp4a_hsaco = cache.compile("matvec_q5_k_dp4a", MATVEC_Q5_K_DP4A_SOURCE)?;
        let matvec_q6_k_dp4a_hsaco = cache.compile("matvec_q6_k_dp4a", MATVEC_Q6_K_DP4A_SOURCE)?;
        let matvec_q8_0_dp4a_hsaco = cache.compile("matvec_q8_0_dp4a", MATVEC_Q8_0_DP4A_SOURCE)?;
        let matvec_q4k_repacked_hsaco =
            cache.compile("matvec_q4k_repacked", MATVEC_Q4K_REPACKED_SOURCE)?;
        let matvec_q5k_repacked_hsaco =
            cache.compile("matvec_q5k_repacked", MATVEC_Q5K_REPACKED_SOURCE)?;
        let matvec_q6k_repacked_hsaco =
            cache.compile("matvec_q6k_repacked", MATVEC_Q6K_REPACKED_SOURCE)?;
        let matvec_q8_0_repacked_hsaco =
            cache.compile("matvec_q8_0_repacked", MATVEC_Q8_0_REPACKED_SOURCE)?;
        let matvec_q4k_batched_hsaco =
            cache.compile("matvec_q4k_repacked_batched", MATVEC_Q4K_REPACKED_BATCHED_SOURCE)?;
        let matvec_q5k_batched_hsaco =
            cache.compile("matvec_q5k_repacked_batched", MATVEC_Q5K_REPACKED_BATCHED_SOURCE)?;
        let matvec_q6k_batched_hsaco =
            cache.compile("matvec_q6k_repacked_batched", MATVEC_Q6K_REPACKED_BATCHED_SOURCE)?;

        // Load every per-layer block's weights from GGUF.
        let mut blocks = Vec::with_capacity(model.block_kinds.len());
        for (i, &kind) in model.block_kinds.iter().enumerate() {
            blocks.push(GpuBlock::from_gguf(gguf, i as u32, kind, true, model.config.is_moe())?);
        }
        // MTP next-N predictor heads (Unsloth Qwen 3.6 MTP). Loaded once
        // here; invoked only by the spec-decode drafter.
        let mtp: Vec<GpuMtpHead> = model.mtp_block_kinds().iter()
            .map(|&(i, _kind)| GpuMtpHead::from_gguf(gguf, i, true, model.config.is_moe()))
            .collect::<Result<_, _>>()?;
        let moe_runtime = match &cfg.moe {
            Some(mc) => Some(MoeRuntime::new(mc, hidden, cache)?),
            None => None,
        };

        // The single stream all launches flow through.
        let stream = Stream::new()?;
        // rocBLAS handle for batched-prefill GEMMs, bound to our stream.
        let rocblas_handle = RocblasHandle::new()?;
        rocblas_handle.set_stream(&stream)?;

        Ok(Self {
            token_embd, output_norm, output_proj,
            hidden_a, hidden_b, mtp_scratch, mtp_chain_hid, verify_hidden,
            pool_f32, pool_u8, ffn_a, ffn_b,
            q_raw, q_buf, gate_buf, k_raw, v_raw, k_norm, attn_concat, logits,
            attn_o_partial, attn_m_partial, attn_l_partial, use_old_attn, d_pos,
            moe_prof_on, prof_mark, prof_buckets,
            rope_cos, rope_sin,
            embed_module:             Module::load(&embed_hsaco)?,
            rmsnorm_module:           Module::load(&rmsnorm_hsaco)?,
            swiglu_module:            Module::load(&swiglu_hsaco)?,
            rmsnorm_multihead_module: Module::load(&rmsnorm_multihead_hsaco)?,
            split_q_gate_module:      Module::load(&split_q_gate_hsaco)?,
            sigmoid_mul_module:       Module::load(&sigmoid_mul_hsaco)?,
            rope_module:              Module::load(&rope_hsaco)?,
            attn_step_module:         Module::load(&attn_step_hsaco)?,
            attn_partial_module:      Module::load(&attn_partial_hsaco)?,
            attn_merge_module:        Module::load(&attn_merge_hsaco)?,
            kv_write_module:          Module::load(&kv_write_hsaco)?,
            add_inplace_module:       Module::load(&add_inplace_hsaco)?,
            gdn_recurrent_step_fused_module: Module::load(&gdn_recurrent_step_fused_hsaco)?,
            conv1d_step_silu_module:      Module::load(&conv1d_step_silu_hsaco)?,
            l2norm_qk_module:             Module::load(&l2norm_qk_hsaco)?,
            rmsnorm_gated_multihead_module: Module::load(&rmsnorm_gated_multihead_hsaco)?,
            conv1d_step_silu_batched_module:
                Module::load(&conv1d_step_silu_batched_hsaco)?,
            l2norm_qk_batched_module:
                Module::load(&l2norm_qk_batched_hsaco)?,
            gdn_recurrent_step_fused_batched_module:
                Module::load(&gdn_recurrent_step_fused_batched_hsaco)?,
            rmsnorm_gated_multihead_batched_module:
                Module::load(&rmsnorm_gated_multihead_batched_hsaco)?,
            matvec_f16_module:    Module::load(&matvec_f16_hsaco)?,
            matvec_f32_b256_module: Module::load(&matvec_f32_b256_hsaco)?,
            embed_lookup_q6_k_module: Module::load(&embed_lookup_q6_k_hsaco)?,
            embed_lookup_q4_k_module: Module::load(&embed_lookup_q4_k_hsaco)?,
            embed_lookup_q8_0_module: Module::load(&embed_lookup_q8_0_hsaco)?,
            rocblas:                  rocblas_handle,
            cvt_module:               Module::load(&cache.compile("cvt_f32_f16", CVT_F32_F16_SOURCE)?)?,
            dequant_q4_k_module:      Module::load(&cache.compile("dequant_q4_k_f16", DEQUANT_Q4_K_F16_SOURCE)?)?,
            dequant_q5_k_module:      Module::load(&cache.compile("dequant_q5_k_f16", DEQUANT_Q5_K_F16_SOURCE)?)?,
            dequant_q6_k_module:      Module::load(&cache.compile("dequant_q6_k_f16", DEQUANT_Q6_K_F16_SOURCE)?)?,
            dequant_q8_0_module:      Module::load(&cache.compile("dequant_q8_0_f16", DEQUANT_Q8_0_F16_SOURCE)?)?,
            dequant_iq4_xs_module:    Module::load(&cache.compile("dequant_iq4_xs_f16", DEQUANT_IQ4_XS_F16_SOURCE)?)?,
            dequant_q4k_repacked_module: Module::load(&cache.compile(
                "dequant_q4k_repacked_f16", DEQUANT_Q4K_REPACKED_F16_SOURCE)?)?,
            dequant_q5k_repacked_module: Module::load(&cache.compile(
                "dequant_q5k_repacked_f16", DEQUANT_Q5K_REPACKED_F16_SOURCE)?)?,
            dequant_q6k_repacked_module: Module::load(&cache.compile(
                "dequant_q6k_repacked_f16", DEQUANT_Q6K_REPACKED_F16_SOURCE)?)?,
            dequant_q8_0_repacked_module: Module::load(&cache.compile(
                "dequant_q8_0_repacked_f16", DEQUANT_Q8_0_REPACKED_F16_SOURCE)?)?,
            rope_batched_module:      Module::load(&cache.compile("rope_batched", ROPE_BATCHED_SOURCE)?)?,
            attn_step_batched_module: Module::load(&cache.compile("attn_prefill_flash", ATTN_STEP_BATCHED_SOURCE)?)?,
            mmq_q4k_module:           Module::load(&cache.compile("mmq_gemm_q4k_repacked", MMQ_GEMM_Q4K_SOURCE)?)?,
            mmq_q8_0_module:          Module::load(&cache.compile("mmq_gemm_q8_0_repacked", MMQ_GEMM_Q8_0_SOURCE)?)?,
            mmq_q5k_module:           Module::load(&cache.compile("mmq_gemm_q5k_repacked", MMQ_GEMM_Q5K_SOURCE)?)?,
            mmq_q6k_module:           Module::load(&cache.compile("mmq_gemm_q6k_repacked", MMQ_GEMM_Q6K_SOURCE)?)?,
            matvec_q4_k_wave64_module:   Module::load(&matvec_q4_k_wave64_hsaco)?,
            matvec_q5_k_wave64_module:   Module::load(&matvec_q5_k_wave64_hsaco)?,
            matvec_q6_k_wave64_module:   Module::load(&matvec_q6_k_wave64_hsaco)?,
            matvec_q8_0_wave64_module:   Module::load(&matvec_q8_0_wave64_hsaco)?,
            matvec_iq4_xs_wave64_module: Module::load(&matvec_iq4_xs_wave64_hsaco)?,
            matvec_f16_wave64_module:    Module::load(&matvec_f16_wave64_hsaco)?,
            quantize_q8_module:      Module::load(&quantize_q8_hsaco)?,
            matvec_q4_k_dp4a_module: Module::load(&matvec_q4_k_dp4a_hsaco)?,
            matvec_q5_k_dp4a_module: Module::load(&matvec_q5_k_dp4a_hsaco)?,
            matvec_q6_k_dp4a_module: Module::load(&matvec_q6_k_dp4a_hsaco)?,
            matvec_q8_0_dp4a_module: Module::load(&matvec_q8_0_dp4a_hsaco)?,
            matvec_q4k_repacked_module: Module::load(&matvec_q4k_repacked_hsaco)?,
            matvec_q5k_repacked_module: Module::load(&matvec_q5k_repacked_hsaco)?,
            matvec_q6k_repacked_module: Module::load(&matvec_q6k_repacked_hsaco)?,
            matvec_q4k_batched_module: Module::load(&matvec_q4k_batched_hsaco)?,
            matvec_q5k_batched_module: Module::load(&matvec_q5k_batched_hsaco)?,
            matvec_q6k_batched_module: Module::load(&matvec_q6k_batched_hsaco)?,
            matvec_q8_0_repacked_module: Module::load(&matvec_q8_0_repacked_hsaco)?,
            xq8: DeviceBuf::new(((xq8_max_in + 31) / 32) * 40)?,
            dp4a_enabled: std::env::var_os("REINSTINCT_QWEN_NO_DP4A").is_none(),
            blocks,
            stream,
            hidden, ffn, vocab, n_heads, n_kv_heads, head_dim, rotary_dim,
            gdn_value_dim, gdn_key_dim, gdn_conv_dim, gdn_n_heads, gdn_n_k_heads,
            gdn_head_dim, gdn_conv_kernel,
            gdn_qkv, gdn_conv_out, gdn_z, gdn_a, gdn_b, gdn_q, gdn_k,
            gdn_core_out,
            rms_eps: cfg.rms_norm_eps,
            max_seq,
            moe: moe_runtime,
            mtp,
        })
    }

    /// `true` when this GGUF carries one or more MTP next-N predictor
    /// heads — the spec-decode drafter can call `mtp_forward`.
    pub fn has_mtp(&self) -> bool { !self.mtp.is_empty() }

    /// Number of MTP heads available (= `nextn_predict_layers`).
    pub fn n_mtp_heads(&self) -> usize { self.mtp.len() }

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

    fn launch_embed_lookup_q8_0(&self, table: *mut c_void, out: *mut c_void,
                                token: u32, hidden: u32) -> Result<(), String>
    {
        let f = self.embed_lookup_q8_0_module.function("embed_lookup_q8_0_v_f32")?;
        let block: u32 = 256;
        let grid = (hidden + block - 1) / block;
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
            GgmlType::Q8_0 => self.launch_embed_lookup_q8_0(table.data.raw_ptr(), out, token, hidden),
            other => Err(format!("embed_lookup: no kernel for {:?}", other)),
        }
    }

    fn launch_rmsnorm(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void,
                      n: u32, eps: f32) -> Result<(), String>
    {
        let f = self.rmsnorm_module.function("rmsnorm_f32")?;
        // One block per vector — use the full 1024 (16 wavefronts) so the
        // single occupied CU has enough wavefronts to hide memory latency.
        let block: u32 = 1024;
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

    /// Per-quant-type matvec launchers. All fused dequant+GEMV kernels
    /// share the (W bytes, x f32, y f32, in_dim, out_dim) interface.
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

    // ===== batched (n_rows-collapsed) GDN launchers, used by prefill =====
    // Each of these replaces a per-row launch loop with a single launch
    // whose kernel loops over n_rows internally. State (conv history,
    // recurrent matrix) threads through the inner loop on-GPU.

    #[allow(clippy::too_many_arguments)]
    fn launch_conv1d_step_silu_batched(&self,
        x_new_batch: *mut c_void, w: *mut c_void, history: *mut c_void,
        y_batch: *mut c_void, n_channels: u32, kernel_size: u32, n_rows: u32)
        -> Result<(), String>
    {
        let f = self.conv1d_step_silu_batched_module.function("conv1d_step_silu_batched_f32")?;
        let block: u32 = 256;
        let grid = (n_channels + block - 1) / block;
        let mut xa=x_new_batch; let mut wa=w; let mut ha=history; let mut ya=y_batch;
        let mut nc=n_channels; let mut ks=kernel_size; let mut nr=n_rows;
        let mut args: [*mut c_void; 7] = [
            &mut xa as *mut _ as *mut c_void, &mut wa as *mut _ as *mut c_void,
            &mut ha as *mut _ as *mut c_void, &mut ya as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void, &mut ks as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_l2norm_qk_batched(&self,
        q_in: *mut c_void, q_out: *mut c_void,
        k_in: *mut c_void, k_out: *mut c_void,
        n_heads: u32, head_dim: u32, eps: f32, q_scale: f32,
        n_rows: u32,
        q_in_row_stride: u32, q_out_row_stride: u32,
        k_in_row_stride: u32, k_out_row_stride: u32)
        -> Result<(), String>
    {
        let f = self.l2norm_qk_batched_module.function("l2norm_qk_batched_f32")?;
        let block: u32 = 128;
        let mut qa=q_in; let mut qo=q_out; let mut ka=k_in; let mut ko=k_out;
        let mut nh=n_heads; let mut hd=head_dim; let mut ep=eps; let mut sc=q_scale;
        let mut nr=n_rows;
        let mut qis=q_in_row_stride; let mut qos=q_out_row_stride;
        let mut kis=k_in_row_stride; let mut kos=k_out_row_stride;
        let mut args: [*mut c_void; 13] = [
            &mut qa as *mut _ as *mut c_void, &mut qo as *mut _ as *mut c_void,
            &mut ka as *mut _ as *mut c_void, &mut ko as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut qis as *mut _ as *mut c_void, &mut qos as *mut _ as *mut c_void,
            &mut kis as *mut _ as *mut c_void, &mut kos as *mut _ as *mut c_void];
        let smem = block * 4;
        unsafe { f.launch((n_heads, 2, n_rows), (block, 1, 1), smem,
                          Some(&self.stream), &mut args) }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_gdn_recurrent_step_fused_batched(&self,
        q_in: *mut c_void, k_in: *mut c_void, v_in: *mut c_void,
        a_in: *mut c_void, b_in: *mut c_void,
        ssm_a: *mut c_void, dt_bias: *mut c_void,
        state: *mut c_void, out: *mut c_void,
        n_heads: u32, head_dim: u32, n_k_heads: u32, n_rows: u32,
        qk_row_stride: u32, v_row_stride: u32, ab_row_stride: u32,
        out_row_stride: u32) -> Result<(), String>
    {
        let f = self.gdn_recurrent_step_fused_batched_module
            .function("gdn_recurrent_step_fused_batched_f32")?;
        const COLS: u32 = 16;
        let block: u32 = 64;
        let smem = 2 * head_dim * 4;
        let mut qa=q_in; let mut ka=k_in; let mut va=v_in;
        let mut aa=a_in; let mut ba=b_in;
        let mut sa=ssm_a; let mut dta=dt_bias;
        let mut st=state; let mut ou=out;
        let mut nh=n_heads; let mut hd=head_dim; let mut nkh=n_k_heads; let mut nr=n_rows;
        let mut qrs=qk_row_stride; let mut vrs=v_row_stride;
        let mut abs_=ab_row_stride; let mut ors=out_row_stride;
        let mut args: [*mut c_void; 17] = [
            &mut qa as *mut _ as *mut c_void, &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void, &mut aa as *mut _ as *mut c_void,
            &mut ba as *mut _ as *mut c_void, &mut sa as *mut _ as *mut c_void,
            &mut dta as *mut _ as *mut c_void, &mut st as *mut _ as *mut c_void,
            &mut ou as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut nkh as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void, &mut qrs as *mut _ as *mut c_void,
            &mut vrs as *mut _ as *mut c_void, &mut abs_ as *mut _ as *mut c_void,
            &mut ors as *mut _ as *mut c_void];
        unsafe { f.launch((n_heads, head_dim / COLS, 1), (block, 1, 1), smem,
                          Some(&self.stream), &mut args) }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_rmsnorm_gated_multihead_batched(&self,
        x_batch: *mut c_void, z_batch: *mut c_void, w: *mut c_void,
        y_batch: *mut c_void, n_heads: u32, head_dim: u32, eps: f32,
        n_rows: u32, row_stride: u32) -> Result<(), String>
    {
        let f = self.rmsnorm_gated_multihead_batched_module
            .function("rmsnorm_gated_multihead_batched_f32")?;
        let block: u32 = 128;
        let mut xa=x_batch; let mut za=z_batch; let mut wa=w; let mut ya=y_batch;
        let mut nh=n_heads; let mut hd=head_dim; let mut ep=eps;
        let mut nr=n_rows; let mut rs=row_stride;
        let mut args: [*mut c_void; 9] = [
            &mut xa as *mut _ as *mut c_void, &mut za as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void, &mut ya as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut ep as *mut _ as *mut c_void, &mut nr as *mut _ as *mut c_void,
            &mut rs as *mut _ as *mut c_void];
        let smem = block * 4;
        unsafe { f.launch((n_heads, n_rows, 1), (block, 1, 1), smem,
                          Some(&self.stream), &mut args) }
    }

    /// 256-thread-block fp32 matvec — one block per output row, 4 waves
    /// of 64 each reduce through a 4-element LDS slot. Wins on small
    /// `out_dim` where the wave64 layout (one wavefront per row) leaves
    /// the GPU idle. See kernels/matvec_f32_b256.cpp.
    fn launch_matvec_f32_b256(&self, w: *mut c_void, x: *mut c_void, y: *mut c_void,
                              in_dim: u32, out_dim: u32) -> Result<(), String>
    {
        let f = self.matvec_f32_b256_module.function("matvec_f32_b256")?;
        let mut wa = w; let mut xa = x; let mut ya = y;
        let mut ia = in_dim; let mut oa = out_dim;
        let mut args: [*mut c_void; 5] = [
            &mut wa as *mut _ as *mut c_void,
            &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((out_dim, 1, 1), (256, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    /// Force the fp32 or int8-dp4a matvec path. Used by the GPU-vs-CPU
    /// consistency tests to compare against the fp32 CPU oracle.
    pub fn set_dp4a(&mut self, on: bool) { self.dp4a_enabled = on; }

    /// Quantize an fp32 activation row to int8 `BlockQ8` (40 B / 32 vals)
    /// into the shared `xq8` scratch — the dp4a matvec's left input.
    fn launch_quantize_q8(&self, x: *mut c_void, in_dim: u32) -> Result<(), String> {
        let f = self.quantize_q8_module.function("quantize_q8_f32")?;
        let mut xa = x; let mut oa = self.xq8.raw_ptr(); let mut ia = in_dim;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void];
        unsafe { f.launch(((in_dim + 255) / 256, 1, 1), (256, 1, 1),
                          0, Some(&self.stream), &mut args) }
    }

    /// Dispatch a matvec to the right kernel based on the weight's on-disk
    /// dtype. Output `y` always lands as fp32. K-quants and Q8_0 take the
    /// int8 dp4a path (quantize the activation, then v_dot4_i32_i8 — far
    /// fewer instructions than the fp32-dequant wave64 matvec, so the
    /// kernel is bandwidth-bound instead of issue-bound).
    fn launch_matvec_dispatch(&self, w: &GpuMatvecTensor,
                              x: *mut c_void, y: *mut c_void) -> Result<(), String>
    {
        let in_d  = w.in_dim;
        let out_d = w.out_dim;
        let wp    = w.data.raw_ptr();

        // Repacked Q4_K: contiguous two-plane layout, its own kernel.
        // 256-thread workgroup (4 wavefronts × 2 rows = 8 rows).
        if w.repacked {
            // fp32-consistency path (tests, set_dp4a(false)): dequant the
            // repacked weight → fp16 and run the f16 matvec, so the GPU
            // forward can be checked against the f32 CPU oracle without
            // the int8 activation quantisation. Production is always dp4a.
            if !self.dp4a_enabled {
                let wf16 = self.dequant_weight(w)?;
                let r = if out_d <= 512 {
                    self.launch_matvec_q_kernel(&self.matvec_f16_module, "matvec_f16_f32",
                        wf16.raw_ptr(), x, y, in_d, out_d)
                } else {
                    self.launch_matvec_wave64(&self.matvec_f16_wave64_module,
                        "matvec_f16_wave64_f32", wf16.raw_ptr(), x, y, in_d, out_d)
                };
                self.stream.synchronize()?;     // wf16 is local
                return r;
            }
            self.launch_quantize_q8(x, in_d)?;
            // K-quants: 256-thread / 8-row workgroup. Q8_0 uses the
            // dp4a 64-thread / 2-row footprint on the new two-plane
            // layout (alignment win, not occupancy).
            let (module, kname, grid, kblock): (&Module, &str, u32, u32) = match w.dtype {
                GgmlType::Q5_K => (&self.matvec_q5k_repacked_module, "matvec_q5k_repacked_f32",
                                   (out_d + 7) / 8, 256),
                GgmlType::Q6_K => (&self.matvec_q6k_repacked_module, "matvec_q6k_repacked_f32",
                                   (out_d + 7) / 8, 256),
                // Q8_0: ROWS=1 (grid=out_d) for large out_dim — doubles
                // the wavefront count, which sustains HBM bandwidth that
                // ROWS=2 starves at out_dim≥4096. ROWS=2 stays best for
                // mid-size out_dim (~2048).
                GgmlType::Q8_0 if out_d >= 4096 =>
                    (&self.matvec_q8_0_repacked_module, "matvec_q8_0_repacked_r1_f32",
                     out_d, 64),
                GgmlType::Q8_0 =>
                    (&self.matvec_q8_0_repacked_module, "matvec_q8_0_repacked_f32",
                     (out_d + 1) / 2, 64),
                _              => (&self.matvec_q4k_repacked_module, "matvec_q4k_repacked_f32",
                                   (out_d + 7) / 8, 256),
            };
            let f = module.function(kname)?;
            let mut wa = wp; let mut xa = self.xq8.raw_ptr(); let mut ya = y;
            let mut ia = in_d; let mut oa = out_d;
            let mut args: [*mut c_void; 5] = [
                &mut wa as *mut _ as *mut c_void, &mut xa as *mut _ as *mut c_void,
                &mut ya as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
                &mut oa as *mut _ as *mut c_void];
            return unsafe {
                f.launch((grid, 1, 1), (kblock, 1, 1), 0, Some(&self.stream), &mut args)
            };
        }

        let dp4a = self.dp4a_enabled
            && matches!(w.dtype, GgmlType::Q4_K | GgmlType::Q5_K
                               | GgmlType::Q6_K | GgmlType::Q8_0);
        if dp4a {
            self.launch_quantize_q8(x, in_d)?;
            // Q4_K: 256-thread workgroup (4 independent wavefronts, 8 rows);
            // others: 64-thread, 2 rows per wavefront.
            let (module, kname, rows, block) = match w.dtype {
                GgmlType::Q4_K => (&self.matvec_q4_k_dp4a_module, "matvec_q4_k_dp4a_f32", 8u32, 256u32),
                GgmlType::Q5_K => (&self.matvec_q5_k_dp4a_module, "matvec_q5_k_dp4a_f32", DP4A_ROWBLOCK, 64),
                GgmlType::Q6_K => (&self.matvec_q6_k_dp4a_module, "matvec_q6_k_dp4a_f32", DP4A_ROWBLOCK, 64),
                _              => (&self.matvec_q8_0_dp4a_module, "matvec_q8_0_dp4a_f32", DP4A_ROWBLOCK, 64),
            };
            let f = module.function(kname)?;
            let grid = (out_d + rows - 1) / rows;
            let mut wa = wp; let mut xa = self.xq8.raw_ptr(); let mut ya = y;
            let mut ia = in_d; let mut oa = out_d;
            let mut args: [*mut c_void; 5] = [
                &mut wa as *mut _ as *mut c_void, &mut xa as *mut _ as *mut c_void,
                &mut ya as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
                &mut oa as *mut _ as *mut c_void];
            return unsafe {
                f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args)
            };
        }

        match w.dtype {
            // 256-thread/block fp32 — the wave64 alternative starves the
            // GPU on small `out_dim` matvecs (qwen GDN's ssm_alpha/beta
            // are [hidden, n_heads=48], so wave64 launches only 48
            // wavefronts ≈ 0.8/CU. b256 launches 4× more in flight).
            GgmlType::F32    => self.launch_matvec_f32_b256(wp, x, y, in_d, out_d),
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
            // F16 weights are the tiny GDN projections (ssm_alpha/beta,
            // out_dim = n_v_heads). wave64's one-wavefront-per-row leaves
            // the GPU starved at that size; the block-256 kernel gives 4×
            // the wavefronts. Large F16 matvecs (none in Qwen 3.5) would
            // still prefer wave64.
            GgmlType::F16 if out_d <= 512
                             => self.launch_matvec_q_kernel(&self.matvec_f16_module,
                                    "matvec_f16_f32", wp, x, y, in_d, out_d),
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

    fn launch_rope(&self, x: *mut c_void, n_heads: u32) -> Result<(), String> {
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
        let mut p  = self.d_pos.raw_ptr();
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

    /// Stage the decode position into the device-resident `d_pos`.
    fn set_pos(&self, pos: usize) -> Result<(), String> {
        self.d_pos.copy_from_host(&[pos as u32])
    }

    /// `REINSTINCT_MOE_PROFILE` per-stage timer. `prof_lap` syncs the
    /// stream and charges the elapsed time since the last lap to
    /// `label`'s bucket; `prof_reset` re-marks without charging.
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

    /// Write a K/V row into the cache at the device-resident position
    /// `d_pos` — a graph-capturable replacement for the host-offset
    /// memcpy.
    fn launch_kv_write(&self, src: *mut c_void, cache: *mut c_void, kv_dim: u32)
        -> Result<(), String>
    {
        let f = self.kv_write_module.function("kv_write_f32")?;
        let block: u32 = 256;
        let grid = (kv_dim + block - 1) / block;
        let mut sa = src; let mut ca = cache;
        let mut pp = self.d_pos.raw_ptr(); let mut kd = kv_dim;
        let mut args: [*mut c_void; 4] = [
            &mut sa as *mut _ as *mut c_void, &mut ca as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void, &mut kd as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(block,1,1), 0, Some(&self.stream), &mut args) }
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

    #[allow(clippy::too_many_arguments)]
    fn launch_gdn_recurrent_step_fused(&self,
        q: *mut c_void, k: *mut c_void, v: *mut c_void,
        a: *mut c_void, b: *mut c_void, ssm_a: *mut c_void, dt_bias: *mut c_void,
        state: *mut c_void, out: *mut c_void,
        n_heads: u32, head_dim: u32, n_k_heads: u32) -> Result<(), String>
    {
        let f = self.gdn_recurrent_step_fused_module.function("gdn_recurrent_step_fused_f32")?;
        // Four threads per value-dim column (split kk); grid.y splits
        // head_dim into COLS-wide chunks (COLS = 16, the kernel #define).
        // block = 4*COLS = 64 (one wavefront). LDS = q | k.
        const COLS: u32 = 16;
        let block: u32 = 4 * COLS;
        let grid_y = (head_dim + COLS - 1) / COLS;
        let smem = 2 * head_dim * std::mem::size_of::<f32>() as u32;
        let mut qa = q; let mut ka = k; let mut va = v;
        let mut aa = a; let mut ba = b; let mut sma = ssm_a; let mut dta = dt_bias;
        let mut sa = state; let mut oa = out;
        let mut nh = n_heads; let mut hd = head_dim; let mut nkh = n_k_heads;
        let mut args: [*mut c_void; 12] = [
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
            &mut nkh as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((n_heads, grid_y, 1), (block, 1, 1), smem, Some(&self.stream), &mut args) }
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

    /// Decode attention. FlashDecoding split-K: grid (n_heads, n_splits)
    /// writes a partial (m, l, o) per head, then a merge kernel combines
    /// the splits — keeps every CU busy at depth and shortens the serial
    /// P·V scan. `REINSTINCT_OLD_ATTN` selects the original kernel.
    fn launch_attn_step(&self, q: *mut c_void, k_cache: *mut c_void, v_cache: *mut c_void,
                        out: *mut c_void, scaling: f32) -> Result<(), String>
    {
        let block: u32 = 256;
        let n_heads = self.n_heads as u32;
        let n_kv    = self.n_kv_heads as u32;
        let head_dim = self.head_dim as u32;
        let max_seq = self.max_seq as u32;
        let mut pp = self.d_pos.raw_ptr();   // decode pos, device-resident

        if self.use_old_attn {
            let f = self.attn_step_module.function("attn_step_f32")?;
            // scores[total_len] in LDS — size for the worst case.
            let smem = (head_dim + max_seq + block) * 4;
            let mut qa=q; let mut ka=k_cache; let mut va=v_cache; let mut oa=out;
            let mut nh=n_heads; let mut nkv=n_kv; let mut hd=head_dim;
            let mut sc=scaling;
            let mut args: [*mut c_void; 9] = [
                &mut qa as *mut _ as *mut c_void, &mut ka as *mut _ as *mut c_void,
                &mut va as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void, &mut pp as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void];
            return unsafe { f.launch((n_heads,1,1),(block,1,1), smem,
                                     Some(&self.stream), &mut args) };
        }

        // Split count — a per-model constant (depends only on max_seq) so
        // the grid is fixed ⇒ graph-capture safe; the kernel reads the
        // live position from `d_pos`.
        let n_splits = ((max_seq + 255) / 256).clamp(1, ATTN_MAX_SPLITS);
        let chunk_max = (max_seq + n_splits - 1) / n_splits;
        // LDS: qf[head_dim f32] | scores[chunk_max f32] | tmp[block f32]
        let smem = (head_dim + chunk_max + block) * 4;

        let fp = self.attn_partial_module.function("attn_partial_f32")?;
        let mut qa=q; let mut ka=k_cache; let mut va=v_cache;
        let mut op=self.attn_o_partial.raw_ptr();
        let mut mp=self.attn_m_partial.raw_ptr();
        let mut lp=self.attn_l_partial.raw_ptr();
        let mut nh=n_heads; let mut nkv=n_kv; let mut hd=head_dim;
        let mut sc=scaling; let mut ns=n_splits;
        let mut pargs: [*mut c_void; 12] = [
            &mut qa as *mut _ as *mut c_void, &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void,
            &mut mp as *mut _ as *mut c_void, &mut lp as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut pp as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void, &mut ns as *mut _ as *mut c_void];
        unsafe {
            fp.launch((n_heads, n_splits, 1), (block,1,1), smem, Some(&self.stream), &mut pargs)?;
        }

        let fm = self.attn_merge_module.function("attn_merge_f32")?;
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
        self.launch_rope(self.q_buf.raw_ptr(), self.n_heads as u32)?;
        self.launch_rmsnorm_multihead(self.k_raw.raw_ptr(), weights.attn_k_norm.raw_ptr(),
                                      self.k_norm.raw_ptr(),
                                      self.n_kv_heads as u32, self.head_dim as u32, self.rms_eps)?;
        self.launch_rope(self.k_norm.raw_ptr(), self.n_kv_heads as u32)?;
        // Append this token's K/V into the cache at the device-resident
        // position — a kernel (not a host-offset memcpy) so the forward
        // is graph-capturable.
        let kv_dim = kv_cache.kv_dim as u32;
        self.launch_kv_write(self.k_norm.raw_ptr(), kv_cache.k.raw_ptr(), kv_dim)?;
        self.launch_kv_write(self.v_raw.raw_ptr(),  kv_cache.v.raw_ptr(), kv_dim)?;
        self.launch_attn_step(self.q_buf.raw_ptr(),
                              kv_cache.k.raw_ptr(), kv_cache.v.raw_ptr(),
                              self.attn_concat.raw_ptr(), scaling)?;
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

    /// Single-token FFN — dense SwiGLU or MoE, dispatched on the block's
    /// FFN kind.
    fn step_ffn(&self, input_ptr: *mut c_void, output_ptr: *mut c_void, ffn: &BlockFfn)
        -> Result<(), String>
    {
        match ffn {
            BlockFfn::Dense(d) => self.step_swiglu_ffn(input_ptr, output_ptr, d),
            BlockFfn::Moe(m)   => self.step_moe_ffn(input_ptr, output_ptr, m),
        }
    }

    /// Qwen MoE FFN for one token: router → top-k routed experts (SwiGLU)
    /// → sigmoid-gated shared expert. Writes `output_ptr` [hidden].
    fn step_moe_ffn(&self, input_ptr: *mut c_void, output_ptr: *mut c_void, w: &GpuMoeFfn)
        -> Result<(), String>
    {
        let moe = self.moe.as_ref().expect("step_moe_ffn on a non-MoE model");
        let h  = self.hidden as u32;
        let ff = moe.expert_ff as u32;
        let shff = moe.shexp_ff as u32;
        let n_used = moe.n_used as u32;

        self.prof_lap("attn+norm");
        // --- Router: logits → top-k expert ids + renormalised weights ---
        self.launch_matvec_dispatch(&w.gate_inp, input_ptr, moe.logits.raw_ptr())?;
        self.prof_lap("router_matvec");
        self.launch_moe_topk(moe, 1)?;
        self.prof_lap("router_topk");

        // --- Routed experts --- shared int8 activation, quantised once.
        // (Routed result lands in e_out; the combine into `output` is
        //  deferred until the shared expert has read `input` — callers
        //  pass input == output, so the combine would clobber it.)
        self.launch_quantize_q8(input_ptr, h)?;
        self.prof_lap("moe_quant_in");
        // Fused gate+up+SwiGLU when both expert slabs are Q4_K (one
        // launch vs three); otherwise the unfused path.
        if w.gate_exps.dtype == GgmlType::Q4_K && w.up_exps.dtype == GgmlType::Q4_K {
            self.launch_moe_gate_up_swiglu(moe, &w.gate_exps, &w.up_exps,
                self.xq8.raw_ptr(), moe.e_gate.raw_ptr(), h, ff, 1, 0, 0)?;
        } else {
            self.launch_moe_expert_matvec(moe, &w.gate_exps, self.xq8.raw_ptr(),
                                          moe.e_gate.raw_ptr(), h, ff, 1, 0, 0)?;
            self.launch_moe_expert_matvec(moe, &w.up_exps, self.xq8.raw_ptr(),
                                          moe.e_up.raw_ptr(), h, ff, 1, 0, 0)?;
            self.launch_swiglu(moe.e_gate.raw_ptr(), moe.e_up.raw_ptr(),
                               moe.e_gate.raw_ptr(), n_used * ff)?;
        }
        self.prof_lap("moe_gate_up");
        // down: each expert has its own activation — quantise the batch.
        self.launch_quantize_q8_into(moe.e_gate.raw_ptr(), moe.xq8_exp.raw_ptr(),
                                     ff, n_used)?;
        self.launch_moe_down(moe, &w.down_exps, moe.xq8_exp.raw_ptr(),
                             moe.e_out.raw_ptr(), ff, h, 1, 0, ff / 32)?;
        self.prof_lap("moe_down");

        // --- Shared expert --- runs every token, scaled by a sigmoid gate.
        // Reads `input` — must finish before the combine writes `output`.
        self.launch_matvec_dispatch(&w.gate_shexp, input_ptr, moe.sh_gate.raw_ptr())?;
        self.launch_matvec_dispatch(&w.up_shexp,   input_ptr, moe.sh_up.raw_ptr())?;
        self.launch_swiglu(moe.sh_gate.raw_ptr(), moe.sh_up.raw_ptr(),
                           moe.sh_gate.raw_ptr(), shff)?;
        self.launch_matvec_dispatch(&w.down_shexp, moe.sh_gate.raw_ptr(),
                                    moe.sh_out.raw_ptr())?;
        self.launch_moe_shexp_gate(moe, moe.sh_out.raw_ptr(), input_ptr,
                                   w.gate_inp_shexp.raw_ptr(), 1)?;
        self.prof_lap("moe_shared");

        // `input` fully consumed — combine routed experts, add the shared.
        self.launch_moe_combine(moe, moe.e_out.raw_ptr(), output_ptr, 1)?;
        self.launch_add_inplace(output_ptr, moe.sh_out.raw_ptr(), h)?;
        self.prof_lap("moe_combine");
        Ok(())
    }

    /// Batched MoE FFN — processes `n` (≤ MOE_PREFILL_CHUNK) prefill rows
    /// in one set of launches. Routed-expert matvecs batch over tokens via
    /// grid.z; the router and shared expert (shared weights) go through
    /// rocBLAS GEMMs. Callers pass distinct `input` / `output` buffers.
    fn step_moe_ffn_batched(&self, input_ptr: *mut c_void, output_ptr: *mut c_void,
                            w: &GpuMoeFfn, n: usize) -> Result<(), String>
    {
        debug_assert!(n <= MOE_PREFILL_CHUNK, "MoE prefill batch exceeds chunk");
        let moe = self.moe.as_ref().expect("step_moe_ffn_batched on a non-MoE model");
        let h  = self.hidden as u32;
        let ff = moe.expert_ff as u32;
        let shff = moe.shexp_ff as u32;
        let n_used = moe.n_used as u32;
        let nt = n as u32;

        // --- Router: GEMM over all rows → logits, then per-token top-k ---
        self.bmm(&w.gate_inp, input_ptr, n, moe.logits.raw_ptr())?;
        self.launch_moe_topk(moe, nt)?;

        // Grouped-expert GEMM groundwork (M1): expert-routing sort. Not
        // yet consumed — the matvec path below still runs. Gated for now.
        if std::env::var_os("REINSTINCT_MOE_SORT_CHECK").is_some() {
            self.launch_moe_sort(moe, nt)?;
            self.verify_moe_sort(moe, nt)?;
        }

        // --- Routed experts --- one int8 activation per token, then the
        // expert matvecs batch over tokens (grid.z). gate/up share the
        // token activation across the 8 experts (slot stride 0); down has
        // a distinct activation per (token, expert).
        self.launch_quantize_q8_into(input_ptr, moe.xq8_in.raw_ptr(), h, nt)?;
        // Grouped-expert GEMM: sort tokens by expert, gather, one tiled
        // GEMM per expert (weight read once per expert, not once per
        // routed token). The whole FFN — gate, up, down — runs in
        // expert-sorted order; only a single scatter at the end returns
        // to [token, slot] order. Default-on for MoE; opt out with
        // `REINSTINCT_MOE_NO_GROUPED=1`.
        let grouped = std::env::var_os("REINSTINCT_MOE_NO_GROUPED").is_none()
            && w.gate_exps.dtype == GgmlType::Q4_K && w.gate_exps.repacked
            && w.up_exps.dtype == GgmlType::Q4_K && w.up_exps.repacked
            && matches!(w.down_exps.dtype, GgmlType::Q5_K | GgmlType::Q6_K)
            && w.down_exps.repacked;
        if grouped {
            self.launch_moe_sort(moe, nt)?;
            self.launch_moe_gather_xq(moe, h / 32, nt)?;
            self.launch_moe_grouped_gemm(moe, &w.gate_exps, moe.g_in.raw_ptr(),
                                         moe.e_gate.raw_ptr(), h, ff, nt)?;
            self.launch_moe_grouped_gemm(moe, &w.up_exps, moe.g_in.raw_ptr(),
                                         moe.e_up.raw_ptr(), h, ff, nt)?;
            // gate/up/swiglu/quantize all stay in expert-sorted order.
            self.launch_swiglu(moe.e_gate.raw_ptr(), moe.e_up.raw_ptr(),
                               moe.e_gate.raw_ptr(), nt * n_used * ff)?;
            self.launch_quantize_q8_into(moe.e_gate.raw_ptr(), moe.xq8_exp.raw_ptr(),
                                         ff, nt * n_used)?;
            self.launch_moe_grouped_gemm(moe, &w.down_exps, moe.xq8_exp.raw_ptr(),
                                         moe.g_out.raw_ptr(), ff, h, nt)?;
            // single scatter back to [token, slot] order for the combine.
            self.launch_moe_scatter_rows(moe, moe.g_out.raw_ptr(),
                                         moe.e_out.raw_ptr(), h, nt)?;
        } else {
            self.launch_moe_expert_matvec(moe, &w.gate_exps, moe.xq8_in.raw_ptr(),
                                          moe.e_gate.raw_ptr(), h, ff, nt, h / 32, 0)?;
            self.launch_moe_expert_matvec(moe, &w.up_exps, moe.xq8_in.raw_ptr(),
                                          moe.e_up.raw_ptr(), h, ff, nt, h / 32, 0)?;
            self.launch_swiglu(moe.e_gate.raw_ptr(), moe.e_up.raw_ptr(),
                               moe.e_gate.raw_ptr(), nt * n_used * ff)?;
            self.launch_quantize_q8_into(moe.e_gate.raw_ptr(), moe.xq8_exp.raw_ptr(),
                                         ff, nt * n_used)?;
            self.launch_moe_expert_matvec(moe, &w.down_exps, moe.xq8_exp.raw_ptr(),
                                          moe.e_out.raw_ptr(), ff, h, nt,
                                          n_used * (ff / 32), ff / 32)?;
        }

        // --- Shared expert --- dense, shared weights → batched GEMMs ---
        self.bmm(&w.gate_shexp, input_ptr, n, moe.sh_gate.raw_ptr())?;
        self.bmm(&w.up_shexp,   input_ptr, n, moe.sh_up.raw_ptr())?;
        self.launch_swiglu(moe.sh_gate.raw_ptr(), moe.sh_up.raw_ptr(),
                           moe.sh_gate.raw_ptr(), nt * shff)?;
        self.bmm(&w.down_shexp, moe.sh_gate.raw_ptr(), n, moe.sh_out.raw_ptr())?;
        self.launch_moe_shexp_gate(moe, moe.sh_out.raw_ptr(), input_ptr,
                                   w.gate_inp_shexp.raw_ptr(), nt)?;

        // Combine routed experts into `output`, add the shared expert.
        self.launch_moe_combine(moe, moe.e_out.raw_ptr(), output_ptr, nt)?;
        self.launch_add_inplace(output_ptr, moe.sh_out.raw_ptr(), nt * h)?;
        Ok(())
    }

    /// Top-k router selection for `n_tok` tokens (one workgroup per token).
    fn launch_moe_topk(&self, moe: &MoeRuntime, n_tok: u32) -> Result<(), String> {
        let f = moe.m_topk.function("moe_topk_f32")?;
        let mut la = moe.logits.raw_ptr();
        let mut ne = moe.n_expert as i32;
        let mut nu = moe.n_used as i32;
        let mut ida = moe.ids.raw_ptr();
        let mut wa  = moe.weights.raw_ptr();
        let mut args: [*mut c_void; 5] = [
            &mut la as *mut _ as *mut c_void, &mut ne as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void, &mut ida as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void];
        let smem = moe.n_expert as u32 * 4;
        unsafe { f.launch((n_tok,1,1),(128,1,1), smem, Some(&self.stream), &mut args) }
    }

    /// Counting-sort the `n_tok * n_used` routing entries by expert id
    /// into `moe.sort_perm`, with `sort_eoff` (entry offsets) and
    /// `sort_toff` (GEMM-tile offsets). Foundation of the grouped-expert
    /// GEMM prefill path — see kernels/moe_expert_sort.cpp.
    fn launch_moe_sort(&self, moe: &MoeRuntime, n_tok: u32) -> Result<(), String> {
        let ne = moe.n_expert as u32;
        let n_entries = n_tok * moe.n_used as u32;
        let zero = |buf: *mut c_void, n: u32| -> Result<(), String> {
            let f = moe.m_expert_sort.function("moe_sort_zero")?;
            let mut a0 = buf; let mut a1 = n;
            let mut args: [*mut c_void; 2] = [
                &mut a0 as *mut _ as *mut c_void, &mut a1 as *mut _ as *mut c_void];
            unsafe { f.launch(((n + 255) / 256, 1, 1), (256, 1, 1), 0,
                              Some(&self.stream), &mut args) }
        };
        zero(moe.sort_count.raw_ptr(), ne)?;
        {
            let f = moe.m_expert_sort.function("moe_sort_histogram")?;
            let mut a0 = moe.ids.raw_ptr(); let mut a1 = moe.sort_count.raw_ptr();
            let mut a2 = n_entries; let mut a3 = ne;
            let mut args: [*mut c_void; 4] = [
                &mut a0 as *mut _ as *mut c_void, &mut a1 as *mut _ as *mut c_void,
                &mut a2 as *mut _ as *mut c_void, &mut a3 as *mut _ as *mut c_void];
            unsafe { f.launch(((n_entries + 255) / 256, 1, 1), (256, 1, 1), 0,
                              Some(&self.stream), &mut args)?; }
        }
        {
            let f = moe.m_expert_sort.function("moe_sort_scan")?;
            let mut a0 = moe.sort_count.raw_ptr(); let mut a1 = moe.sort_eoff.raw_ptr();
            let mut a2 = moe.sort_toff.raw_ptr(); let mut a3 = ne; let mut a4 = MOE_GEMM_BN;
            let mut args: [*mut c_void; 5] = [
                &mut a0 as *mut _ as *mut c_void, &mut a1 as *mut _ as *mut c_void,
                &mut a2 as *mut _ as *mut c_void, &mut a3 as *mut _ as *mut c_void,
                &mut a4 as *mut _ as *mut c_void];
            unsafe { f.launch((1, 1, 1), (64, 1, 1), 0, Some(&self.stream), &mut args)?; }
        }
        zero(moe.sort_cursor.raw_ptr(), ne)?;
        {
            let f = moe.m_expert_sort.function("moe_sort_scatter")?;
            let mut a0 = moe.ids.raw_ptr(); let mut a1 = moe.sort_eoff.raw_ptr();
            let mut a2 = moe.sort_cursor.raw_ptr(); let mut a3 = moe.sort_perm.raw_ptr();
            let mut a4 = n_entries; let mut a5 = ne;
            let mut args: [*mut c_void; 6] = [
                &mut a0 as *mut _ as *mut c_void, &mut a1 as *mut _ as *mut c_void,
                &mut a2 as *mut _ as *mut c_void, &mut a3 as *mut _ as *mut c_void,
                &mut a4 as *mut _ as *mut c_void, &mut a5 as *mut _ as *mut c_void];
            unsafe { f.launch(((n_entries + 255) / 256, 1, 1), (256, 1, 1), 0,
                              Some(&self.stream), &mut args)?; }
        }
        Ok(())
    }

    /// Host-side check of `launch_moe_sort` — syncs, downloads the sort
    /// buffers, asserts the permutation is a valid expert grouping.
    /// Gated by `REINSTINCT_MOE_SORT_CHECK`.
    fn verify_moe_sort(&self, moe: &MoeRuntime, n_tok: u32) -> Result<(), String> {
        self.stream.synchronize()?;
        let ne = moe.n_expert;
        let n_entries = n_tok as usize * moe.n_used;
        let mut ids  = vec![0i32; n_entries];
        let mut perm = vec![0i32; n_entries];
        let mut eoff = vec![0i32; ne + 1];
        let mut toff = vec![0i32; ne + 1];
        moe.ids.copy_range_to_host(&mut ids, 0)?;
        moe.sort_perm.copy_range_to_host(&mut perm, 0)?;
        moe.sort_eoff.copy_to_host(&mut eoff)?;
        moe.sort_toff.copy_to_host(&mut toff)?;

        let mut count = vec![0i32; ne];
        for &e in &ids { if e >= 0 && (e as usize) < ne { count[e as usize] += 1; } }
        let mut fail = String::new();
        let mut eacc = 0i32; let mut tacc = 0i32;
        for e in 0..ne {
            if eoff[e] != eacc { fail = format!("eoff[{e}]={} != {eacc}", eoff[e]); break; }
            if toff[e] != tacc { fail = format!("toff[{e}]={} != {tacc}", toff[e]); break; }
            eacc += count[e];
            tacc += (count[e] + MOE_GEMM_BN as i32 - 1) / MOE_GEMM_BN as i32;
        }
        if fail.is_empty() && eoff[ne] != eacc {
            fail = format!("eoff[ne]={} != n_entries-ish {eacc}", eoff[ne]);
        }
        let mut seen = vec![false; n_entries];
        if fail.is_empty() {
            for e in 0..ne {
                for p in eoff[e]..eoff[e + 1] {
                    let entry = perm[p as usize];
                    if entry < 0 || (entry as usize) >= n_entries {
                        fail = format!("perm[{p}]={entry} out of range"); break;
                    }
                    if seen[entry as usize] { fail = format!("perm dup {entry}"); break; }
                    seen[entry as usize] = true;
                    if ids[entry as usize] != e as i32 {
                        fail = format!("perm[{p}]→entry {entry} expert {} != {e}",
                                       ids[entry as usize]);
                        break;
                    }
                }
                if !fail.is_empty() { break; }
            }
        }
        if fail.is_empty() {
            eprintln!("[moe-sort-check] OK  n_tok={n_tok} entries={n_entries} \
                       tiles={}", toff[ne]);
        } else {
            eprintln!("[moe-sort-check] FAIL: {fail}");
        }
        Ok(())
    }

    /// Gather gate/up activations into expert-sorted order:
    /// `g_in[p] = xq8_in[perm[p] / n_used]`. `nsub = hidden/32`.
    fn launch_moe_gather_xq(&self, moe: &MoeRuntime, nsub: u32, n_tok: u32)
        -> Result<(), String>
    {
        let f = moe.m_expert_sort.function("moe_gather_xq")?;
        let n_entries = n_tok * moe.n_used as u32;
        let mut a0 = moe.xq8_in.raw_ptr(); let mut a1 = moe.sort_perm.raw_ptr();
        let mut a2 = moe.g_in.raw_ptr(); let mut a3 = nsub;
        let mut a4 = moe.n_used as u32; let mut a5 = n_entries;
        let mut args: [*mut c_void; 6] = [
            &mut a0 as *mut _ as *mut c_void, &mut a1 as *mut _ as *mut c_void,
            &mut a2 as *mut _ as *mut c_void, &mut a3 as *mut _ as *mut c_void,
            &mut a4 as *mut _ as *mut c_void, &mut a5 as *mut _ as *mut c_void];
        unsafe { f.launch(((nsub + 255) / 256, n_entries, 1), (256, 1, 1), 0,
                          Some(&self.stream), &mut args) }
    }

    /// Scatter sorted-order rows back to entry order: `dst[perm[p]] = src[p]`.
    fn launch_moe_scatter_rows(&self, moe: &MoeRuntime, src: *mut c_void,
                               dst: *mut c_void, dim: u32, n_tok: u32)
        -> Result<(), String>
    {
        let f = moe.m_expert_sort.function("moe_scatter_rows")?;
        let n_entries = n_tok * moe.n_used as u32;
        let mut a0 = src; let mut a1 = moe.sort_perm.raw_ptr(); let mut a2 = dst;
        let mut a3 = dim; let mut a4 = n_entries;
        let mut args: [*mut c_void; 5] = [
            &mut a0 as *mut _ as *mut c_void, &mut a1 as *mut _ as *mut c_void,
            &mut a2 as *mut _ as *mut c_void, &mut a3 as *mut _ as *mut c_void,
            &mut a4 as *mut _ as *mut c_void];
        unsafe { f.launch(((dim + 255) / 256, n_entries, 1), (256, 1, 1), 0,
                          Some(&self.stream), &mut args) }
    }

    /// Grouped-expert MMQ GEMM — repacked Q4_K/Q5_K/Q6_K. `xq` / `y`
    /// are in expert-sorted order; one launch covers all experts (each
    /// workgroup maps to its expert via `sort_toff`).
    #[allow(clippy::too_many_arguments)]
    fn launch_moe_grouped_gemm(&self, moe: &MoeRuntime, et: &GpuExpertTensor,
                               xq: *mut c_void, y: *mut c_void,
                               in_dim: u32, out_dim: u32, n_tok: u32)
        -> Result<(), String>
    {
        let (module, kname) = match et.dtype {
            GgmlType::Q5_K => (&moe.m_grouped_q5k, "mmq_gemm_q5k_grouped_f32"),
            GgmlType::Q6_K => (&moe.m_grouped_q6k, "mmq_gemm_q6k_grouped_f32"),
            _              => (&moe.m_grouped_q4k, "mmq_gemm_q4k_grouped_f32"),
        };
        let f = module.function(kname)?;
        let n_entries = n_tok * moe.n_used as u32;
        let tile_ub = (n_entries + MOE_GEMM_BN - 1) / MOE_GEMM_BN + moe.n_expert as u32;
        let mut a0 = et.data.raw_ptr(); let mut a1 = et.bytes_per_expert as u32;
        let mut a2 = moe.sort_eoff.raw_ptr(); let mut a3 = moe.sort_toff.raw_ptr();
        let mut a4 = moe.n_expert as u32; let mut a5 = xq; let mut a6 = y;
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

    /// Routed-expert matvec — grid (out_dim/8, n_used, n_tok). For decode
    /// pass `n_tok = 1`. `xq` is indexed `tok*xq_tok_stride +
    /// slot*xq_slot_stride` (BlockQ8 units): gate/up share one activation
    /// per token (slot stride 0), down has one per (token, expert).
    #[allow(clippy::too_many_arguments)]
    fn launch_moe_expert_matvec(&self, moe: &MoeRuntime, et: &GpuExpertTensor,
                                xq8: *mut c_void, y: *mut c_void,
                                in_dim: u32, out_dim: u32, n_tok: u32,
                                xq_tok_stride: u32, xq_slot_stride: u32)
        -> Result<(), String>
    {
        let (module, kname) = match et.dtype {
            GgmlType::Q4_K => (&moe.m_mv_q4k, "moe_matvec_q4k_repacked_f32"),
            GgmlType::Q5_K => (&moe.m_mv_q5k, "moe_matvec_q5k_repacked_f32"),
            GgmlType::Q6_K => (&moe.m_mv_q6k, "moe_matvec_q6k_repacked_f32"),
            other => return Err(format!("moe expert matvec: dtype {other:?}")),
        };
        let f = module.function(kname)?;
        let grid_x = (out_dim + 7) / 8;
        let mut sa = et.data.raw_ptr(); let mut ida = moe.ids.raw_ptr();
        let mut xa = xq8; let mut ya = y;
        let mut ia = in_dim; let mut oa = out_dim;
        let mut bpe = et.bytes_per_expert as u32;
        let mut tst = xq_tok_stride; let mut sst = xq_slot_stride;
        let mut nu = moe.n_used as u32;
        let mut args: [*mut c_void; 10] = [
            &mut sa as *mut _ as *mut c_void, &mut ida as *mut _ as *mut c_void,
            &mut xa as *mut _ as *mut c_void, &mut ya as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut bpe as *mut _ as *mut c_void, &mut tst as *mut _ as *mut c_void,
            &mut sst as *mut _ as *mut c_void, &mut nu as *mut _ as *mut c_void];
        unsafe { f.launch((grid_x, moe.n_used as u32, n_tok), (256,1,1), 0,
                          Some(&self.stream), &mut args) }
    }

    /// Expert DOWN matvec — the row-packed kernel (all 64 lanes busy at
    /// the down projection's small in_dim) for Q5_K/Q6_K experts; the
    /// generic expert matvec for any other dtype.
    #[allow(clippy::too_many_arguments)]
    fn launch_moe_down(&self, moe: &MoeRuntime, et: &GpuExpertTensor,
                       xq8: *mut c_void, y: *mut c_void,
                       in_dim: u32, out_dim: u32, n_tok: u32,
                       xq_tok_stride: u32, xq_slot_stride: u32) -> Result<(), String>
    {
        let (module, kname) = match et.dtype {
            GgmlType::Q5_K => (&moe.m_down_q5k, "moe_matvec_q5k_down_f32"),
            GgmlType::Q6_K => (&moe.m_down_q6k, "moe_matvec_q6k_down_f32"),
            _ => return self.launch_moe_expert_matvec(moe, et, xq8, y, in_dim, out_dim,
                                                      n_tok, xq_tok_stride, xq_slot_stride),
        };
        let f = module.function(kname)?;
        let rpb = (256 / (in_dim / 32)).max(1);
        let grid_x = (out_dim + rpb - 1) / rpb;
        let mut sa = et.data.raw_ptr(); let mut ida = moe.ids.raw_ptr();
        let mut xa = xq8; let mut ya = y;
        let mut ia = in_dim; let mut oa = out_dim;
        let mut bpe = et.bytes_per_expert as u32;
        let mut tst = xq_tok_stride; let mut sst = xq_slot_stride;
        let mut nu = moe.n_used as u32;
        let mut args: [*mut c_void; 10] = [
            &mut sa as *mut _ as *mut c_void, &mut ida as *mut _ as *mut c_void,
            &mut xa as *mut _ as *mut c_void, &mut ya as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut bpe as *mut _ as *mut c_void, &mut tst as *mut _ as *mut c_void,
            &mut sst as *mut _ as *mut c_void, &mut nu as *mut _ as *mut c_void];
        unsafe { f.launch((grid_x, moe.n_used as u32, n_tok), (256,1,1), 0,
                          Some(&self.stream), &mut args) }
    }

    /// Fused gate+up expert matvec + SwiGLU (Q4_K experts) — one launch
    /// in place of gate matvec + up matvec + swiglu. `y` receives the
    /// SwiGLU activation `[n_tok, n_used, out_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn launch_moe_gate_up_swiglu(&self, moe: &MoeRuntime,
                                 gate_et: &GpuExpertTensor, up_et: &GpuExpertTensor,
                                 xq8: *mut c_void, y: *mut c_void,
                                 in_dim: u32, out_dim: u32, n_tok: u32,
                                 xq_tok_stride: u32, xq_slot_stride: u32)
        -> Result<(), String>
    {
        let f = moe.m_gate_up_swiglu_q4k.function("moe_gate_up_swiglu_q4k_repacked_f32")?;
        let grid_x = (out_dim + 7) / 8;
        let mut ga = gate_et.data.raw_ptr(); let mut ua = up_et.data.raw_ptr();
        let mut ida = moe.ids.raw_ptr(); let mut xa = xq8; let mut ya = y;
        let mut ia = in_dim; let mut oa = out_dim;
        let mut bpe = gate_et.bytes_per_expert as u32;
        let mut tst = xq_tok_stride; let mut sst = xq_slot_stride;
        let mut nu = moe.n_used as u32;
        let mut args: [*mut c_void; 11] = [
            &mut ga as *mut _ as *mut c_void, &mut ua as *mut _ as *mut c_void,
            &mut ida as *mut _ as *mut c_void, &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void, &mut bpe as *mut _ as *mut c_void,
            &mut tst as *mut _ as *mut c_void, &mut sst as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void];
        unsafe { f.launch((grid_x, moe.n_used as u32, n_tok), (256,1,1), 0,
                          Some(&self.stream), &mut args) }
    }

    fn launch_moe_combine(&self, moe: &MoeRuntime, experts: *mut c_void,
                          out: *mut c_void, n_tok: u32) -> Result<(), String>
    {
        let f = moe.m_combine.function("moe_combine_f32")?;
        let block: u32 = 256;
        let h = self.hidden as u32;
        let grid = (h + block - 1) / block;
        let mut ea = experts; let mut ida = moe.ids.raw_ptr();
        let mut wa = moe.weights.raw_ptr(); let mut sa = moe.ones.raw_ptr();
        let mut oa = out; let mut ha = h; let mut nu = moe.n_used as u32;
        let mut args: [*mut c_void; 7] = [
            &mut ea as *mut _ as *mut c_void, &mut ida as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void, &mut sa as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void, &mut ha as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void];
        unsafe { f.launch((grid,n_tok,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_moe_shexp_gate(&self, moe: &MoeRuntime, sh_out: *mut c_void,
                             hidden: *mut c_void, gate_w: *mut c_void, n_tok: u32)
        -> Result<(), String>
    {
        let f = moe.m_shexp_gate.function("moe_shexp_gate_f32")?;
        let block: u32 = 256;
        let mut sa = sh_out; let mut ha = hidden; let mut ga = gate_w;
        let mut na = self.hidden as u32;
        let mut args: [*mut c_void; 4] = [
            &mut sa as *mut _ as *mut c_void, &mut ha as *mut _ as *mut c_void,
            &mut ga as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void];
        unsafe { f.launch((n_tok,1,1),(block,1,1), block * 4, Some(&self.stream), &mut args) }
    }

    /// Quantize `n_vec` activation vectors of `in_dim` into `out` (BlockQ8).
    fn launch_quantize_q8_into(&self, x: *mut c_void, out: *mut c_void,
                               in_dim: u32, n_vec: u32) -> Result<(), String> {
        let f = self.quantize_q8_module.function("quantize_q8_f32")?;
        let mut xa = x; let mut oa = out; let mut ia = in_dim;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void];
        unsafe { f.launch(((in_dim + 255) / 256, n_vec, 1), (256,1,1),
                          0, Some(&self.stream), &mut args) }
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
        self.step_ffn(scratch, scratch, &weights.ffn)?;
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
        self.step_ffn(scratch, scratch, &weights.ffn)?;
        self.launch_add_inplace(hidden_io, scratch, h)?;
        Ok(())
    }

    /// MTP-head draft forward — predicts the token *after* `embed_next`
    /// using the in-GGUF "nextn" head (DeepSeek-V3 style). Inputs:
    ///   `prev_hidden` — the main model's final block output (pre
    ///                   output-norm), i.e. `hidden_a` after a decode;
    ///   `embed_next`  — the token that follows `prev_hidden`'s position
    ///                   in the sequence (the main model's prediction).
    ///
    ///   emb          = embed_lookup(embed_next)
    ///   concat[0..h] = rmsnorm(emb,         enorm)
    ///   concat[h..2h]= rmsnorm(prev_hidden, hnorm)
    ///   hid          = eh_proj · concat                  ([2h] → [h])
    ///   hid          = full_attn_block(hid)              attn(mtp_kv)+ffn
    ///   normed       = rmsnorm(hid, shared_head_norm)
    ///   logits       = lm_head · normed                  (tied output)
    ///
    /// The MTP block's attention reads/writes its own KV cache `mtp_kv`
    /// at the device-resident `d_pos` — the caller must `set_pos` first.
    /// Logits land in `self.logits`; the caller syncs and reads them.
    /// Does not advance any sequence position.
    fn mtp_draft_forward(&self,
        mtp: &GpuMtpHead, prev_hidden: *mut c_void, embed_next: u32,
        mtp_kv: &mut GpuKvCache,
    ) -> Result<(), String>
    {
        let h  = self.hidden as u32;
        let sp = self.mtp_scratch.raw_ptr() as *mut f32;
        let concat  = sp;                                       // [0  .. 2h]
        let mtp_hid = unsafe { sp.add(2 * self.hidden) };       // [2h .. 3h]
        let scr     = self.hidden_b.raw_ptr() as *mut f32;      // emb / scratch / normed

        self.launch_embed_lookup_dispatch(
            &self.token_embd, scr as *mut c_void, embed_next)?;
        self.launch_rmsnorm(scr as *mut c_void, mtp.enorm.raw_ptr(),
                            concat as *mut c_void, h, self.rms_eps)?;
        self.launch_rmsnorm(prev_hidden, mtp.hnorm.raw_ptr(),
                            unsafe { concat.add(self.hidden) } as *mut c_void,
                            h, self.rms_eps)?;
        self.launch_matvec_dispatch(&mtp.eh_proj, concat as *mut c_void,
                                    mtp_hid as *mut c_void)?;
        self.step_full_attention_block_dev(mtp_hid as *mut c_void,
                                           scr as *mut c_void, &mtp.block, mtp_kv)?;
        self.launch_rmsnorm(mtp_hid as *mut c_void, mtp.shared_head_norm.raw_ptr(),
                            scr as *mut c_void, h, self.rms_eps)?;
        self.launch_matvec_dispatch(self.output_proj_tensor(),
                                    scr as *mut c_void, self.logits.raw_ptr())?;
        Ok(())
    }

    /// QMTP-1 diagnostic — decode `n_tokens` greedily from the current
    /// `state` with the main model, and at each step run the MTP head
    /// alongside. Returns `(accept_rate, matches, total)` where a "match"
    /// is the MTP head's argmax equalling the token the main model
    /// actually decodes for that position (the K=1 spec-decode accept
    /// rate). NOTE: the MTP block's KV cache is built cold here (it does
    /// not mirror the prefill), so early steps under-report — QMTP-2
    /// warms the cache properly.
    pub fn mtp_accept_probe(&self, first_token: u32, n_tokens: usize,
                            state: &mut Qwen35GpuState)
        -> Result<(f32, usize, usize), String>
    {
        if self.mtp.is_empty() {
            return Err("model has no MTP (nextn) head".into());
        }
        let mut mtp_kv = GpuKvCache::new(
            self.max_seq, self.n_kv_heads, self.head_dim)?;
        let mut logits_host = vec![0.0f32; self.vocab];
        let mut tok = first_token;
        let mut pending: Option<u32> = None;
        let (mut matches, mut total) = (0usize, 0usize);

        for i in 0..n_tokens {
            let main_logits = self.forward_token(tok, state)?;
            let next = crate::sampling::argmax(&main_logits);
            if let Some(d) = pending.take() {
                total += 1;
                if d == next { matches += 1; }
            }
            // MTP block runs as its own fresh sequence: position = step i.
            self.set_pos(i)?;
            self.mtp_draft_forward(&self.mtp[0], self.hidden_a.raw_ptr(),
                                   next, &mut mtp_kv)?;
            self.stream.synchronize()?;
            self.logits.copy_to_host(&mut logits_host)?;
            pending = Some(crate::sampling::argmax(&logits_host));
            tok = next;
        }
        let rate = if total == 0 { 0.0 } else { matches as f32 / total as f32 };
        Ok((rate, matches, total))
    }

    /// Device pointer to row `r` of the hidden states stashed by the
    /// most recent `forward_tokens_verify`.
    fn verify_hidden_row(&self, r: usize) -> *mut c_void {
        unsafe { (self.verify_hidden.raw_ptr() as *mut f32)
            .add(r * self.hidden) as *mut c_void }
    }

    /// Chain the MTP head `k` times to produce `k` speculative drafts.
    /// Link 0 drafts from `(prev_hidden, first_embed)`; each later link
    /// feeds the previous link's block-hidden and drafted token. The MTP
    /// block's KV cache advances by `k` (drafts occupy MTP positions
    /// `mtp_pos .. mtp_pos+k`). Returns the `k` drafted token ids.
    fn mtp_draft_chain(&self, mtp: &GpuMtpHead, prev_hidden: *mut c_void,
                       first_embed: u32, mtp_kv: &mut GpuKvCache,
                       k: usize, mtp_pos: usize) -> Result<Vec<u32>, String>
    {
        let h = self.hidden;
        let mut drafts = Vec::with_capacity(k);
        let mut logits_host = vec![0.0f32; self.vocab];
        let mut embed = first_embed;
        for i in 0..k {
            let prev = if i == 0 { prev_hidden } else { self.mtp_chain_hid.raw_ptr() };
            self.set_pos(mtp_pos + i)?;
            self.mtp_draft_forward(mtp, prev, embed, mtp_kv)?;
            if i + 1 < k {
                // Preserve this link's block-hidden (mtp_scratch[2h..3h])
                // as the next link's prev_hidden.
                self.mtp_chain_hid.copy_range_from_device_async(
                    &self.mtp_scratch, 2 * h, 0, h, &self.stream)?;
            }
            self.stream.synchronize()?;
            self.logits.copy_to_host(&mut logits_host)?;
            embed = crate::sampling::argmax(&logits_host);
            drafts.push(embed);
        }
        Ok(drafts)
    }

    /// QMTP-3 — MTP speculative-decode generation loop.
    ///
    /// `state` must be prefilled; `first_token` is the first token to
    /// emit (typically the argmax of the prefill logits). It is
    /// committed with a normal decode to bootstrap, then each round:
    /// chain `k` MTP drafts, batch-verify `[t, d1..dk]` in one forward,
    /// and accept all-or-nothing — on any mismatch the round is rolled
    /// back via `snapshot` and the certain token `t` re-decoded.
    ///
    /// Returns the generated tokens (stopping at `eos` or `max_tokens`)
    /// and per-call stats.
    pub fn mtp_spec_generate(&self,
        state: &mut Qwen35GpuState,
        mtp_kv: &mut GpuKvCache,
        snapshot: &mut Qwen35Snapshot,
        first_token: u32,
        eos: u32, max_tokens: usize, k: usize,
    ) -> Result<(Vec<u32>, QwenSpecStats), String>
    {
        if self.mtp.is_empty() {
            return Err("model has no MTP (nextn) head".into());
        }
        assert!(k >= 1 && k + 1 <= VERIFY_MAX_TOKENS, "mtp_spec_generate: bad k");
        let mtp = &self.mtp[0];
        let mut stats = QwenSpecStats::default();
        let mut mtp_pos = 0usize;

        // Bootstrap: commit `first_token` with a normal decode so the
        // loop starts with its logits + hidden state.
        let mut verify_logits = self.forward_token(first_token, state)?;
        let mut prev_hidden = self.hidden_a.raw_ptr();
        let mut generated: Vec<u32> = vec![first_token];
        if first_token == eos {
            stats.hit_eos = true;
            return Ok((generated, stats));
        }

        while generated.len() < max_tokens {
            let t = crate::sampling::argmax(&verify_logits);
            snapshot.save(state)?;

            // Chain k MTP drafts, then batch-verify [t, d1..dk].
            let drafts = self.mtp_draft_chain(mtp, prev_hidden, t, mtp_kv, k, mtp_pos)?;
            mtp_pos += k;
            let mut batch = Vec::with_capacity(k + 1);
            batch.push(t);
            batch.extend_from_slice(&drafts);
            let verify_out = self.forward_tokens_verify(&batch, state)?;

            // All-or-nothing: draft[i] must equal the main model's
            // prediction for the slot immediately after batch[i].
            let all_ok = (0..k).all(|i|
                crate::sampling::argmax(&verify_out[i]) == drafts[i]);

            stats.rounds  += 1;
            stats.drafted += k;

            if all_ok {
                stats.accepted += k;
                generated.push(t);
                generated.extend_from_slice(&drafts);
                verify_logits = verify_out[k].clone();
                prev_hidden   = self.verify_hidden_row(k);
            } else {
                // Reject every draft: roll the verify back and commit
                // only `t` with a normal single-token decode.
                snapshot.restore(state)?;
                verify_logits = self.forward_token(t, state)?;
                generated.push(t);
                prev_hidden = self.hidden_a.raw_ptr();
            }

            if let Some(p) = generated.iter().position(|&g| g == eos) {
                generated.truncate(p + 1);
                stats.hit_eos = true;
                break;
            }
            if generated.len() >= max_tokens {
                generated.truncate(max_tokens);
                break;
            }
        }
        Ok((generated, stats))
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
        let conv_dim  = self.gdn_conv_dim  as u32;
        let n_heads   = self.gdn_n_heads   as u32;
        let n_k_heads = self.gdn_n_k_heads as u32;
        let head_dim  = self.gdn_head_dim  as u32;
        let q_scale   = (self.gdn_head_dim as f32).powf(-0.5);

        // Embed lookup → hidden_a
        self.set_pos(state.pos)?;
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
            let q_in_ptr = unsafe { conv_out_ptr.add(0)                    } as *mut c_void;
            let k_in_ptr = unsafe { conv_out_ptr.add(self.gdn_key_dim)     } as *mut c_void;
            let v_in_ptr = unsafe { conv_out_ptr.add(2 * self.gdn_key_dim) } as *mut c_void;
            traced!("l2norm_qk", self.launch_l2norm_qk(q_in_ptr, self.gdn_q.raw_ptr(),
                k_in_ptr, self.gdn_k.raw_ptr(), n_k_heads, head_dim, 1e-6, q_scale));
            traced!("recurrent_step_fused", self.launch_gdn_recurrent_step_fused(
                self.gdn_q.raw_ptr(), self.gdn_k.raw_ptr(), v_in_ptr,
                self.gdn_a.raw_ptr(), self.gdn_b.raw_ptr(),
                w.attn.ssm_a.raw_ptr(), w.attn.ssm_dt_bias.raw_ptr(),
                lstate.recurrent.raw_ptr(), self.gdn_core_out.raw_ptr(),
                n_heads, head_dim, n_k_heads));
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
            self.step_ffn(self.hidden_b.raw_ptr(), self.hidden_b.raw_ptr(), &w.ffn)?;
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
        self.set_pos(state.pos)?;
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

    /// On-device decode body: transformer blocks + output norm +
    /// projection. Reads only `d_pos` and persistent device buffers —
    /// no token id, no host position — so it captures into a HIP graph
    /// that replays at every decode step. The token-dependent embed
    /// runs separately, just before this.
    fn enqueue_decode_body(&self, state: &mut Qwen35GpuState) -> Result<(), String> {
        assert_eq!(state.block_states.len(), self.blocks.len());
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

    /// On-device portion of `forward_token`: stage the position, embed
    /// the token, run the decode body. No host syncs.
    fn enqueue_forward_token(&self, token: u32, state: &mut Qwen35GpuState)
        -> Result<(), String>
    {
        self.set_pos(state.pos)?;
        self.launch_embed_lookup_dispatch(&self.token_embd, self.hidden_a.raw_ptr(), token)?;
        self.prof_reset();
        self.enqueue_decode_body(state)
    }

    /// Capture the decode body into a replayable HIP graph. The graph
    /// reads `d_pos` (staged per step) and the persistent KV / GDN
    /// state buffers, so one capture serves every decode position.
    pub fn capture_forward_graph(&self, state: &mut Qwen35GpuState)
        -> Result<GraphExec, String>
    {
        Graph::begin_capture(&self.stream, HipStreamCaptureMode::Global)?;
        if let Err(e) = self.enqueue_decode_body(state) {
            let _ = Graph::end_capture(&self.stream);
            return Err(e);
        }
        let graph = Graph::end_capture(&self.stream)?;
        let exec = graph.instantiate()?;
        drop(graph);
        Ok(exec)
    }

    /// Decode one token by replaying the captured graph: stage the
    /// position, embed the token, replay, read back logits.
    pub fn forward_token_via_graph(&self, exec: &GraphExec, token: u32,
                                   state: &mut Qwen35GpuState)
        -> Result<Vec<f32>, String>
    {
        self.set_pos(state.pos)?;
        self.launch_embed_lookup_dispatch(&self.token_embd, self.hidden_a.raw_ptr(), token)?;
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

        // Repacked Q4_K: the two-plane layout dequantizes a sub-block at a
        // time straight into [out_dim, in_dim] fp16 order.
        if w.repacked {
            let (module, kname) = match w.dtype {
                GgmlType::Q5_K => (&self.dequant_q5k_repacked_module, "dequant_q5k_repacked_f16"),
                GgmlType::Q6_K => (&self.dequant_q6k_repacked_module, "dequant_q6k_repacked_f16"),
                GgmlType::Q8_0 => (&self.dequant_q8_0_repacked_module, "dequant_q8_0_repacked_f16"),
                _              => (&self.dequant_q4k_repacked_module, "dequant_q4k_repacked_f16"),
            };
            let f = module.function(kname)?;
            let n_sub_total = (n / 32) as u32;   // out_dim * (in_dim/32)
            let mut w_ptr = w.data.raw_ptr();
            let mut o_ptr = out.raw_ptr();
            let mut ia = w.in_dim;
            let mut oa = w.out_dim;
            let mut args: [*mut c_void; 4] = [
                &mut w_ptr as *mut _ as *mut c_void,
                &mut o_ptr as *mut _ as *mut c_void,
                &mut ia    as *mut _ as *mut c_void,
                &mut oa    as *mut _ as *mut c_void,
            ];
            unsafe { f.launch((n_sub_total, 1, 1), (32, 1, 1), 0,
                              Some(&self.stream), &mut args)?; }
            return Ok(out);
        }

        // F32 weights (qwen35moe's ssm_alpha/beta) need no dequant — just
        // narrow to fp16 for the HGEMM.
        if w.dtype == GgmlType::F32 {
            self.launch_cvt("cvt_f32_to_f16", w.data.raw_ptr(), out.raw_ptr(), n as u32)?;
            return Ok(out);
        }

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

        // Small-N batched K-quant matvec — the spec-decode verify path.
        // The MMQ GEMM's 64-wide token tile makes a 3-row verify pay the
        // full 64-row compute on the compute-bound MI50; this kernel
        // reads each weight sub-block once and dots it against n_rows ≤ 4
        // activation rows, staying HBM-bound like a 1-row decode matvec.
        if w.repacked && n_rows <= 4
            && matches!(w.dtype, GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K) {
            return self.bmm_batched_kquant(w, x_f32, n_rows, y_f32);
        }

        // Repacked K-quants AND repacked Q8_0: the 2D-tiled int8 MMQ
        // GEMM consumes the quantised weight directly — no dequant to
        // fp16, no HGEMM. Q8_0 covers the GDN ssm_out weights (and
        // Unsloth's per-tensor Q8_0 layer overrides).
        if w.repacked && matches!(w.dtype,
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_0) {
            return self.bmm_mmq(w, x_f32, n_rows, y_f32);
        }

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
        // dq / x_f16 / y_f16 are local — freed when this fn returns. The
        // dequant / cvt / rocBLAS kernels above run async on the stream,
        // so sync before the buffers drop: otherwise the freed memory is
        // reused under the still-running kernels, a GPU memory fault
        // (hit on the 27B; the 0.8B raced through by luck).
        self.stream.synchronize()?;
        let _ = dq;
        Ok(())
    }

    /// Small-N (`n_rows` ≤ 4) batched K-quant matvec: quantise X →
    /// BlockQ8, then one launch reads each weight sub-block once and
    /// dots it against all `n_rows` activation rows. HBM-bound, unlike
    /// the 64-wide MMQ tile — the spec-decode verify path. `y_f32` is
    /// written `[n_rows, out_dim]`, identical layout to `bmm_mmq`.
    fn bmm_batched_kquant(&self, w: &GpuMatvecTensor, x_f32: *mut c_void,
                          n_rows: usize, y_f32: *mut c_void) -> Result<(), String>
    {
        let in_d  = w.in_dim as usize;
        let out_d = w.out_dim as usize;
        let (module, kname) = match w.dtype {
            GgmlType::Q5_K => (&self.matvec_q5k_batched_module, "matvec_q5k_repacked_batched_f32"),
            GgmlType::Q6_K => (&self.matvec_q6k_batched_module, "matvec_q6k_repacked_batched_f32"),
            _              => (&self.matvec_q4k_batched_module, "matvec_q4k_repacked_batched_f32"),
        };
        // Quantise the activation rows → BlockQ8 [n_rows, in_dim/32].
        let xq8 = self.pool_u8.take(n_rows * (in_d / 32) * 40)?;
        self.launch_quantize_q8_into(x_f32, xq8.raw_ptr(), in_d as u32, n_rows as u32)?;
        // grid.x = ceil(out_dim / 8) — the kernel emits 8 output rows/WG.
        let f = module.function(kname)?;
        let mut wp = w.data.raw_ptr(); let mut xp = xq8.raw_ptr(); let mut yp = y_f32;
        let mut ia = in_d as u32; let mut oa = out_d as u32; let mut nr = n_rows as u32;
        let mut args: [*mut c_void; 6] = [
            &mut wp as *mut _ as *mut c_void, &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void, &mut nr as *mut _ as *mut c_void];
        unsafe {
            f.launch(((out_d as u32 + 7) / 8, 1, 1), (256, 1, 1),
                     0, Some(&self.stream), &mut args)?;
        }
        Ok(())
    }

    /// Repacked-K-quant `Y = X · Wᵀ` via the 2D-tiled int8 MMQ GEMM:
    /// quantise X → BlockQ8, then one dp4a GEMM straight off the
    /// quantised weight.
    fn bmm_mmq(&self, w: &GpuMatvecTensor, x_f32: *mut c_void, n_rows: usize,
               y_f32: *mut c_void) -> Result<(), String>
    {
        let in_d = w.in_dim as usize;
        let out_d = w.out_dim as usize;
        let (module, kname) = match w.dtype {
            GgmlType::Q5_K => (&self.mmq_q5k_module, "mmq_gemm_q5k_repacked_f32"),
            GgmlType::Q6_K => (&self.mmq_q6k_module, "mmq_gemm_q6k_repacked_f32"),
            GgmlType::Q8_0 => (&self.mmq_q8_0_module, "mmq_gemm_q8_0_repacked_f32"),
            _              => (&self.mmq_q4k_module, "mmq_gemm_q4k_repacked_f32"),
        };
        // Quantise the activation rows → BlockQ8 [n_rows, in_dim/32].
        let xq8 = self.pool_u8.take(n_rows * (in_d / 32) * 40)?;
        self.launch_quantize_q8_into(x_f32, xq8.raw_ptr(), in_d as u32, n_rows as u32)?;

        // MMQ GEMM — BM=64 output rows × BN=64 tokens per workgroup.
        let f = module.function(kname)?;
        let mut wp = w.data.raw_ptr(); let mut xp = xq8.raw_ptr(); let mut yp = y_f32;
        let mut ia = in_d as u32; let mut oa = out_d as u32; let mut pa = n_rows as u32;
        let mut args: [*mut c_void; 6] = [
            &mut wp as *mut _ as *mut c_void, &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void, &mut pa as *mut _ as *mut c_void];
        unsafe { f.launch(((out_d as u32 + 63) / 64, (n_rows as u32 + 63) / 64, 1),
                          (256, 1, 1), 0, Some(&self.stream), &mut args)?; }
        // xq8 is pooled (not freed) — no per-call sync; the single
        // engine stream orders any later reuse after this kernel.
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

    /// Batched causal attention for the qwen full-attention prefill —
    /// the flash-attention kernel (full causal, window 0). BQ=8 queries
    /// per workgroup, BK=8-key LDS tiles; must match the kernel #defines.
    fn launch_attn_step_batched(&self, q: *mut c_void, k_cache: *mut c_void,
                                v_cache: *mut c_void, out: *mut c_void,
                                base_pos: u32, n_rows: u32, scaling: f32)
        -> Result<(), String>
    {
        const BQ: u32 = 8;
        const BK: u32 = 8;
        let f = self.attn_step_batched_module.function("attn_prefill_flash_f32")?;
        let block: u32 = 64 * BQ;
        let head_dim = self.head_dim as u32;
        let smem = 2 * BK * head_dim * 4;
        let mut qa = q; let mut ka = k_cache; let mut va = v_cache; let mut oa = out;
        let mut nh = self.n_heads as u32;
        let mut nkv = self.n_kv_heads as u32;
        let mut hd = head_dim;
        let mut wn = 0u32;          // qwen full-attention layers: full causal
        let mut sc = scaling;
        let mut nr = n_rows;
        let mut bp = base_pos;
        let mut args: [*mut c_void; 11] = [
            &mut qa as *mut _ as *mut c_void,
            &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut wn as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((self.n_heads as u32, (n_rows + BQ - 1) / BQ, 1),
                          (block, 1, 1), smem, Some(&self.stream), &mut args) }
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
        let ba    = self.pool_f32.take(n * h)?;        // running hidden
        let bb    = self.pool_f32.take(n * h)?;        // scratch
        let bnorm = self.pool_f32.take(n * h)?;        // normed scratch

        // 1) Embed all tokens into ba (one row each).
        for (r, &tok) in tokens.iter().enumerate() {
            let row_ptr = unsafe { (ba.raw_ptr() as *mut f32).add(r * h) } as *mut c_void;
            self.launch_embed_lookup_dispatch(&self.token_embd, row_ptr, tok)?;
        }

        // 2) Every block. Optional per-block timing trace
        //    (REINSTINCT_PREFILL_TRACE=1) — syncs after each block, so
        //    only use for diagnosis, not steady-state benchmarking.
        let trace = std::env::var_os("REINSTINCT_PREFILL_TRACE").is_some();
        let mut block_ms: Vec<(char, f64)> = Vec::with_capacity(self.blocks.len());
        for (block, st) in self.blocks.iter().zip(state.block_states.iter_mut()) {
            let t0 = if trace { Some(std::time::Instant::now()) } else { None };
            let kind: char;
            match (block, st) {
                (GpuBlock::Full(w), GpuBlockState::Full(kv)) => {
                    self.batched_full_block(&ba, &bb, &bnorm, w, kv, n, scaling)?;
                    kind = 'F';
                }
                (GpuBlock::Linear(w), GpuBlockState::Linear(s)) => {
                    self.batched_linear_block(&ba, &bb, &bnorm, w, s, n)?;
                    kind = 'L';
                }
                _ => return Err("block kind mismatch".into()),
            }
            if let Some(t0) = t0 {
                self.stream.synchronize()?;
                block_ms.push((kind, t0.elapsed().as_secs_f64() * 1e3));
            }
        }
        if trace {
            let (mut sf, mut nf, mut sl, mut nl) = (0.0, 0usize, 0.0, 0usize);
            for &(k, ms) in &block_ms {
                if k == 'F' { sf += ms; nf += 1; } else { sl += ms; nl += 1; }
            }
            eprintln!("[prefill-trace] {} tokens × {} blocks  ({}F + {}L)",
                n, block_ms.len(), nf, nl);
            eprintln!("[prefill-trace]   full-attn  {:>7.1} ms total  ({:>5.2} ms/block)",
                sf, if nf > 0 { sf / nf as f64 } else { 0.0 });
            eprintln!("[prefill-trace]   GDN linear {:>7.1} ms total  ({:>5.2} ms/block)",
                sl, if nl > 0 { sl / nl as f64 } else { 0.0 });
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

    /// QMTP-2 — K-token verify forward. Runs `tokens` through the main
    /// model in one batched pass (the same block kernels as
    /// `forward_tokens_batched`) but projects EVERY row, returning the
    /// logits at every position rather than only the last.
    ///
    /// Used by the MTP spec-decode loop to verify a block of drafted
    /// tokens in a single forward. `state` advances by `tokens.len()`
    /// positions (KV caches + GDN state stepped once per token) — the
    /// caller (QMTP-3) is responsible for rolling back rejected tail
    /// positions. Works from any mid-sequence state: full-attn blocks
    /// append at `kv.len`, GDN blocks thread their state forward.
    pub fn forward_tokens_verify(&self, tokens: &[u32], state: &mut Qwen35GpuState)
        -> Result<Vec<Vec<f32>>, String>
    {
        assert!(!tokens.is_empty(), "forward_tokens_verify needs ≥1 token");
        let n = tokens.len();
        assert!(n <= VERIFY_MAX_TOKENS,
                "forward_tokens_verify: {n} tokens exceeds VERIFY_MAX_TOKENS");
        let h = self.hidden;
        let scaling = (self.head_dim as f32).powf(-0.5);

        let ba    = self.pool_f32.take(n * h)?;   // running hidden
        let bb    = self.pool_f32.take(n * h)?;   // scratch
        let bnorm = self.pool_f32.take(n * h)?;   // normed scratch

        for (r, &tok) in tokens.iter().enumerate() {
            let row_ptr = unsafe { (ba.raw_ptr() as *mut f32).add(r * h) } as *mut c_void;
            self.launch_embed_lookup_dispatch(&self.token_embd, row_ptr, tok)?;
        }
        for (block, st) in self.blocks.iter().zip(state.block_states.iter_mut()) {
            match (block, st) {
                (GpuBlock::Full(w), GpuBlockState::Full(kv)) =>
                    self.batched_full_block(&ba, &bb, &bnorm, w, kv, n, scaling)?,
                (GpuBlock::Linear(w), GpuBlockState::Linear(s)) =>
                    self.batched_linear_block(&ba, &bb, &bnorm, w, s, n)?,
                _ => return Err("block kind mismatch".into()),
            }
        }

        // Stash the per-position hidden states (pre output-norm) — the
        // MTP spec-decode loop reads a row as the next `prev_hidden`.
        self.verify_hidden.copy_from_device_at(&ba, 0)?;

        // Output norm (all rows) + projection. The projection stays a
        // per-row decode matvec: the (tied) output weight is not in the
        // repacked layout, so routing it through `bmm` would hit the
        // dequant-to-fp16 fallback — far worse than n fast matvecs.
        self.launch_rmsnorm_multihead(ba.raw_ptr(), self.output_norm.raw_ptr(),
                                      bnorm.raw_ptr(), n as u32, h as u32, self.rms_eps)?;
        let logits_all = self.pool_f32.take(n * self.vocab)?;
        for r in 0..n {
            let in_ptr  = unsafe { (bnorm.raw_ptr() as *mut f32).add(r * h) } as *mut c_void;
            let out_ptr = unsafe {
                (logits_all.raw_ptr() as *mut f32).add(r * self.vocab) } as *mut c_void;
            self.launch_matvec_dispatch(self.output_proj_tensor(), in_ptr, out_ptr)?;
        }
        self.stream.synchronize()?;
        state.pos += n;

        let mut flat = vec![0.0f32; n * self.vocab];
        logits_all.copy_to_host(&mut flat)?;
        Ok(flat.chunks(self.vocab).map(|c| c.to_vec()).collect())
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
        let q_raw = self.pool_f32.take(n * 2 * q_dim)?;
        let k_raw = self.pool_f32.take(n * kv_dim)?;
        let v_raw = self.pool_f32.take(n * kv_dim)?;
        self.bmm(&w.attn.attn_q, bnorm.raw_ptr(), n, q_raw.raw_ptr())?;
        self.bmm(&w.attn.attn_k, bnorm.raw_ptr(), n, k_raw.raw_ptr())?;
        self.bmm(&w.attn.attn_v, bnorm.raw_ptr(), n, v_raw.raw_ptr())?;

        // split q_raw → q, gate. The split kernel walks n_heads*head_dim
        // elements; passing n*n_heads covers all rows.
        let q_buf = self.pool_f32.take(n * q_dim)?;
        let gate  = self.pool_f32.take(n * q_dim)?;
        self.launch_split_q_gate(q_raw.raw_ptr(), q_buf.raw_ptr(), gate.raw_ptr(),
                                 (n * self.n_heads) as u32, self.head_dim as u32)?;
        // per-head Q-norm (n*n_heads independent heads).
        self.launch_rmsnorm_multihead(q_buf.raw_ptr(), w.attn.attn_q_norm.raw_ptr(),
                                      q_buf.raw_ptr(),
                                      (n * self.n_heads) as u32, self.head_dim as u32,
                                      self.rms_eps)?;
        self.launch_rope_batched(q_buf.raw_ptr(), self.n_heads as u32, n as u32, base_pos as u32)?;
        // per-kv-head K-norm.
        let k_norm = self.pool_f32.take(n * kv_dim)?;
        self.launch_rmsnorm_multihead(k_raw.raw_ptr(), w.attn.attn_k_norm.raw_ptr(),
                                      k_norm.raw_ptr(),
                                      (n * self.n_kv_heads) as u32, self.head_dim as u32,
                                      self.rms_eps)?;
        self.launch_rope_batched(k_norm.raw_ptr(), self.n_kv_heads as u32, n as u32, base_pos as u32)?;

        // Push all N (k, v) into the cache at slots [base_pos, base_pos+n).
        kv.k.copy_from_device_at_async(&k_norm, base_pos * kv_dim, &self.stream)?;
        kv.v.copy_from_device_at_async(&v_raw,  base_pos * kv_dim, &self.stream)?;

        // Batched causal attention → attn_concat.
        let attn = self.pool_f32.take(n * q_dim)?;
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
                   ffn: &BlockFfn, n: usize) -> Result<(), String>
    {
        match ffn {
            BlockFfn::Dense(d) => {
                let f = self.ffn;
                let gate = self.pool_f32.take(n * f)?;
                let up   = self.pool_f32.take(n * f)?;
                self.bmm(&d.gate, input.raw_ptr(), n, gate.raw_ptr())?;
                self.bmm(&d.up,   input.raw_ptr(), n, up.raw_ptr())?;
                self.launch_swiglu(gate.raw_ptr(), up.raw_ptr(), gate.raw_ptr(), (n * f) as u32)?;
                self.bmm(&d.down, gate.raw_ptr(), n, out_bb.raw_ptr())?;
            }
            BlockFfn::Moe(m) => {
                // Prefill MoE: process the rows in MOE_PREFILL_CHUNK-sized
                // batches — routed-expert matvecs batch over tokens (grid.z),
                // the router + shared expert go through rocBLAS GEMMs.
                let h = self.hidden;
                let mut off = 0;
                while off < n {
                    let c = (n - off).min(MOE_PREFILL_CHUNK);
                    let inp  = unsafe { (input.raw_ptr()  as *mut f32).add(off * h) as *mut c_void };
                    let outp = unsafe { (out_bb.raw_ptr() as *mut f32).add(off * h) as *mut c_void };
                    self.step_moe_ffn_batched(inp, outp, m, c)?;
                    off += c;
                }
            }
        }
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
        let kdim = self.gdn_key_dim;
        let cdim = self.gdn_conv_dim;
        let nh   = self.gdn_n_heads as u32;        // value heads
        let nkh  = self.gdn_n_k_heads as u32;      // key/query heads
        let hd   = self.gdn_head_dim as u32;
        let q_scale = (self.gdn_head_dim as f32).powf(-0.5);

        // pre-norm.
        self.launch_rmsnorm_multihead(ba.raw_ptr(), w.attn.attn_norm.raw_ptr(),
                                      bnorm.raw_ptr(), n as u32, h as u32, self.rms_eps)?;

        // Four projections, batched.
        let qkv = self.pool_f32.take(n * cdim)?;
        let z   = self.pool_f32.take(n * vdim)?;
        let a   = self.pool_f32.take(n * self.gdn_n_heads)?;
        let b   = self.pool_f32.take(n * self.gdn_n_heads)?;
        self.bmm(&w.attn.attn_qkv,  bnorm.raw_ptr(), n, qkv.raw_ptr())?;
        self.bmm(&w.attn.attn_gate, bnorm.raw_ptr(), n, z.raw_ptr())?;
        self.bmm(&w.attn.ssm_alpha, bnorm.raw_ptr(), n, a.raw_ptr())?;
        self.bmm(&w.attn.ssm_beta,  bnorm.raw_ptr(), n, b.raw_ptr())?;

        // conv1d + SiLU, batched: one launch over all n rows (the kernel
        // threads the conv history through the rows internally).
        let conv_out = self.pool_f32.take(n * cdim)?;
        self.launch_conv1d_step_silu_batched(
            qkv.raw_ptr(), w.attn.ssm_conv1d.raw_ptr(),
            st.conv_hist.raw_ptr(), conv_out.raw_ptr(),
            cdim as u32, self.gdn_conv_kernel as u32, n as u32)?;

        // L2-norm Q/K → q_all/k_all [n, kdim]. The conv output is
        // [n, cdim] with layout (q | k | v) per row; the batched kernel
        // strides by cdim to walk rows.
        let q_all = self.pool_f32.take(n * kdim)?;
        let k_all = self.pool_f32.take(n * kdim)?;
        let conv_q_ptr = conv_out.raw_ptr();
        let conv_k_ptr = unsafe { (conv_out.raw_ptr() as *mut f32).add(kdim) } as *mut c_void;
        self.launch_l2norm_qk_batched(
            conv_q_ptr, q_all.raw_ptr(),
            conv_k_ptr, k_all.raw_ptr(),
            nkh, hd, 1e-6, q_scale, n as u32,
            cdim as u32,   // q_in_row_stride  — q half of conv_out, stride cdim
            kdim as u32,   // q_out_row_stride — q_all is dense [n, kdim]
            cdim as u32,   // k_in_row_stride
            kdim as u32)?; // k_out_row_stride

        // Recurrent step + decay/beta — single launch over all n rows;
        // the kernel loops internally with state threaded through.
        // v_in points at the v half of conv_out (offset 2*kdim per row).
        let core = self.pool_f32.take(n * vdim)?;
        let conv_v_ptr = unsafe { (conv_out.raw_ptr() as *mut f32).add(2 * kdim) } as *mut c_void;
        self.launch_gdn_recurrent_step_fused_batched(
            q_all.raw_ptr(), k_all.raw_ptr(), conv_v_ptr,
            a.raw_ptr(), b.raw_ptr(),
            w.attn.ssm_a.raw_ptr(), w.attn.ssm_dt_bias.raw_ptr(),
            st.recurrent.raw_ptr(), core.raw_ptr(),
            nh, hd, nkh, n as u32,
            kdim as u32,                                // qk_row_stride
            cdim as u32,                                // v_row_stride (conv layout)
            self.gdn_n_heads as u32,                    // ab_row_stride
            vdim as u32)?;                              // out_row_stride

        // Gated RMSNorm with z = attn_gate output (already [n, vdim]).
        self.launch_rmsnorm_gated_multihead_batched(
            core.raw_ptr(), z.raw_ptr(), w.attn.ssm_norm.raw_ptr(),
            core.raw_ptr(), nh, hd, self.rms_eps,
            n as u32, vdim as u32)?;

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
        self.step_ffn(self.hidden_b.raw_ptr(), self.hidden_b.raw_ptr(),
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
        let n_heads   = self.gdn_n_heads   as u32;     // value heads
        let n_k_heads = self.gdn_n_k_heads as u32;     // key/query heads
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

        // 4) conv_out is laid out [Q | K | V]: Q/K are key_dim wide
        //    (n_k_heads heads), V is value_dim wide (n_heads heads).
        let conv_out_ptr = self.gdn_conv_out.raw_ptr() as *mut f32;
        let q_in_ptr = unsafe { conv_out_ptr.add(0)                    } as *mut c_void;
        let k_in_ptr = unsafe { conv_out_ptr.add(self.gdn_key_dim)     } as *mut c_void;
        let v_in_ptr = unsafe { conv_out_ptr.add(2 * self.gdn_key_dim) } as *mut c_void;

        // 5) Per-head L2-norm of Q (scale 1/√head_dim) and K (scale 1) —
        //    n_k_heads heads each.
        self.launch_l2norm_qk(q_in_ptr, self.gdn_q.raw_ptr(),
                              k_in_ptr, self.gdn_k.raw_ptr(),
                              n_k_heads, head_dim, 1e-6, q_scale)?;

        // 6+7) Recurrent gated delta-rule update — decay/beta computed
        //      inside the kernel from a/b/ssm_a/dt_bias.
        self.launch_gdn_recurrent_step_fused(self.gdn_q.raw_ptr(), self.gdn_k.raw_ptr(), v_in_ptr,
                                             self.gdn_a.raw_ptr(), self.gdn_b.raw_ptr(),
                                             weights.ssm_a.raw_ptr(), weights.ssm_dt_bias.raw_ptr(),
                                             state.recurrent.raw_ptr(),
                                             self.gdn_core_out.raw_ptr(),
                                             n_heads, head_dim, n_k_heads)?;

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
        self.step_ffn(self.hidden_b.raw_ptr(), self.hidden_b.raw_ptr(), &weights.ffn)?;
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
        let model = Qwen35Model::load(&g).expect("load model");
        let cfg = &model.config;
        let hidden = cfg.hidden_size as usize;
        let vocab  = cfg.vocab_size as usize;

        // Load only the embedding table + output norm/proj — not the
        // whole f32 model (≈100 GB of block weights on the 27B).
        use crate::cpu::qwen3_5::dequant_named;
        let token_embd  = dequant_named(&g, "token_embd.weight").expect("token_embd");
        let output_norm = dequant_named(&g, "output_norm.weight").expect("output_norm");
        let output = if cfg.tied_embeddings { None }
                     else { Some(dequant_named(&g, "output.weight").expect("output")) };

        let mut gpu = GpuQwen35::new(&model, &g, &cache, 32).expect("new GpuQwen35");
        gpu.set_dp4a(false);  // consistency check vs the fp32 CPU oracle

        // Test on a couple of tokens including EOS and a mid-vocab.
        for &token in &[cfg.eos_token_id, 100u32, 50_000u32] {
            // CPU oracle: embed → output_norm → output_proj.
            let off = token as usize * hidden;
            let embed = &token_embd[off..off + hidden];
            let mut normed = vec![0.0f32; hidden];
            crate::cpu::ops::rmsnorm(embed, &output_norm, cfg.rms_norm_eps, &mut normed);
            let proj_w = output.as_deref().unwrap_or(token_embd.as_slice());
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
            let weights = GpuFfnWeights::from_gguf(&g, block_idx as u32, false).expect("alloc ffn weights");
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
        let mut gpu = GpuQwen35::new(&m.model, &g, &cache, max_seq).expect("new GpuQwen35");
        gpu.set_dp4a(false);  // consistency check vs the fp32 CPU oracle

        // Validate against the CPU oracle on a handful of single tokens
        // we already have golden coverage for.
        for &token in &[cfg.eos_token_id, 100u32, 50_000u32] {
            let mut cpu_state = m.new_state(max_seq);
            let cpu_logits = m.forward_token(token, &mut cpu_state);

            let mut gpu_state = Qwen35GpuState::new(&m.model,max_seq).expect("new gpu state");
            let gpu_logits = gpu.forward_token(token, &mut gpu_state).expect("gpu forward");

            assert_eq!(gpu_logits.len(), vocab);

            // The GPU runs with set_dp4a(false) — the fp32-activation
            // path — but its weights are still fp16 (dequantised from the
            // repacked K-quant), so a full 24-layer forward drifts from
            // the all-f32 CPU oracle by ~0.05 of logit magnitude. That is
            // fp16 storage, not an error: the tolerance is sized for it,
            // and top-K agreement is the behaviourally meaningful check.
            const ABS_TOL: f32 = 0.15;
            const REL_TOL: f32 = 0.03;
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

            let topk = |v: &[f32], k: usize| -> Vec<usize> {
                let mut idx: Vec<usize> = (0..v.len()).collect();
                idx.sort_unstable_by(|&a, &b| v[b].total_cmp(&v[a]));
                idx[..k].to_vec()
            };
            let cpu_top = topk(&cpu_logits, 32);
            let gpu_top = topk(&gpu_logits, 32);
            let cpu_set: std::collections::HashSet<_> = cpu_top.iter().collect();
            let top_agree = gpu_top.iter().filter(|i| cpu_set.contains(i)).count();
            eprintln!("token {token}: max_abs={max_abs:.3e}, argmax cpu={} gpu={}, \
                top-32 agreement {top_agree}/32", cpu_top[0], gpu_top[0]);

            assert_eq!(cpu_top[0], gpu_top[0],
                "token {token}: argmax disagree (cpu={} gpu={})", cpu_top[0], gpu_top[0]);
            // ≥30/32: a 0.05 fp16 perturbation can swap a token across the
            // rank-32 boundary; a real regression tanks the agreement.
            assert!(top_agree >= 30, "token {token}: top-32 sets disagree ({top_agree}/32)");
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

        use crate::cpu::qwen3_5::{LinAttnState as CpuLinAttnState,
                                  linear_attention_step, load_linear_attention};

        let g = GgufFile::open(&path).unwrap();
        // Load only the config + one GDN block, not the whole f32 model —
        // Qwen35F32Model::load would materialise the entire model (≈100 GB
        // for the 27B) in host RAM.
        let model = Qwen35Model::load(&g).unwrap();
        let cfg = &model.config;
        let h = cfg.hidden_size as usize;

        let block_idx = model.block_kinds.iter()
            .position(|k| matches!(k, crate::model::qwen3_5::BlockKind::LinearAttention))
            .expect("model has at least one LinearAttention block");
        let weights = &load_linear_attention(&g, block_idx as u32)
            .expect("load CPU GDN block");
        eprintln!("validating GDN step on block {block_idx}");

        let conv_dim = cfg.gdn_qkv_concat_dim() as usize;
        let mut cpu_state = CpuLinAttnState::new(
            cfg.gdn_n_heads as usize,
            cfg.gdn_head_dim as usize,
            cfg.gdn_head_dim as usize,
            conv_dim,
            cfg.gdn_conv_kernel as usize,
        );

        let mut gpu = GpuQwen35::new(&model, &g, &cache, 16).expect("new GpuQwen35");
        gpu.set_dp4a(false);  // consistency check vs the fp32 CPU oracle
        let gpu_w = GpuLinAttnWeights::from_gguf(&g, block_idx as u32, false).expect("upload GDN weights");
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

        use crate::cpu::qwen3_5::{LinAttnState as CpuLinAttnState,
                                  linear_attention_block, load_linear_attention};

        let g = GgufFile::open(&path).unwrap();
        let model = Qwen35Model::load(&g).unwrap();
        let cfg = &model.config;
        let h = cfg.hidden_size as usize;

        let block_idx = model.block_kinds.iter()
            .position(|k| matches!(k, crate::model::qwen3_5::BlockKind::LinearAttention))
            .expect("model has at least one LinearAttention block");
        let weights = &load_linear_attention(&g, block_idx as u32)
            .expect("load CPU GDN block");

        let conv_dim = cfg.gdn_qkv_concat_dim() as usize;
        let mut cpu_state = CpuLinAttnState::new(
            cfg.gdn_n_heads as usize,
            cfg.gdn_head_dim as usize,
            cfg.gdn_head_dim as usize,
            conv_dim,
            cfg.gdn_conv_kernel as usize,
        );

        let mut gpu = GpuQwen35::new(&model, &g, &cache, 16).expect("new GpuQwen35");
        gpu.set_dp4a(false);  // consistency check vs the fp32 CPU oracle
        let gpu_block = GpuLinAttnBlock::from_gguf(&g, block_idx as u32, false, false).expect("upload GDN block");
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

        use crate::cpu::qwen3_5::{LayerKvCache, full_attention_block, load_full_attention};
        use crate::cpu::rope::RopeCache;

        let g = GgufFile::open(&path).unwrap();
        let model = Qwen35Model::load(&g).unwrap();
        let cfg = &model.config;
        let h = cfg.hidden_size as usize;

        let block_idx = model.block_kinds.iter()
            .position(|k| matches!(k, crate::model::qwen3_5::BlockKind::FullAttention))
            .expect("model has at least one FullAttention block");
        let weights = &load_full_attention(&g, block_idx as u32)
            .expect("load CPU full-attn block");
        eprintln!("validating full block {block_idx} (FullAttention + FFN)");

        let max_seq = 16usize;
        let mut layer_kv = LayerKvCache::new(
            max_seq,
            cfg.attn_n_kv_heads as usize,
            cfg.attn_head_dim   as usize,
        );
        let rope = RopeCache::new(cfg.rope_dim_count as usize, max_seq, cfg.rope_freq_base);

        let mut gpu = GpuQwen35::new(&model, &g, &cache, max_seq).expect("new GpuQwen35");
        gpu.set_dp4a(false);  // consistency check vs the fp32 CPU oracle
        let gpu_block = GpuFullAttnBlock::from_gguf(&g, block_idx as u32, false, false).expect("upload block");
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
        let gpu_w = GpuFullAttnWeights::from_gguf(&g, block_idx as u32, false).expect("upload attn weights");
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

            // Compare. The GPU stores K/V as an int8 cache and its
            // weights as fp16; the CPU oracle is all-f32. The block
            // output therefore drifts from the oracle by an amount that
            // grows with the cached-position count — ~1.3e-2 by step 3.
            // That is the engine's real quantised behaviour, not an error.
            const ABS_TOL: f32 = 2.0e-2;
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
