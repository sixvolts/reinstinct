//! Depthwise causal 1-D convolution for the GDN block's mixed_qkv path.
//!
//! Conv: per-channel, kernel size 4, no bias, `padding = kernel - 1` left
//! padding (right-truncated to make it causal). Each output sample depends
//! only on the current and prior `kernel-1` input samples on the same channel.
//!
//! Weights follow GGUF ssm_conv1d.weight layout `[kernel, n_channels]` in
//! ggml convention, which means flat layout `w[ch * kernel + k]` — each
//! channel's 4 kernel values are contiguous. The PyTorch Conv1d weight is
//! `[n_channels, 1, kernel]` squeezed; our flat index matches.

/// Conv: y[t] = sum_{k=0..K-1} w[k] * x[t + k - (K-1)] with x[i<0] = 0.
/// (PyTorch's Conv1d semantics with padding = K-1, then right-truncated.)
pub fn conv1d_full_seq(
    x: &[f32],          // [n_channels, seq_len]
    w: &[f32],          // [n_channels, kernel_size]
    n_channels: usize,
    kernel_size: usize,
    seq_len: usize,
    out: &mut [f32],    // [n_channels, seq_len]
) {
    assert_eq!(x.len(), n_channels * seq_len);
    assert_eq!(w.len(), n_channels * kernel_size);
    assert_eq!(out.len(), n_channels * seq_len);

    for ch in 0..n_channels {
        let xs = &x[ch * seq_len..(ch + 1) * seq_len];
        let ws = &w[ch * kernel_size..(ch + 1) * kernel_size];
        let ys = &mut out[ch * seq_len..(ch + 1) * seq_len];
        for t in 0..seq_len {
            let mut acc = 0.0_f32;
            for k in 0..kernel_size {
                let src = t as i64 + k as i64 - (kernel_size as i64 - 1);
                if src >= 0 {
                    acc += ws[k] * xs[src as usize];
                }
            }
            ys[t] = acc;
        }
    }
}

/// Streaming state for incremental (token-at-a-time) Conv1D.
/// Holds the last `kernel_size - 1` samples per channel.
pub struct Conv1dState {
    /// `[n_channels, kernel_size - 1]` rolling window, flat row-major.
    history: Vec<f32>,
    pub n_channels: usize,
    pub kernel_size: usize,
}

impl Conv1dState {
    pub fn new(n_channels: usize, kernel_size: usize) -> Self {
        assert!(kernel_size >= 1);
        Self {
            history: vec![0.0; n_channels * (kernel_size - 1)],
            n_channels,
            kernel_size,
        }
    }

    /// Process one new sample per channel; advance state. Output `y[ch]` is
    /// the conv at the current timestep on channel `ch`.
    ///
    /// Equivalent to `conv1d_full_seq` applied to a sequence of one new
    /// sample appended to the implicit history. Calling `step` T times
    /// produces the same outputs as `conv1d_full_seq` over T samples.
    pub fn step(&mut self, x_new: &[f32], w: &[f32], y: &mut [f32]) {
        let k = self.kernel_size;
        assert_eq!(x_new.len(), self.n_channels);
        assert_eq!(y.len(), self.n_channels);
        assert_eq!(w.len(), self.n_channels * k);
        let hist_w = k - 1;

        for ch in 0..self.n_channels {
            let h_off = ch * hist_w;
            let w_off = ch * k;
            // y = w[0] * h[0] + w[1] * h[1] + ... + w[K-2] * h[K-2] + w[K-1] * x_new
            let mut acc = w[w_off + k - 1] * x_new[ch];
            for kk in 0..k - 1 {
                acc += w[w_off + kk] * self.history[h_off + kk];
            }
            y[ch] = acc;

            // Shift history left by 1, append x_new[ch] at the end.
            for kk in 0..k.saturating_sub(2) {
                self.history[h_off + kk] = self.history[h_off + kk + 1];
            }
            if k >= 2 {
                self.history[h_off + k - 2] = x_new[ch];
            }
        }
    }

    /// Reset history to zeros (call at the start of a fresh sequence).
    pub fn reset(&mut self) {
        for v in self.history.iter_mut() { *v = 0.0; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn full_seq_single_channel_kernel_2() {
        // K=2, single channel, w=[w0, w1], x=[x0, x1, x2].
        // y[0] = w0 * 0   + w1 * x0
        // y[1] = w0 * x0  + w1 * x1
        // y[2] = w0 * x1  + w1 * x2
        let x = vec![1.0_f32, 2.0, 3.0];
        let w = vec![0.5_f32, -1.0];
        let mut out = vec![0.0_f32; 3];
        conv1d_full_seq(&x, &w, 1, 2, 3, &mut out);
        assert!(approx_eq(out[0],  0.0 +  -1.0,        1e-6)); // w0*0 + w1*1 = -1
        assert!(approx_eq(out[1],  0.5 +  -2.0,        1e-6)); // w0*1 + w1*2 = -1.5
        assert!(approx_eq(out[2],  1.0 +  -3.0,        1e-6)); // w0*2 + w1*3 = -2
    }

    #[test]
    fn full_seq_kernel_4_matches_handcomputed() {
        // K=4, x=[1,2,3,4,5], w=[a,b,c,d].
        // y[0] = a*0 + b*0 + c*0 + d*1 = d
        // y[1] = a*0 + b*0 + c*1 + d*2 = c + 2d
        // y[2] = a*0 + b*1 + c*2 + d*3 = b + 2c + 3d
        // y[3] = a*1 + b*2 + c*3 + d*4 = a + 2b + 3c + 4d
        // y[4] = a*2 + b*3 + c*4 + d*5 = 2a + 3b + 4c + 5d
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let w = vec![0.1_f32, 0.2, 0.3, 0.4]; // a, b, c, d
        let mut out = vec![0.0_f32; 5];
        conv1d_full_seq(&x, &w, 1, 4, 5, &mut out);
        assert!(approx_eq(out[0], 0.4, 1e-6));
        assert!(approx_eq(out[1], 0.3 + 0.8, 1e-6));
        assert!(approx_eq(out[2], 0.2 + 0.6 + 1.2, 1e-6));
        assert!(approx_eq(out[3], 0.1 + 0.4 + 0.9 + 1.6, 1e-6));
        assert!(approx_eq(out[4], 0.2 + 0.6 + 1.2 + 2.0, 1e-6));
    }

    #[test]
    fn full_seq_two_channels_independent() {
        // K=2, 2 channels with different weights.
        // x_ch0 = [1, 2], w_ch0 = [1, 0]   → y_ch0 = [0, 1]
        // x_ch1 = [3, 4], w_ch1 = [0, 1]   → y_ch1 = [3, 4]
        let x = vec![1.0_f32, 2.0,    3.0, 4.0];
        let w = vec![1.0_f32, 0.0,    0.0, 1.0];
        let mut out = vec![0.0_f32; 4];
        conv1d_full_seq(&x, &w, 2, 2, 2, &mut out);
        assert!(approx_eq(out[0], 0.0, 1e-6));
        assert!(approx_eq(out[1], 1.0, 1e-6));
        assert!(approx_eq(out[2], 3.0, 1e-6));
        assert!(approx_eq(out[3], 4.0, 1e-6));
    }

    #[test]
    fn streaming_step_matches_full_seq() {
        // Streaming form must produce the same outputs as the batch form
        // when called sample-by-sample over the same input.
        let n_channels = 3;
        let kernel_size = 4;
        let seq_len = 7;

        let mut x = vec![0.0_f32; n_channels * seq_len];
        for ch in 0..n_channels {
            for t in 0..seq_len {
                x[ch * seq_len + t] = ((ch + 1) as f32) * (t as f32 + 0.5);
            }
        }
        let w: Vec<f32> = (0..n_channels * kernel_size)
            .map(|i| ((i as f32) * 0.1) - 0.3)
            .collect();

        let mut full_out = vec![0.0_f32; n_channels * seq_len];
        conv1d_full_seq(&x, &w, n_channels, kernel_size, seq_len, &mut full_out);

        let mut state = Conv1dState::new(n_channels, kernel_size);
        let mut x_step = vec![0.0_f32; n_channels];
        let mut y_step = vec![0.0_f32; n_channels];
        let mut stream_out = vec![0.0_f32; n_channels * seq_len];
        for t in 0..seq_len {
            for ch in 0..n_channels {
                x_step[ch] = x[ch * seq_len + t];
            }
            state.step(&x_step, &w, &mut y_step);
            for ch in 0..n_channels {
                stream_out[ch * seq_len + t] = y_step[ch];
            }
        }

        for i in 0..stream_out.len() {
            assert!(approx_eq(stream_out[i], full_out[i], 1e-6),
                "i={i}: streaming {} vs full {}", stream_out[i], full_out[i]);
        }
    }

    #[test]
    fn streaming_reset_clears_history() {
        let mut state = Conv1dState::new(1, 4);
        let w = vec![1.0_f32, 0.0, 0.0, 0.0]; // y[t] = x[t-3]
        let mut x = vec![0.0_f32];
        let mut y = vec![0.0_f32];
        for v in [1.0_f32, 2.0, 3.0, 4.0] {
            x[0] = v;
            state.step(&x, &w, &mut y);
        }
        // w = [1,0,0,0] makes the conv y[t] = x[t-3] (since output reads
        // hist[0] = sample from K-1 = 3 steps ago).
        // After 4 samples [1,2,3,4]: y[3] = x[0] = 1.
        assert_eq!(y[0], 1.0);

        state.reset();
        x[0] = 99.0;
        state.step(&x, &w, &mut y);
        // After reset + 1 step: hist=[0,0,99]. y = w[0]*0 = 0.
        assert_eq!(y[0], 0.0);
    }
}
