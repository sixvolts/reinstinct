            GgmlType::Q4_K => Some((crate::quant::q4_k::repack_for_matvec(
                bytes, in_dim as usize, out_dim as usize), GgmlType::Q4_K)),
            GgmlType::Q5_K => Some((crate::quant::q5_k::repack_for_matvec(
                bytes, in_dim as usize, out_dim as usize), GgmlType::Q5_K)),
            GgmlType::Q6_K => Some((crate::quant::q6_k::repack_for_matvec(
                bytes, in_dim as usize, out_dim as usize), GgmlType::Q6_K)),
            GgmlType::Q8_0 => Some((crate::quant::q8_0::repack_for_matvec(
                bytes, in_dim as usize, out_dim as usize), GgmlType::Q8_0)),
            // IQ4_XS gets its own repacked kernels: the layout is Q4_0's with
            // the codebook applied in-kernel, so it stays at 4.25 bpw on
            // device. It used to be transcoded to Q8_0; on Qwen3.8-27B that
            // cost +21% VRAM and -21% decode because 2.9 GB of the file is
            // IQ4_XS.
            GgmlType::IQ4_XS => Some((crate::quant::iq4_xs::repack_for_matvec(
                bytes, in_dim as usize, out_dim as usize), GgmlType::IQ4_XS)),
            // The other formats Unsloth's UD-XL files reach for all relabel
            // exactly onto a layout with kernels. IQ4_NL is IQ4_XS without
            // the sub-scales: same codebook, `d` per 32 — the repacked IQ4_XS
            // layout bit for bit, so it runs as IQ4_XS.
            GgmlType::IQ4_NL => Some((crate::quant::iq4_nl::repack_for_matvec(
                bytes, in_dim as usize, out_dim as usize), GgmlType::IQ4_XS)),
            // IQ3_S: odd values in -15..15 under one scale per 32 — a nibble
            // through a 16-entry codebook, so the IQ4_XS layout again, and
            // the IQ4_XS kernels compiled with that codebook (4.5 bpw).
            GgmlType::IQ3_S => Some((crate::quant::iq3_s::repack_for_matvec(
                bytes, in_dim as usize, out_dim as usize), GgmlType::IQ3_S)),
            // Q3_K is Q6_K with a narrower quant: `q6 = q3 + 28`,
            // `sc6 = sc3 - 32`, same `d` — bit-exact, and 6.56 bpw on the
            // Q6_K kernels instead of a lossy 8.5-bpw Q8_0 requantization.
            GgmlType::Q3_K => {
                let q6 = crate::quant::q3_k::transcode_to_q6_k(
                    bytes, in_dim as usize * out_dim as usize);
                Some((crate::quant::q6_k::repack_for_matvec(
                    &q6, in_dim as usize, out_dim as usize), GgmlType::Q6_K))
            }
            // BF16 matvec weights (Unsloth's Q8_K_XL keeps `output.weight`
            // and the full-attention `attn_q` in BF16) requantize to Q8_0:
            // gfx906 has no BF16 ALU, `from_gguf` would widen them to F32
            // (4x the bytes, on the non-dp4a matvec), and a Q8_0 step is
            // 1/254 of the block peak — far below what the rest of the
            // file already carries at 8 bits.
