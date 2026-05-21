//! Kernel source compilation cache and HIP module management.
//!
//! HIP kernel sources live as `&'static str` constants embedded in the
//! binary (via `include_str!`). At first use they are compiled with
//! `hipcc --genco --offload-arch=<arch>` to a `.hsaco` blob and cached on
//! disk under `~/.cache/reinstinct/kernels/{xxh3}.hsaco`. The cache key
//! includes the source text, target arch, hipcc version, and flags — any
//! change in any of these forces a recompile, but unchanged sources skip
//! the ~1-2 s `hipcc` invocation.

pub mod kernels;
pub mod prefill;
pub mod qwen35;
pub mod gemma4;
pub mod gemma4_assistant;
pub mod spec_decode;
pub mod smoke;

use std::path::{Path, PathBuf};
use std::process::Command;

use xxhash_rust::xxh3::Xxh3;

/// Default GPU arch this engine targets. Override with
/// `REINSTINCT_OFFLOAD_ARCH=gfx906` etc.
pub const DEFAULT_ARCH: &str = "gfx906";

/// Optimisation flags passed to `hipcc`. Kept in the cache key so a flag
/// change forces a recompile.
pub const COMPILE_FLAGS: &[&str] = &["-O3", "-std=c++17"];

/// Filesystem-backed compile cache for HIP kernel sources.
pub struct KernelCache {
    cache_dir: PathBuf,
    arch: String,
    /// Output of `hipcc --version`, used in the cache key so a HIP upgrade
    /// invalidates compiled blobs automatically.
    hipcc_version: String,
}

impl KernelCache {
    /// Build a cache under `~/.cache/reinstinct/kernels/`. Resolves and
    /// pins the hipcc version up-front so repeated lookups are cheap.
    /// `REINSTINCT_HIPCC_VERSION` overrides the probe — needed when
    /// running under rocprof, whose LD_PRELOAD makes hipcc/clang abort
    /// (LLVM CLI option double-registration). With the cache warm this
    /// is the only hipcc call, so bypassing it is enough.
    pub fn new() -> Result<Self, String> {
        let arch = std::env::var("REINSTINCT_OFFLOAD_ARCH")
            .unwrap_or_else(|_| DEFAULT_ARCH.to_string());
        let cache_dir = Self::default_cache_dir()?;
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| format!("create cache dir {}: {e}", cache_dir.display()))?;
        let hipcc_version = match std::env::var("REINSTINCT_HIPCC_VERSION") {
            Ok(v) => v,
            Err(_) => Self::hipcc_version()?,
        };
        Ok(Self { cache_dir, arch, hipcc_version })
    }

    fn default_cache_dir() -> Result<PathBuf, String> {
        let base = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .ok_or_else(|| "neither XDG_CACHE_HOME nor HOME is set".to_string())?;
        Ok(base.join("reinstinct").join("kernels"))
    }

    fn hipcc_version() -> Result<String, String> {
        let out = Command::new("hipcc")
            .arg("--version")
            .output()
            .map_err(|e| format!("invoke hipcc --version: {e}"))?;
        if !out.status.success() {
            return Err(format!("hipcc --version exited {}", out.status));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    pub fn arch(&self) -> &str { &self.arch }
    pub fn cache_dir(&self) -> &Path { &self.cache_dir }

    /// Hex digest covering source + arch + flags + hipcc version. Forms
    /// the filename for the cached `.hsaco`.
    fn cache_key(&self, source: &str) -> String {
        let mut h = Xxh3::new();
        h.update(source.as_bytes());
        h.update(b"\0arch=");
        h.update(self.arch.as_bytes());
        h.update(b"\0flags=");
        for f in COMPILE_FLAGS {
            h.update(f.as_bytes());
            h.update(b" ");
        }
        h.update(b"\0hipcc=");
        h.update(self.hipcc_version.as_bytes());
        format!("{:016x}", h.digest())
    }

    /// Return the path to a `.hsaco` for `source`, compiling on miss.
    /// `name` is used purely for cosmetic file naming and error messages.
    pub fn compile(&self, name: &str, source: &str) -> Result<PathBuf, String> {
        let key = self.cache_key(source);
        let stem = format!("{name}-{key}");
        let out = self.cache_dir.join(format!("{stem}.hsaco"));
        if out.exists() { return Ok(out); }

        // Per-invocation tmp filenames so concurrent compiles of the same
        // kernel don't collide. Whoever wins the rename publishes the
        // .hsaco; the loser just discards its temp output.
        let pid = std::process::id();
        let tid = std::thread::current().id();
        let unique = format!("{stem}.p{pid}.t{tid:?}");
        let src_path = self.cache_dir.join(format!("{unique}.cpp"));
        std::fs::write(&src_path, source)
            .map_err(|e| format!("write kernel source {}: {e}", src_path.display()))?;

        let tmp_out = self.cache_dir.join(format!("{unique}.hsaco.tmp"));
        let mut cmd = Command::new("hipcc");
        cmd.arg("--genco")
           .arg(format!("--offload-arch={}", self.arch))
           .args(COMPILE_FLAGS)
           .arg("-o").arg(&tmp_out)
           .arg(&src_path);
        let started = std::time::Instant::now();
        let result = cmd.output()
            .map_err(|e| format!("invoke hipcc for {name}: {e}"))?;
        let elapsed_ms = started.elapsed().as_millis();
        let _ = std::fs::remove_file(&src_path);
        if !result.status.success() {
            let _ = std::fs::remove_file(&tmp_out);
            return Err(format!(
                "hipcc failed for {name} (exit {}, {} ms):\n  cmd: {cmd:?}\n  stderr:\n{}",
                result.status, elapsed_ms,
                String::from_utf8_lossy(&result.stderr)));
        }

        // Race-tolerant publish: rename our tmp into place; if a peer beat
        // us to it, the final file already exists and we just clean up.
        match std::fs::rename(&tmp_out, &out) {
            Ok(()) => {
                eprintln!("kernel-cache: compiled {name} ({} ms) → {}", elapsed_ms, out.display());
            }
            Err(_) if out.exists() => {
                let _ = std::fs::remove_file(&tmp_out);
            }
            Err(e) => return Err(format!("rename {} → {}: {e}", tmp_out.display(), out.display())),
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_differs_with_arch() {
        let mut k1 = KernelCache { cache_dir: PathBuf::from("/tmp"), arch: "gfx906".into(),
                                   hipcc_version: "v".into() };
        let h_a = k1.cache_key("kernel x");
        k1.arch = "gfx1030".into();
        let h_b = k1.cache_key("kernel x");
        assert_ne!(h_a, h_b);
    }

    #[test]
    fn cache_key_differs_with_source() {
        let k = KernelCache { cache_dir: PathBuf::from("/tmp"), arch: "gfx906".into(),
                              hipcc_version: "v".into() };
        assert_ne!(k.cache_key("a"), k.cache_key("b"));
    }
}
