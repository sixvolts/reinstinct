# llama-server shutdown hang on a second SIGTERM (upstream bug, not fork-specific)

Seen in the S1 R15 run (baselines, "Harness note"): llama-server hung for good after "Received second interrupt, terminating immediately.", with 4 threads in futex/KFD waits, until SIGKILL. Reproduced and root-caused 2026-10-04 on podcast GPU 1, fork + R15 build, with gdb 17.1 run from a local unpack (no system changes).

## Cause

`tools/server/server.cpp` `signal_handler` calls `exit(1)` on the second SIGINT/SIGTERM. That line dates from upstream a693bea1e ("server : hit Ctrl+C twice to exit", 2024-02-28) and is still on upstream master. `exit()` is not async-signal-safe. The main thread's stack at the hang, bottom to top:

```
~common_init_result -> llama_free -> ~llama_context
  -> ggml_backend_cuda_free -> ~ggml_backend_cuda_context      (or ggml_backend_cuda_host_buffer_free_buffer)
  -> libamdhip64 -> libhsa-runtime64 (takes a runtime mutex) -> libhsakmt -> munmap
  <signal handler called>                                       second SIGTERM lands here
  -> signal_handler -> exit -> __call_tls_dtors
  -> libamdhip64 TLS destructor -> libhsa-runtime64 (same function, same call chain) -> pthread_mutex_lock   <- blocks forever
```

The first signal starts normal cleanup, which frees HIP/HSA resources. If the second signal arrives while the main thread holds the hsa-runtime lock (cleanup takes about 100 ms), `exit()` runs HIP's thread-local destructor on that same thread, and it waits on the non-recursive lock the interrupted frame already holds. The other threads are idle: the common_log worker waits on a condvar, and two HSA event threads wait in `kfd_wait_on_events`.

## Repro

Serve 2 completions, then send two SIGTERMs back to back. The S1 harness did exactly that: `kill $pid; pkill -x llama-server`, where the pkill /proc scan puts the second signal a few ms after the first.

| build | model | second signal | result |
|---|---|---|---|
| R15 | 35B-A3B | back to back (S1, x2) | hang x2 |
| R15 | 35B-A3B | back to back (gdb, x2) | hang x2 |
| R15 | 27B | back to back (gdb, x2) | hang x1, exit(1) x1 |
| R15 | 27B | back to back (S1 MTP section, x4) | signals coalesced, clean exit |
| R15 | 35B-A3B | +2 s / +0.3 s | first cleanup finished first, clean exit |
| R15 + `_exit` fix | 35B-A3B x3 (0 / 0 / +50 ms), 27B x2 (0 / +50 ms) | second interrupt hit every time | exit code 1 at once, x5 |
| R15 + `_exit` fix | 35B-A3B, +100 ms | first cleanup finished first | clean exit |

Whether it hangs is a timing race, not a property of the model. A single SIGTERM always exits cleanly in about 0.1 s. Only a second signal inside the cleanup window can hang, so systemd/docker stop (one SIGTERM, then SIGKILL after the timeout) cannot hang forever. Ctrl+C twice, or scripts that signal twice, can.

## Fix

patches/upstream/server-second-interrupt-_exit.diff: `_exit(1)` instead of `exit(1)`, plus `<unistd.h>` on non-Windows (MSVC declares `_exit` in `<stdlib.h>`). This "terminate immediately" path doesn't need the destructors, and the KFD driver frees GPU resources at process exit. The `fprintf` before it isn't async-signal-safe either (stderr FILE lock); a `write(2, ...)` would remove that rarer window too. Not applied to the fork and not sent upstream.

Harness side: the S1 serve() helper now SIGKILLs after 60 s. Sending one SIGTERM and waiting avoids the race entirely.
