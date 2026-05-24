# reinstinct bench-all  (2026-05-24 23:19:21)

Hardware: unknown

| Model                  | Decode tok/s | Prefill ms/tok |
| ---------------------- | ----------: | -------------: |
| gemma-31B              |       27.5 |           8.43 |
| gemma-26B-A4B-MoE      |       86.5 |           2.35 |
| gemma-E4B-PLE          |       96.6 |           1.94 |
| qwen-3.5-0.8B          |      275.9 |           0.63 |
| qwen-3.5-4B            |      111.2 |           1.86 |
| qwen-3.5-4B-MTP        |      112.9 |           1.71 |
| qwen-3.5-27B           |       28.1 |           8.43 |
| qwen-3.5-35B-A3B-MoE   |      101.3 |           1.87 |
| qwen-3.6-27B           |       20.2 |           8.33 |
| qwen-3.6-27B-MTP       |       28.4 |           8.15 |
| qwen-3.6-35B-A3B-MoE   |       93.5 |           2.05 |

## mtp-gen on Gemma 31B (the one model where MTP wins on MI50)

| Prompt                         |   K |      tok/s |     Accept |
| ------------------------------ | --: |  --------: |   -------: |
| What is 17 times 23?           |   1 |       16.5 |        73% |
| What is 17 times 23?           |   2 |       24.9 |        73% |
| What is 17 times 23?           |   3 |       26.7 |        67% |
| What is 17 times 23?           |   4 |       30.3 |        71% |
| List 5 prime numbers           |   1 |       21.1 |        81% |
| List 5 prime numbers           |   2 |       28.5 |        89% |
| List 5 prime numbers           |   3 |       31.8 |        85% |
| List 5 prime numbers           |   4 |       30.6 |        72% |
| Capital of France?             |   1 |       22.3 |        92% |
| Capital of France?             |   2 |       28.2 |        88% |
| Capital of France?             |   3 |       32.8 |        89% |
| Capital of France?             |   4 |       30.1 |        71% |
| Explain how to make tea        |   1 |       20.7 |        78% |
| Explain how to make tea        |   2 |       25.0 |        72% |
| Explain how to make tea        |   3 |       25.7 |        63% |
| Explain how to make tea        |   4 |       27.4 |        62% |
| Write a haiku about programmin |   1 |       20.3 |        75% |
| Write a haiku about programmin |   2 |       21.7 |        56% |
| Write a haiku about programmin |   3 |       23.6 |        55% |
| Write a haiku about programmin |   4 |       22.2 |        46% |

Done.
