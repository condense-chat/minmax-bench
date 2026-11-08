# What data went into the benchmark?

The committed cost runs replay software-engineering conversations from SWE-chat. Their manifests identify the dataset selector and the sessions actually included. The quality guide covers a separate Terminal-Bench task suite.

| Saved run | Dataset selector | Sessions in manifest | Scored Condense turns | Input limit |
| --- | --- | --- | --- | --- |
| [Opus 4.8](../runs/run-5c61ab52-8eea-4fee-97a4-5c64ee5344af/README.md) | `swe-chat:64` | 64 | 11,712 | No configured cap |
| [Haiku 4.5, capped](../runs/run-202f98bd-a2f1-4390-8307-658b451b7727/README.md) | `swe-chat:32` | 6 | 1,600 | 190,000 tokens |
| [Haiku 4.5, uncapped](../runs/run-cba32b86-99ba-4ed7-bf7c-e385edf2ec99/README.md) | `swe-chat:32` | 6 | 2,909 | No configured cap |

The Haiku manifests list the same six session IDs. Removing the cap exposes later turns in those conversations. A turn is one request point and many turns belong to the same session. Adding the rows would double-count sessions.

## What ships here

| Artifact | Contents |
| --- | --- |
| [Saved runs](../runs/) | Session IDs, settings, per-turn usage, and derived report tables |
| [Request template](cc_request_template.json) | Claude Code request structure used in reconstruction |
| [Caveman fixtures](caveman/README.txt) | Pinned rulesets shared by the quality modes |
| [Dataset loaders](../minmax_bench/data/) | Readers for built-in samples, SWE-chat, local agent logs, and normalized JSONL |

## Try another workload

Use `sample` for a built-in example. For SWE-chat, install the optional dataset dependencies with `uv sync --extra hf`, then choose `swe-chat:N` in the cost wizard. For your own recorded sessions, use `claude-code:/path/*.jsonl`, a Codex or OpenCode session loader, or normalized JSONL. Inspect the run’s actual session IDs and matched turns before comparing it with a reference run.

[Cost measurement](../docs/cost.md) · [Quality task selection](../docs/quality.md)
