# Source provenance

This release contains source and public configuration only.

## Paper version

The setting catalog was checked against the PDF supplied on 2026-09-29. The PDF
metadata reports a creation/modification date of 2026-09-26. In that version Table 10
uses GPT-6 Astra, all ten LIBERO-10 tasks, and seeds 0–4; its no-tactile condition
removes force-related information and gripper state.

## LIBERO

- Upstream: `https://github.com/Lifelong-Robot-Learning/LIBERO`
- Base commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Public patch: camera-only environment support.
- Expected patched tree: `8b1c5e7faed6c486cecc9453be8b23e4cd0d25f5`
- The random-yaw task patch is not applied or distributed.

## H800 source snapshots

The latest paper tactile runner was synchronized from
`tasks/astra-high-libero10-full-vs-strict-s01234-r1` on 2026-09-25. The source
was relocated into the three public modules; machine paths, private providers, and
obsolete compatibility health checks were parameterized or removed.

On H800, the Table 10 Full condition reused the already completed 50-cell Astra
baseline and the new Strict condition ran at concurrency four. A clean clone has no
historical Full batch, so the public tactile entry can construct both conditions in
parallel at two cells per condition, preserving a total concurrency of four. This is
a scheduling adaptation only; the task/seed matrix, observations, actions, deadline,
model settings, and per-cell Harness are unchanged.

Pre-relocation SHA-256 values:

| Source role | SHA-256 |
| --- | --- |
| native Codex cell | `e6b2d67ca182d3270b3540610f9aeb134762d619d51610d400e97350848998a5` |
| native Claude acquisition cell | `beb8ef745069f112b60cce8450e5a77416e2ca45874814166eac3c6b24b41e65` |
| Codex Program Prior cell | `ce07c8aa84e906353af9f1f0e9f1272d8d5a0e5c070b8e1708d8d71ba376d0c5` |
| Claude Program Prior cell | `dac63174976b84fb06907f3623c3381c796faee91a0b029d6c668dc662767750` |
| Codex text-action cell | `82918dd25564da6e36b270ffee9731e7fab3a25029c18e79b1be6034fd66651b` |
| Claude text-action cell | `96be8cb827ca5f8b715e40c3cf1925ece3bb6569bca73a83cf26276798f7b643` |
| Codex evolution cell | `cf67c91e09a1473999dcb841c42c7dbfaf183d0311c80ccc02ee56e162bd226c` |
| full trusted runtime | `66512781e4811e7f05912e93b9c6434da41137545a546aa6a0023161dd72cc3a` |
| strict trusted runtime | `7e1fdd0768890e563b7bb53150ae77ff6b022b6bee565b3f048754ce00bda9a2` |
| latest Codex scheduler | `759003f9420598a590b0478383236deb06d7cd852eb284034f386a71977ded67` |
| Claude scheduler | `8e2355d9e913c54112dbe4a3a1bdfeb8f2ad38ecf1581583ee33d31e913ed503` |
| Top-K orchestrator | `0d0be9cbe7daae9e48143dc31fcf6a7d01b9dab80f0c0ccd0db8b2c1729703d1` |
| evolution orchestrator | `26b76fc23a9a22b603720554bb2e064b6e2b47eaf8ed56964a3ca216588bdfb4` |
| tactile suite orchestrator | `83b864d5780c6e9f7af11dcaf45b4d0800308327e703a0d1dd61378fc7033548` |
| network-audit rerunner | `4653b035cf22bcbef798c216b7f2ce1523ce66a20be7abe8573a1c3c489579d7` |

Relocation changes cover repository-relative imports, public executable/auth
configuration, runtime-state redirection to `.runtime/`, removal of private provider
values, and wiring the frozen sources into the three-module layout. Release validation
also fixes a missing network-rerun import and makes the existing Claude reviewer flag
effective. Single-cell control flow and termination semantics are kept; public-only
source preparation and the tactile scheduling adaptation are documented above.

## Exclusions

Excluded: credentials, provider endpoints, SSH/tunnel configuration, machine-specific
paths, runtime workspaces, logs, JSONL ledgers, videos, results, generated experience,
generated skills, demos, Kimi integration, DeepSeek Harness, and the unrelated portable
ledger/MCP implementation.
