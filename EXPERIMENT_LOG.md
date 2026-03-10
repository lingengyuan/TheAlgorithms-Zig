# Vibe Coding Experiment Log / Vibe Coding 实验日志

This file is the bilingual index for experiment tracking in this repository.
本文件是本仓库实验记录的双语索引页。

The detailed long-form records are split under `docs/experiment_logs/` to keep the root log reviewable.
详细长日志已拆分到 `docs/experiment_logs/` 下，以保持根日志文件可审阅、可维护。

Reference Project for Behavior Alignment / 行为对齐参考项目:
- `https://github.com/TheAlgorithms/Python`

## Logging Policy / 记录规则

- Record algorithm implementation quality with a focus on functional correctness, edge-case robustness, and consistency with Python reference behavior.
- 记录重点聚焦于算法功能正确性、极端/边界情况鲁棒性，以及与 Python 参考实现的一致性。
- Keep `EXPERIMENT_LOG.md` as the bilingual summary/index when logs grow large; move detailed records into linked date-based files under `docs/experiment_logs/phase5/by-date/`.
- 当日志变长时，将 `EXPERIMENT_LOG.md` 保持为双语摘要/索引，并把详细记录拆分到 `docs/experiment_logs/phase5/by-date/` 下按日期组织的链接文件中。
- Each batch entry must truthfully record failures, root causes, fixes, and post-fix verification in the relevant dated log file.
- 每个批次条目都必须在对应日期日志文件中如实记录失败现象、根因、修复措施和修复后验证结果。
- New experiment-log content must be bilingual (English + Simplified Chinese).
- 新增实验日志内容必须为双语（英文 + 简体中文）。

## Current Status / 当前状态

- `build.zig` registered algorithms: `832`
- `build.zig` 已注册算法数：`832`
- Portable target after Phase 6 scope reconciliation: `916`
- 第 6 阶段范围校准后的可移植目标总量：`916`
- Effective completed count under plan-category caps: `824`
- 按计划分类上限口径的有效完成数：`824`
- Remaining planned gap: `92`
- 剩余计划缺口：`92`

## Phase 5 Logs / 第 5 阶段日志

- Overview / 总览: [`docs/experiment_logs/phase5/overview.md`](docs/experiment_logs/phase5/overview.md)
- 2026-03-04: [`docs/experiment_logs/phase5/by-date/2026-03-04.md`](docs/experiment_logs/phase5/by-date/2026-03-04.md)
- 2026-03-05: [`docs/experiment_logs/phase5/by-date/2026-03-05.md`](docs/experiment_logs/phase5/by-date/2026-03-05.md)
- 2026-03-06: [`docs/experiment_logs/phase5/by-date/2026-03-06.md`](docs/experiment_logs/phase5/by-date/2026-03-06.md)
- 2026-03-07: [`docs/experiment_logs/phase5/by-date/2026-03-07.md`](docs/experiment_logs/phase5/by-date/2026-03-07.md)
- 2026-03-08: [`docs/experiment_logs/phase5/by-date/2026-03-08.md`](docs/experiment_logs/phase5/by-date/2026-03-08.md)
- 2026-03-09: [`docs/experiment_logs/phase5/by-date/2026-03-09.md`](docs/experiment_logs/phase5/by-date/2026-03-09.md)

## Phase 6 Logs / 第 6 阶段日志

- Overview / 总览: [`docs/experiment_logs/phase6/overview.md`](docs/experiment_logs/phase6/overview.md)
- 2026-03-09: [`docs/experiment_logs/phase6/by-date/2026-03-09.md`](docs/experiment_logs/phase6/by-date/2026-03-09.md)
- 2026-03-10: [`docs/experiment_logs/phase6/by-date/2026-03-10.md`](docs/experiment_logs/phase6/by-date/2026-03-10.md)

## Migration Note / 迁移说明

- On 2026-03-09, the original monolithic `EXPERIMENT_LOG.md` was first split into batch files and then normalized into date-based detailed logs with bilingual historical content.
- 自 2026-03-09 起，原先单文件的 `EXPERIMENT_LOG.md` 先按 Batch 拆分，随后进一步整理为按日期拆分的详细双语日志。
