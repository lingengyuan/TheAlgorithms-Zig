# Phase 6 Experiment Overview / 第 6 阶段实验总览

This file keeps the Phase 6 scope decisions, accounting basis, and cross-batch logging rules.
本文件记录第 6 阶段的范围决策、统计口径与跨批次日志规则。

## Ongoing Logging Scope / 持续记录范围

For each batch/review cycle, only record:
对于每个批次或评审周期，只记录：
- failing command/step,
- 失败的命令或步骤，
- error symptom,
- 错误现象，
- root cause,
- 根因，
- fix applied,
- 修复措施，
- post-fix verification result.
- 修复后验证结果。

## Phase 6 Scope Reconciliation / 第 6 阶段范围校准 (2026-03-09)

Scope / 范围:
- [`phase6-plan.md`](/root/projects/plans/TheAlgorithms-Zig/phase6-plan.md)
- [`phase6-execution-guideline.md`](/root/projects/plans/TheAlgorithms-Zig/phase6-execution-guideline.md)
- [`README.md`](/root/projects/TheAlgorithms-Zig/README.md)
- [`EXPERIMENT_LOG.md`](/root/projects/TheAlgorithms-Zig/EXPERIMENT_LOG.md)

Result / 结果:
- Excluded `audio_filters/show_response.py` from the portable target because it is a visualization helper built around `numpy` and `matplotlib`, not a pure portable algorithm module.
- 将 `audio_filters/show_response.py` 排除出可移植目标，因为它本质上是基于 `numpy` 与 `matplotlib` 的可视化辅助脚本，而不是纯可移植算法模块。
- Portable target total updated from `929` to `928`.
- 可移植目标总量从 `929` 调整为 `928`。
- Current checkpoint accounting after Batch A Wave 1 integration:
- Batch A 第 1 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `748`
  - `build.zig` 已注册算法数：`748`
  - effective completed count under plan-category caps: `740`
  - 按计划分类上限口径的有效完成数：`740`
  - remaining planned gap: `188`
  - 剩余计划缺口：`188`

## Batch A Reconciliation Update / Batch A 口径修正更新 (2026-03-09)

Result / 结果:
- Added the four real remaining `searches` algorithms: `fibonacci_search`, `hill_climbing`, `simulated_annealing`, and `tabu_search`.
- 已补齐 `searches` 分类中 4 个真实剩余算法：`fibonacci_search`、`hill_climbing`、`simulated_annealing` 与 `tabu_search`。
- Confirmed that `bit_manipulation` has no real missing file under the current Python-reference snapshot; the earlier `remaining 1` was a category-count drift.
- 确认在当前 Python 参考快照下，`bit_manipulation` 没有真实缺项；之前的“剩余 1”属于分类统计漂移。
- Current checkpoint accounting after Batch A Wave 2 integration:
- Batch A 第 2 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `752`
  - `build.zig` 已注册算法数：`752`
  - effective completed count under plan-category caps: `744`
  - 按计划分类上限口径的有效完成数：`744`
  - remaining planned gap: `184`
  - 剩余计划缺口：`184`

## Batch B Wave 3 Update / Batch B 第 3 波更新 (2026-03-09)

Result / 结果:
- Added seven low-risk `dynamic_programming` files: `factorial`, `fibonacci`, `minimum_coin_change`, `floyd_warshall`, `narcissistic_number`, `sum_of_subset`, and `wildcard_matching`.
- 新增 7 个低风险 `dynamic_programming` 文件：`factorial`、`fibonacci`、`minimum_coin_change`、`floyd_warshall`、`narcissistic_number`、`sum_of_subset`、`wildcard_matching`。
- Current checkpoint accounting after Batch B Wave 3 integration:
- Batch B 第 3 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `759`
  - `build.zig` 已注册算法数：`759`
  - effective completed count under plan-category caps: `751`
  - 按计划分类上限口径的有效完成数：`751`
  - remaining planned gap: `177`
  - 剩余计划缺口：`177`

## Batch B Wave 4 Update / Batch B 第 4 波更新 (2026-03-09)

Result / 结果:
- Added and registered five remaining portable `dynamic_programming` files: `all_construct`, `minimum_steps_to_one`, `smith_waterman`, `subset_generation`, and `viterbi`.
- 已新增并注册 5 个剩余的可移植 `dynamic_programming` 文件：`all_construct`、`minimum_steps_to_one`、`smith_waterman`、`subset_generation` 与 `viterbi`。
- Excluded `dynamic_programming/k_means_clustering_tensorflow.py` from the portable target because the module is centered on TensorFlow-specific clustering infrastructure rather than a pure portable algorithm implementation.
- 因 `dynamic_programming/k_means_clustering_tensorflow.py` 主要依赖 TensorFlow 聚类基础设施，而不是纯可移植算法实现，已将其排除出可移植目标统计口径。
- Current checkpoint accounting after Batch B Wave 4 integration:
- Batch B 第 4 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `764`
  - `build.zig` 已注册算法数：`764`
  - effective completed count under plan-category caps: `756`
  - 按计划分类上限口径的有效完成数：`756`
  - remaining planned gap: `171`
  - 剩余计划缺口：`171`

## Batch B Wave 5 Reconciliation Update / Batch B 第 5 波口径修正更新 (2026-03-09)

Result / 结果:
- Reconciled `ciphers` against the current local Python-reference snapshot and confirmed there are no real missing files in that category.
- 已将 `ciphers` 与当前本地 Python 参考快照完成对账，并确认该分类没有真实缺项。
- Wrote off the earlier `remaining 11` in `ciphers` as category-count drift rather than implementation work.
- 将此前 `ciphers` 的“剩余 11”核销为分类统计漂移，而非实现缺口。
- Current checkpoint accounting after Batch B Wave 5 reconciliation:
- Batch B 第 5 波口径修正后的当前检查点统计：
  - `build.zig` registered algorithms: `764`
  - `build.zig` 已注册算法数：`764`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `756`
  - 按计划分类上限口径的有效完成数：`756`
  - remaining planned gap: `160`
  - 剩余计划缺口：`160`

## Batch C Wave 1 Update / Batch C 第 1 波更新 (2026-03-09)

Result / 结果:
- Added and registered seven low-risk `strings` files: `check_anagrams`, `edit_distance`, `barcode_validator`, `credit_card_validator`, `indian_phone_validator`, `is_srilankan_phone_number`, and `is_valid_email_address`.
- 已新增并注册 7 个低风险 `strings` 文件：`check_anagrams`、`edit_distance`、`barcode_validator`、`credit_card_validator`、`indian_phone_validator`、`is_srilankan_phone_number` 与 `is_valid_email_address`。
- Current checkpoint accounting after Batch C Wave 1 integration:
- Batch C 第 1 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `771`
  - `build.zig` 已注册算法数：`771`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `763`
  - 按计划分类上限口径的有效完成数：`763`
  - remaining planned gap: `153`
  - 剩余计划缺口：`153`

## Batch C Wave 2 Update / Batch C 第 2 波更新 (2026-03-09)

Result / 结果:
- Added and registered the remaining `strings` files from the current local Python-reference snapshot, completing the category reconciliation.
- 已新增并注册当前本地 Python 参考快照下剩余的 `strings` 文件，完成该分类的对账收口。
- Current checkpoint accounting after Batch C Wave 2 integration:
- Batch C 第 2 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `785`
  - `build.zig` 已注册算法数：`785`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `777`
  - 按计划分类上限口径的有效完成数：`777`
  - remaining planned gap: `139`
  - 剩余计划缺口：`139`

## Batch D Wave 1 Update / Batch D 第 1 波更新 (2026-03-10)

Result / 结果:
- Added and registered 28 `graphs` compatibility-entry files that map Python filenames to already-existing Zig implementations.
- 已新增并注册 28 个 `graphs` 兼容入口文件，用于把 Python 文件名与已存在的 Zig 实现对齐。
- Reduced the real remaining `graphs` gap from a naming-drift-heavy inventory to only 8 true missing algorithms.
- 将 `graphs` 的真实剩余缺口压缩到仅剩 8 个真正缺实现的算法，基本清除了命名漂移造成的统计噪音。
- Current checkpoint accounting after Batch D Wave 1 integration:
- Batch D 第 1 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `813`
  - `build.zig` 已注册算法数：`813`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `805`
  - 按计划分类上限口径的有效完成数：`805`
  - remaining planned gap: `111`
  - 剩余计划缺口：`111`

## Batch D Wave 2 Update / Batch D 第 2 波更新 (2026-03-10)

Result / 结果:
- Added and registered two real missing `graphs` modules: `basic_graphs` and `edmonds_karp_multiple_source_and_sink`.
- 已新增并注册两个真实缺失的 `graphs` 模块：`basic_graphs` 与 `edmonds_karp_multiple_source_and_sink`。
- Reduced the real remaining `graphs` gap to 6 files: `ant_colony_optimization_algorithms`, `bidirectional_a_star`, `directed_and_undirected_weighted_graph`, `frequent_pattern_graph_miner`, `lanczos_eigenvectors`, and `multi_heuristic_astar`.
- 将 `graphs` 的真实剩余缺口进一步压缩到 6 个文件：`ant_colony_optimization_algorithms`、`bidirectional_a_star`、`directed_and_undirected_weighted_graph`、`frequent_pattern_graph_miner`、`lanczos_eigenvectors` 与 `multi_heuristic_astar`。
- Current checkpoint accounting after Batch D Wave 2 integration:
- Batch D 第 2 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `815`
  - `build.zig` 已注册算法数：`815`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `807`
  - 按计划分类上限口径的有效完成数：`807`
  - remaining planned gap: `109`
  - 剩余计划缺口：`109`

## Batch D Wave 3 Update / Batch D 第 3 波更新 (2026-03-10)

Result / 结果:
- Added and registered two more real missing `graphs` modules: `bidirectional_a_star` and `directed_and_undirected_weighted_graph`.
- 已新增并注册另外两个真实缺失的 `graphs` 模块：`bidirectional_a_star` 与 `directed_and_undirected_weighted_graph`。
- Reduced the real remaining `graphs` gap to 4 files: `ant_colony_optimization_algorithms`, `frequent_pattern_graph_miner`, `lanczos_eigenvectors`, and `multi_heuristic_astar`.
- 将 `graphs` 的真实剩余缺口进一步压缩到 4 个文件：`ant_colony_optimization_algorithms`、`frequent_pattern_graph_miner`、`lanczos_eigenvectors` 与 `multi_heuristic_astar`。
- Current checkpoint accounting after Batch D Wave 3 integration:
- Batch D 第 3 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `817`
  - `build.zig` 已注册算法数：`817`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `809`
  - 按计划分类上限口径的有效完成数：`809`
  - remaining planned gap: `107`
  - 剩余计划缺口：`107`

## Batch D Wave 4 Update / Batch D 第 4 波更新 (2026-03-10)

Result / 结果:
- Added and registered two more real missing `graphs` modules: `ant_colony_optimization_algorithms` and `frequent_pattern_graph_miner`.
- 已新增并注册另外两个真实缺失的 `graphs` 模块：`ant_colony_optimization_algorithms` 与 `frequent_pattern_graph_miner`。
- Reduced the real remaining `graphs` gap to 2 files: `lanczos_eigenvectors` and `multi_heuristic_astar`.
- 将 `graphs` 的真实剩余缺口进一步压缩到 2 个文件：`lanczos_eigenvectors` 与 `multi_heuristic_astar`。
- Current checkpoint accounting after Batch D Wave 4 integration:
- Batch D 第 4 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `819`
  - `build.zig` 已注册算法数：`819`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `811`
  - 按计划分类上限口径的有效完成数：`811`
  - remaining planned gap: `105`
  - 剩余计划缺口：`105`

## Batch D Wave 5 Update / Batch D 第 5 波更新 (2026-03-10)

Result / 结果:
- Added and registered one more real missing `graphs` module: `multi_heuristic_astar`.
- 已新增并注册另外一个真实缺失的 `graphs` 模块：`multi_heuristic_astar`。
- Reduced the real remaining `graphs` gap to 1 file: `lanczos_eigenvectors`.
- 将 `graphs` 的真实剩余缺口进一步压缩到 1 个文件：`lanczos_eigenvectors`。
- Current checkpoint accounting after Batch D Wave 5 integration:
- Batch D 第 5 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `820`
  - `build.zig` 已注册算法数：`820`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `812`
  - 按计划分类上限口径的有效完成数：`812`
  - remaining planned gap: `104`
  - 剩余计划缺口：`104`

## Batch D Wave 6 Update / Batch D 第 6 波更新 (2026-03-10)

Result / 结果:
- Added and registered the last real missing `graphs` module: `lanczos_eigenvectors`.
- 已新增并注册最后一个真实缺失的 `graphs` 模块：`lanczos_eigenvectors`。
- Reduced the real remaining `graphs` gap to `0`; Batch D is now complete.
- 将 `graphs` 的真实剩余缺口压缩到 `0`；Batch D 已完成收口。
- Current checkpoint accounting after Batch D Wave 6 integration:
- Batch D 第 6 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `821`
  - `build.zig` 已注册算法数：`821`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `813`
  - 按计划分类上限口径的有效完成数：`813`
  - remaining planned gap: `103`
  - 剩余计划缺口：`103`

## Batch E Wave 1 Update / Batch E 第 1 波更新 (2026-03-10)

Result / 结果:
- Started Batch E with three low-risk `maths` modules: `least_common_multiple`, `extended_euclidean_algorithm`, and `print_multiplication_table`.
- 以三个低风险 `maths` 模块启动 Batch E：`least_common_multiple`、`extended_euclidean_algorithm` 与 `print_multiplication_table`。
- Reduced the remaining planned gap to `100`.
- 将剩余计划缺口压到 `100`。
- Current checkpoint accounting after Batch E Wave 1 integration:
- Batch E 第 1 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `824`
  - `build.zig` 已注册算法数：`824`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `816`
  - 按计划分类上限口径的有效完成数：`816`
  - remaining planned gap: `100`
  - 剩余计划缺口：`100`

## Batch E Wave 2 Update / Batch E 第 2 波更新 (2026-03-10)

Result / 结果:
- Added one more low-risk `maths` module: `sin`.
- 已新增一个低风险 `maths` 模块：`sin`。
- Reduced the remaining planned gap to `99`.
- 将剩余计划缺口压到 `99`。
- Current checkpoint accounting after Batch E Wave 2 integration:
- Batch E 第 2 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `825`
  - `build.zig` 已注册算法数：`825`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `817`
  - 按计划分类上限口径的有效完成数：`817`
  - remaining planned gap: `99`
  - 剩余计划缺口：`99`

## Batch E Wave 3 Update / Batch E 第 3 波更新 (2026-03-10)

Result / 结果:
- Completed the low-risk `maths` pass by adding `allocation_number`, `entropy`, `euler_method`, `euler_modified`, `hardy_ramanujanalgo`, `area`, and `volume`.
- 通过新增 `allocation_number`、`entropy`、`euler_method`、`euler_modified`、`hardy_ramanujanalgo`、`area` 与 `volume`，完成了低风险 `maths` 批次。
- Reduced the remaining planned gap to `92`.
- 将剩余计划缺口压到 `92`。
- Current checkpoint accounting after Batch E Wave 3 integration:
- Batch E 第 3 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `832`
  - `build.zig` 已注册算法数：`832`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `824`
  - 按计划分类上限口径的有效完成数：`824`
  - remaining planned gap: `92`
  - 剩余计划缺口：`92`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - manual Phase 6 inventory reconciliation against Python category contents and portable-target rules
  - Symptom / 现象: `audio_filters/show_response.py` inflated the algorithm target even though the module mainly wrapped visualization-only behavior.
  - 中文说明：`audio_filters/show_response.py` 会抬高算法目标，但该模块主体其实只是可视化行为封装。
  - Root Cause / 根因: the earlier target accounting treated every Python file in the category as equally portable, without separating pure algorithms from `numpy`/`matplotlib` tooling scripts.
  - 中文说明：此前的目标统计把分类内所有 Python 文件都视作同等可移植，没有把纯算法与依赖 `numpy`/`matplotlib` 的工具脚本区分开。
  - Fix Applied / 修复措施: removed `show_response.py` from the portable target basis and updated the Phase 6 plan plus repository-facing accounting documents to use the corrected total.
  - 中文说明：将 `show_response.py` 从可移植目标基线中移除，并同步更新了 Phase 6 计划与仓库内的统计文档。
  - Post-Fix Verification / 修复后验证: that checkpoint established the first consistent `928`-algorithm portable-target basis across the repository status page and Phase 6 logs.
  - 中文说明：该检查点首次在仓库状态页与 Phase 6 日志之间统一了 `928` 个可移植算法的统计口径。

## Batch E Wave 4 Update / Batch E 第 4 波更新 (2026-03-10)

Result / 结果:
- Added four medium-risk `maths` modules: `gamma`, `pollard_rho`, `primelib`, and `solovay_strassen_primality_test`.
- 新增 4 个中风险 `maths` 模块：`gamma`、`pollard_rho`、`primelib` 与 `solovay_strassen_primality_test`。
- Reduced the remaining planned gap to `88`.
- 将剩余计划缺口压到 `88`。
- Current checkpoint accounting after Batch E Wave 4 integration:
- Batch E 第 4 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `836`
  - `build.zig` 已注册算法数：`836`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `828`
  - 按计划分类上限口径的有效完成数：`828`
  - remaining planned gap: `88`
  - 剩余计划缺口：`88`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test maths/solovay_strassen_primality_test.zig`
  - Symptom / 现象: the first draft failed to compile because `aa /= 2` was used on `i128`.
  - 中文说明：初版编译失败，因为在 `i128` 上直接写了 `aa /= 2`。
  - Root Cause / 根因: Zig requires explicit signed-division builtins such as `@divTrunc` for signed integer division.
  - 中文说明：Zig 对有符号整数除法要求显式使用 `@divTrunc` 这类内建函数。
  - Fix Applied / 修复措施: replaced the update with `aa = @divTrunc(aa, 2)`.
  - 中文说明：已将该更新改为 `aa = @divTrunc(aa, 2)`。
  - Post-Fix Verification / 修复后验证: `zig test maths/solovay_strassen_primality_test.zig` passed.
  - 中文说明：修复后 `zig test maths/solovay_strassen_primality_test.zig` 已通过。
- Failing Step/Command / 失败步骤或命令:
  - `zig test maths/primelib.zig`
  - Symptom / 现象: the first draft failed to compile because the public error set omitted allocator-driven `OutOfMemory`.
  - 中文说明：初版编译失败，因为公开错误集遗漏了 allocator 相关的 `OutOfMemory`。
  - Root Cause / 根因: the implementation returned owned slices from several helpers but the declared error set only covered semantic errors.
  - 中文说明：多个辅助函数会返回拥有所有权的切片，但声明的错误集只覆盖了语义错误，没有覆盖分配失败。
  - Fix Applied / 修复措施: added `OutOfMemory` to `PrimeLibError` and kept allocator failures explicit.
  - 中文说明：已将 `OutOfMemory` 加入 `PrimeLibError`，并显式保留分配失败路径。
  - Post-Fix Verification / 修复后验证: `zig test maths/primelib.zig` passed.
  - 中文说明：修复后 `zig test maths/primelib.zig` 已通过。

## Batch E Wave 5 Update / Batch E 第 5 波更新 (2026-03-10)

Result / 结果:
- Added two more deterministic `maths` modules: `bailey_borwein_plouffe` and `simultaneous_linear_equation_solver`.
- 新增 2 个确定性更强的 `maths` 模块：`bailey_borwein_plouffe` 与 `simultaneous_linear_equation_solver`。
- Reduced the remaining planned gap to `86`.
- 将剩余计划缺口压到 `86`。
- Current checkpoint accounting after Batch E Wave 5 integration:
- Batch E 第 5 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `838`
  - `build.zig` 已注册算法数：`838`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `830`
  - 按计划分类上限口径的有效完成数：`830`
  - remaining planned gap: `86`
  - 剩余计划缺口：`86`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test maths/bailey_borwein_plouffe.zig`
  - Symptom / 现象: the first draft failed to compile because `var total = 0.0` inferred `comptime_float` instead of a runtime `f64`.
  - 中文说明：初版编译失败，因为 `var total = 0.0` 被推断成了 `comptime_float`，而不是运行时 `f64`。
  - Root Cause / 根因: the running summation variable lacked an explicit floating-point type annotation.
  - 中文说明：运行时累加变量缺少显式浮点类型标注。
  - Fix Applied / 修复措施: changed the declaration to `var total: f64 = 0.0`.
  - 中文说明：已将声明改为 `var total: f64 = 0.0`。
  - Post-Fix Verification / 修复后验证: `zig test maths/bailey_borwein_plouffe.zig` passed.
  - 中文说明：修复后 `zig test maths/bailey_borwein_plouffe.zig` 已通过。
- Failing Step/Command / 失败步骤或命令:
  - `zig test maths/simultaneous_linear_equation_solver.zig`
  - Symptom / 现象: the first draft failed to compile because the solver error set omitted allocator-driven `OutOfMemory`.
  - 中文说明：初版编译失败，因为求解器错误集遗漏了 allocator 相关的 `OutOfMemory`。
  - Root Cause / 根因: the implementation returned owned buffers but the public error set only declared semantic failures.
  - 中文说明：实现会返回拥有所有权的缓冲区，但公开错误集只声明了语义错误，没有包含分配失败。
  - Fix Applied / 修复措施: added `OutOfMemory` to `SolveError`.
  - 中文说明：已将 `OutOfMemory` 加入 `SolveError`。
  - Post-Fix Verification / 修复后验证: the file compiled and progressed to runtime tests.
  - 中文说明：修复后文件已能编译，并进入运行时测试阶段。
- Failing Step/Command / 失败步骤或命令:
  - `zig test maths/simultaneous_linear_equation_solver.zig`
  - Symptom / 现象: the first extreme-case test failed with `InvalidEquationSet`.
  - 中文说明：第一版极端测试用例以 `InvalidEquationSet` 失败。
  - Root Cause / 根因: the test matrix violated the Python reference precondition that at least one equation row must have no zero coefficients.
  - 中文说明：测试矩阵违反了 Python 参考实现的前提条件，即至少要有一行方程不含零系数。
  - Fix Applied / 修复措施: replaced that case with a dense, numerically skewed but valid system whose solution is still easy to verify.
  - 中文说明：已将该测试替换为系数稠密、数值尺度悬殊但满足前提条件且易于验证答案的线性系统。
  - Post-Fix Verification / 修复后验证: `zig test maths/simultaneous_linear_equation_solver.zig` passed.
  - 中文说明：修复后 `zig test maths/simultaneous_linear_equation_solver.zig` 已通过。

## Batch E Wave 6 Update / Batch E 第 6 波更新 (2026-03-10)

Result / 结果:
- Added two more deterministic `maths` modules: `dual_number_automatic_differentiation` and `qr_decomposition`.
- 新增 2 个确定性更强的 `maths` 模块：`dual_number_automatic_differentiation` 与 `qr_decomposition`。
- Reduced the remaining planned gap to `84`.
- 将剩余计划缺口压到 `84`。
- Current checkpoint accounting after Batch E Wave 6 integration:
- Batch E 第 6 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `840`
  - `build.zig` 已注册算法数：`840`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `832`
  - 按计划分类上限口径的有效完成数：`832`
  - remaining planned gap: `84`
  - 剩余计划缺口：`84`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test maths/qr_decomposition.zig`
  - Symptom / 现象: the first draft failed to compile due to a non-mutated `var q` declaration and a `comptime_float` sign value flowing through runtime control flow.
  - 中文说明：初版编译失败，一处是 `var q` 实际没有重新赋值，另一处是符号值在运行时分支里被推断成了 `comptime_float`。
  - Root Cause / 根因: the implementation mixed slice-content mutation with variable rebinding semantics, and the Householder sign constant lacked an explicit runtime type.
  - 中文说明：实现混淆了“切片内容可变”和“变量重新绑定”这两类语义，同时 Householder 符号常量缺少显式运行时类型。
  - Fix Applied / 修复措施: changed `q` to `const` and annotated the sign as `f64`.
  - 中文说明：已将 `q` 改为 `const`，并把符号显式标注为 `f64`。
  - Post-Fix Verification / 修复后验证: `zig test maths/qr_decomposition.zig` passed.
  - 中文说明：修复后 `zig test maths/qr_decomposition.zig` 已通过。
- Failing Step/Command / 失败步骤或命令:
  - `zig test maths/dual_number_automatic_differentiation.zig`
  - Symptom / 现象: one extreme-case assertion expected `720` but the implementation returned `360`.
  - 中文说明：一个极端测试断言写成了 `720`，而实现返回 `360`。
  - Root Cause / 根因: the test expectation was mathematically wrong; for `0.5 * (y + 3)^6`, the sixth derivative at `y = -3` is `0.5 * 6! = 360`.
  - 中文说明：测试期望本身数学上写错了；对 `0.5 * (y + 3)^6` 来说，在 `y = -3` 处的六阶导应为 `0.5 * 6! = 360`。
  - Fix Applied / 修复措施: corrected the expected value to `360` and tightened the helper to keep using the caller allocator consistently.
  - 中文说明：已将期望值修正为 `360`，并顺手把辅助函数收紧为一致使用调用方 allocator。
  - Post-Fix Verification / 修复后验证: `zig test maths/dual_number_automatic_differentiation.zig` passed.
  - 中文说明：修复后 `zig test maths/dual_number_automatic_differentiation.zig` 已通过。

## Batch E Wave 7 Update / Batch E 第 7 波更新 (2026-03-10)

Result / 结果:
- Added one deterministic `maths` module: `radix2_fft`.
- 新增 1 个确定性 `maths` 模块：`radix2_fft`。
- Reduced the remaining planned gap to `83`.
- 将剩余计划缺口压到 `83`。
- Current checkpoint accounting after Batch E Wave 7 integration:
- Batch E 第 7 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `841`
  - `build.zig` 已注册算法数：`841`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `833`
  - 按计划分类上限口径的有效完成数：`833`
  - remaining planned gap: `83`
  - 剩余计划缺口：`83`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test maths/radix2_fft.zig`
  - Symptom / 现象: the first draft failed to compile because the FFT direction factor inside runtime control flow was inferred as `comptime_float`.
  - 中文说明：初版编译失败，因为 FFT 方向因子在运行时分支中被推断成了 `comptime_float`。
  - Root Cause / 根因: the sign multiplier in the angle computation lacked an explicit runtime `f64` type.
  - 中文说明：角度计算中的符号乘子缺少显式运行时 `f64` 类型。
  - Fix Applied / 修复措施: introduced an explicitly typed `direction: f64` before computing the angle.
  - 中文说明：已先引入显式类型的 `direction: f64`，再计算角度。
  - Post-Fix Verification / 修复后验证: the file compiled and advanced to runtime assertions.
  - 中文说明：修复后文件可以编译，并进入运行时断言阶段。
- Failing Step/Command / 失败步骤或命令:
  - `zig test maths/radix2_fft.zig`
  - Symptom / 现象: two tests failed due to incorrect test assumptions, not due to the FFT implementation itself.
  - 中文说明：有两个测试因测试假设写错而失败，不是 FFT 实现本身出错。
  - Root Cause / 根因: one expected polynomial shift omitted a leading zero coefficient, and another compared the FFT result against an untrimmed naive convolution while the Python reference semantics trim trailing zero coefficients.
  - 中文说明：一个移位多项式期望值漏写了前导零系数，另一个把 FFT 结果与未裁剪尾零的朴素卷积直接比较，而 Python 参考语义会裁剪尾零系数。
  - Fix Applied / 修复措施: corrected the shifted expected coefficients and trimmed the naive comparison baseline to the same semantic form.
  - 中文说明：已修正移位期望系数，并把朴素卷积的对比基线裁剪到相同语义。
  - Post-Fix Verification / 修复后验证: `zig test maths/radix2_fft.zig` passed.
  - 中文说明：修复后 `zig test maths/radix2_fft.zig` 已通过。

## Batch E Wave 8 Update / Batch E 第 8 波更新 (2026-03-10)

Result / 结果:
- Added one seeded-random `maths` module: `monte_carlo_dice`.
- 新增 1 个带固定 seed 测试策略的 `maths` 模块：`monte_carlo_dice`。
- Reduced the remaining planned gap to `82`.
- 将剩余计划缺口压到 `82`。
- Current checkpoint accounting after Batch E Wave 8 integration:
- Batch E 第 8 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `842`
  - `build.zig` 已注册算法数：`842`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `834`
  - 按计划分类上限口径的有效完成数：`834`
  - remaining planned gap: `82`
  - 剩余计划缺口：`82`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test maths/monte_carlo_dice.zig`
  - Symptom / 现象: the first draft failed to compile because the public error set omitted allocator-driven `OutOfMemory`.
  - 中文说明：初版编译失败，因为公开错误集遗漏了 allocator 相关的 `OutOfMemory`。
  - Root Cause / 根因: the function allocates count and probability buffers, but the declared error set only covered semantic invalid-input failures.
  - 中文说明：该函数会分配计数与概率缓冲区，但声明的错误集只覆盖了语义上的非法输入，没有包含分配失败。
  - Fix Applied / 修复措施: added `OutOfMemory` to `MonteCarloDiceError`.
  - 中文说明：已将 `OutOfMemory` 加入 `MonteCarloDiceError`。
  - Post-Fix Verification / 修复后验证: the file compiled and all distribution tests passed.
  - 中文说明：修复后文件可以编译，且所有分布测试均已通过。

## Batch E Wave 9 Update / Batch E 第 9 波更新 (2026-03-10)

Result / 结果:
- Added two more seeded-random `maths` modules: `pi_monte_carlo_estimation` and `monte_carlo`.
- 新增 2 个带固定 seed 测试策略的 `maths` 模块：`pi_monte_carlo_estimation` 与 `monte_carlo`。
- Reduced the remaining planned gap to `80`.
- 将剩余计划缺口压到 `80`。
- Current checkpoint accounting after Batch E Wave 9 integration:
- Batch E 第 9 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `844`
  - `build.zig` 已注册算法数：`844`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `836`
  - 按计划分类上限口径的有效完成数：`836`
  - remaining planned gap: `80`
  - 剩余计划缺口：`80`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test maths/pi_monte_carlo_estimation.zig`
  - Symptom / 现象: the first draft failed to compile because a composite literal method call lacked parentheses.
  - 中文说明：初版编译失败，因为复合字面量调用方法时缺少括号。
  - Root Cause / 根因: Zig requires parentheses around a struct literal before method dispatch in that expression form.
  - 中文说明：在该表达式形式下，Zig 要求先用括号包住结构体字面量，才能继续做方法调用。
  - Fix Applied / 修复措施: wrapped the composite literals in parentheses before calling `isInUnitCircle()`.
  - 中文说明：已在调用 `isInUnitCircle()` 前为复合字面量补上括号。
  - Post-Fix Verification / 修复后验证: `zig test maths/pi_monte_carlo_estimation.zig` passed.
  - 中文说明：修复后 `zig test maths/pi_monte_carlo_estimation.zig` 已通过。

## Batch E Wave 10 Update / Batch E 第 10 波更新 (2026-03-10)

Result / 结果:
- Added the final two remaining `maths` modules: `pi_generator` and `chudnovsky_algorithm`.
- 已新增最后 2 个剩余 `maths` 模块：`pi_generator` 与 `chudnovsky_algorithm`。
- Reduced the remaining planned gap to `78`.
- 将剩余计划缺口压到 `78`。
- Current checkpoint accounting after Batch E Wave 10 integration:
- Batch E 第 10 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `846`
  - `build.zig` 已注册算法数：`846`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `838`
  - 按计划分类上限口径的有效完成数：`838`
  - remaining planned gap: `78`
  - 剩余计划缺口：`78`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test maths/pi_generator.zig`
  - Symptom / 现象: the first draft failed to compile because `BigInt.toInt(u8)` introduced conversion errors not covered by the public error set.
  - 中文说明：初版编译失败，因为 `BigInt.toInt(u8)` 会引入额外的转换错误，而这些错误没有包含在公开错误集中。
  - Root Cause / 根因: Zig correctly propagated `NegativeIntoUnsigned` / `TargetTooSmall`, even though the digit-emission invariant guarantees the current digit is always `0..9`.
  - 中文说明：尽管“当前输出位一定在 `0..9`”这一算法不变量成立，Zig 仍会正确传播 `NegativeIntoUnsigned` / `TargetTooSmall` 这两个转换错误。
  - Fix Applied / 修复措施: converted that point to `catch unreachable` with the invariant documented in code.
  - 中文说明：已将该位置改为 `catch unreachable`，并依赖代码中的算法不变量。
  - Post-Fix Verification / 修复后验证: both `zig test maths/pi_generator.zig` and `zig test maths/chudnovsky_algorithm.zig` passed.
  - 中文说明：修复后 `zig test maths/pi_generator.zig` 与 `zig test maths/chudnovsky_algorithm.zig` 均已通过。

## Batch E Wave 11 Update / Batch E 第 11 波更新 (2026-03-10)

Result / 结果:
- Added `Project Euler` problems `055` through `058`.
- 已新增 `Project Euler` 第 `055` 到 `058` 题。
- Reduced the remaining planned gap to `74`.
- 将剩余计划缺口压到 `74`。
- Current checkpoint accounting after Batch E Wave 11 integration:
- Batch E 第 11 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `850`
  - `build.zig` 已注册算法数：`850`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `842`
  - 按计划分类上限口径的有效完成数：`842`
  - remaining planned gap: `74`
  - 剩余计划缺口：`74`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_057.zig`
  - Symptom / 现象: the reference-scale test took noticeably longer because the `10_000`-expansion case exercises repeated decimal big-number additions.
  - 中文说明：参考规模测试耗时明显更长，因为 `10_000` 次展开会触发大量十进制大数加法。
  - Root Cause / 根因: this implementation intentionally uses exact decimal digit vectors to keep behavior aligned with Python without introducing approximate arithmetic.
  - 中文说明：该实现有意采用精确的十进制数位数组，以在不引入近似运算的前提下与 Python 结果对齐。
  - Fix Applied / 修复措施: no semantic code change was required; the test was allowed to complete and verified successfully.
  - 中文说明：无需修改语义实现，只是继续等待测试跑完并确认其成功。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_057.zig` passed.
  - 中文说明：最终 `zig test project_euler/problem_057.zig` 已通过。

## Batch E Wave 12 Update / Batch E 第 12 波更新 (2026-03-10)

Result / 结果:
- Added `Project Euler` problems `059`, `062`, `063`, `064`, and `065`.
- 已新增 `Project Euler` 第 `059`、`062`、`063`、`064`、`065` 题。
- Reduced the remaining planned gap to `69`.
- 将剩余计划缺口压到 `69`。
- Current checkpoint accounting after Batch E Wave 12 integration:
- Batch E 第 12 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `855`
  - `build.zig` 已注册算法数：`855`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `847`
  - 按计划分类上限口径的有效完成数：`847`
  - remaining planned gap: `69`
  - 剩余计划缺口：`69`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_059.zig`
  - Symptom / 现象: the first draft failed to compile because it used `std.ArrayList(...).init(...)`, which is not available in Zig `0.15.2`.
  - 中文说明：初版编译失败，因为它使用了 Zig `0.15.2` 中不可用的 `std.ArrayList(...).init(...)` 写法。
  - Root Cause / 根因: the implementation accidentally used the newer `ArrayList` initialization style instead of the repository's current `ArrayListUnmanaged`-compatible pattern.
  - 中文说明：实现误用了较新的 `ArrayList` 初始化风格，而不是当前仓库兼容的 `ArrayListUnmanaged` 模式。
  - Fix Applied / 修复措施: replaced those temporary containers with allocator-driven `std.ArrayListUnmanaged`.
  - 中文说明：已将这些临时容器改为显式传 allocator 的 `std.ArrayListUnmanaged`。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_059.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_059.zig` 已通过。
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_063.zig`
  - Symptom / 现象: the first edge-case assertions failed because the expected counts for `solution(10, 2)` and `solution(10, 3)` were too low.
  - 中文说明：初版边界断言失败，因为 `solution(10, 2)` 与 `solution(10, 3)` 的期望值写低了。
  - Root Cause / 根因: the test draft forgot that the Python reference uses exclusive upper bounds via `range(1, max_power)`, so those cases include both first and second powers.
  - 中文说明：测试初稿忽略了 Python 参考实现使用 `range(1, max_power)` 的上界排他语义，因此这些 case 会同时包含一次幂和二次幂。
  - Fix Applied / 修复措施: corrected the expected values to `9` and `15` after re-checking the Python reference semantics.
  - 中文说明：重新核对 Python 参考语义后，已将期望值修正为 `9` 和 `15`。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_063.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_063.zig` 已通过。
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_065.zig`
  - Symptom / 现象: the first draft failed to compile because `std.math.big.int.Managed` in Zig `0.15.2` does not provide the `mulScalar` member used in the recurrence.
  - 中文说明：初版编译失败，因为 Zig `0.15.2` 的 `std.math.big.int.Managed` 不提供递推里使用的 `mulScalar` 成员。
  - Root Cause / 根因: the implementation assumed a scalar-multiplication convenience API that is not present in this toolchain version.
  - 中文说明：实现错误地假设当前工具链里存在一个标量乘法便捷接口。
  - Fix Applied / 修复措施: created a temporary bigint coefficient with `initSet` and switched the recurrence step to `mul` + `add`.
  - 中文说明：已用 `initSet` 创建临时大整数系数，并将递推步骤改为 `mul` + `add`。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_065.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_065.zig` 已通过。

## Batch E Wave 13 Update / Batch E 第 13 波更新 (2026-03-10)

Result / 结果:
- Added `Project Euler` problems `067`, `069`, `071`, `072`, and `073`.
- 已新增 `Project Euler` 第 `067`、`069`、`071`、`072`、`073` 题。
- Reduced the remaining planned gap to `64`.
- 将剩余计划缺口压到 `64`。
- Current checkpoint accounting after Batch E Wave 13 integration:
- Batch E 第 13 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `860`
  - `build.zig` 已注册算法数：`860`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `852`
  - 按计划分类上限口径的有效完成数：`852`
  - remaining planned gap: `64`
  - 剩余计划缺口：`64`

Failure Log / 失败记录:
- No implementation or test failures were encountered in this wave; all five files passed on the first file-level test run.
- 本波未遇到实现或测试失败；5 个文件均在首次文件级测试中通过。

## Batch E Wave 14 Update / Batch E 第 14 波更新 (2026-03-10)

Result / 结果:
- Added `Project Euler` problems `074`, `075`, `076`, and `077`.
- 已新增 `Project Euler` 第 `074`、`075`、`076`、`077` 题。
- Reduced the remaining planned gap to `60`.
- 将剩余计划缺口压到 `60`。
- Current checkpoint accounting after Batch E Wave 14 integration:
- Batch E 第 14 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `864`
  - `build.zig` 已注册算法数：`864`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `856`
  - 按计划分类上限口径的有效完成数：`856`
  - remaining planned gap: `60`
  - 剩余计划缺口：`60`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_075.zig`
  - Symptom / 现象: the first draft failed to compile because a single-line `for` loop body used an assignment-style `if` expression that Zig rejected.
  - 中文说明：初版编译失败，因为单行 `for` 循环体里用了 Zig 不接受的赋值式 `if` 写法。
  - Root Cause / 根因: the implementation compressed the counting loop too aggressively instead of using an explicit block body.
  - 中文说明：实现把计数循环压缩得过头了，没有使用显式块体。
  - Fix Applied / 修复措施: expanded the loop into a normal block and kept the conditional increment inside it.
  - 中文说明：已把循环展开为普通块体，并将条件累加保留在块内。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_075.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_075.zig` 已通过。

## Batch E Wave 15 Update / Batch E 第 15 波更新 (2026-03-10)

Result / 结果:
- Added `Project Euler` problems `079`, `081`, `082`, `085`, and `087`.
- 已新增 `Project Euler` 第 `079`、`081`、`082`、`085`、`087` 题。
- Reduced the remaining planned gap to `55`.
- 将剩余计划缺口压到 `55`。
- Current checkpoint accounting after Batch E Wave 15 integration:
- Batch E 第 15 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `869`
  - `build.zig` 已注册算法数：`869`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `861`
  - 按计划分类上限口径的有效完成数：`861`
  - remaining planned gap: `55`
  - 剩余计划缺口：`55`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_079.zig`
  - Symptom / 现象: the first draft failed to compile because the helper that counted present digits reused the same invalid single-line `for` + assignment-style `if` pattern.
  - 中文说明：初版编译失败，因为统计出现数字个数的辅助逻辑复用了同样不合法的单行 `for` + 赋值式 `if` 写法。
  - Root Cause / 根因: the quick counting block was written too compactly instead of using a normal loop body.
  - 中文说明：快速统计代码写得过于紧凑，没有使用正常的循环块体。
  - Fix Applied / 修复措施: expanded that loop into a block body.
  - 中文说明：已将该循环展开为块体写法。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_079.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_079.zig` 已通过。
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_085.zig`
  - Symptom / 现象: the first tiny-target assertion expected area `2` for `solution(2)`, but the implementation and Python behavior both returned `1`.
  - 中文说明：初版极小目标断言把 `solution(2)` 的面积写成了 `2`，但实现和 Python 行为都返回 `1`。
  - Root Cause / 根因: the added edge test assumed “closer target implies wider grid”, which is false for this rectangle-count objective.
  - 中文说明：补充的边界测试错误地假设“目标更接近 2 就应该选更宽的网格”，但矩形计数目标并不满足这个假设。
  - Fix Applied / 修复措施: corrected the expected value to `1` after checking the search result.
  - 中文说明：在核对搜索结果后，已将期望值修正为 `1`。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_085.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_085.zig` 已通过。

## Batch E Wave 16 Update / Batch E 第 16 波更新 (2026-03-10)

Result / 结果:
- Added `Project Euler` problems `089`, `091`, `092`, `094`, and `095`.
- 已新增 `Project Euler` 第 `089`、`091`、`092`、`094`、`095` 题。
- Reduced the remaining planned gap to `50`.
- 将剩余计划缺口压到 `50`。
- Current checkpoint accounting after Batch E Wave 16 integration:
- Batch E 第 16 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `874`
  - `build.zig` 已注册算法数：`874`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `866`
  - 按计划分类上限口径的有效完成数：`866`
  - remaining planned gap: `50`
  - 剩余计划缺口：`50`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_095.zig`
  - Symptom / 现象: the first cycle-detection implementation returned `220` for the `200000` case instead of the Python reference result `12496`.
  - 中文说明：第一版环检测在 `200000` case 上返回了 `220`，而不是 Python 参考结果 `12496`。
  - Root Cause / 根因: the first pruning strategy marked whole traversed paths as processed before evaluating every cycle by the cycle itself, so longer amicable chains reachable from later starts were skipped.
  - 中文说明：第一版剪枝过早把整条遍历路径标记为已处理，但没有按“检测到的环本身”统一评估，导致后续起点可达的更长亲和链被跳过。
  - Fix Applied / 修复措施: reworked the search to use per-run visit stamps and update the best answer from any detected loop segment, then safely reintroduced `done` pruning only after the full current traversal had been analyzed.
  - 中文说明：已改成按轮次打标的访问戳，并在检测到任意环时直接按该环片段更新最优答案；随后只在完整分析当前遍历后才安全地恢复 `done` 剪枝。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_095.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_095.zig` 已通过。

## Batch E Wave 17 Update / Batch E 第 17 波更新 (2026-03-10)

Result / 结果:
- Added `Project Euler` problems `097`, `099`, `100`, and `102`.
- 已新增 `Project Euler` 第 `097`、`099`、`100`、`102` 题。
- Reduced the remaining planned gap to `46`.
- 将剩余计划缺口压到 `46`。
- Current checkpoint accounting after Batch E Wave 17 integration:
- Batch E 第 17 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `878`
  - `build.zig` 已注册算法数：`878`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `870`
  - 按计划分类上限口径的有效完成数：`870`
  - remaining planned gap: `46`
  - 剩余计划缺口：`46`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_097.zig`
  - Symptom / 现象: the first modular exponentiation draft overflowed when squaring the current base under a `10^10` modulus.
  - 中文说明：第一版模幂实现里，在 `10^10` 模数下做底数平方时发生了整数溢出。
  - Root Cause / 根因: the implementation used `u64` multiplication directly even though intermediate products can exceed `u64` before the modulus is applied.
  - 中文说明：实现直接使用了 `u64` 乘法，但在取模前的中间乘积可能超过 `u64` 上限。
  - Fix Applied / 修复措施: promoted the intermediate modular multiplications to `u128` and cast the reduced result back to `u64`.
  - 中文说明：已将模乘中间值提升到 `u128`，再把取模后的结果转回 `u64`。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_097.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_097.zig` 已通过。
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_100.zig`
  - Symptom / 现象: the first tiny-threshold test panicked from unsigned underflow and then, after the boundary fix, still failed because the expected value for `min_total = 0` was incorrect.
  - 中文说明：第一版极小阈值测试先因无符号下溢崩溃，修复边界后又因为 `min_total = 0` 的期望值写错而失败。
  - Root Cause / 根因: the recurrence bound used `2 * min_total - 1` without guarding `min_total = 0`, and the added edge-case assertion did not reflect the Python behavior for that input.
  - 中文说明：递推边界直接使用了 `2 * min_total - 1`，没有保护 `min_total = 0`；同时新增的边界断言也没有按 Python 对该输入的行为来写。
  - Fix Applied / 修复措施: added an explicit zero-threshold guard for the recurrence bound and corrected the test expectation from `3` to `1`.
  - 中文说明：已为递推上界补上 `0` 阈值保护，并把测试期望从 `3` 修正为 `1`。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_100.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_100.zig` 已通过。

## Batch E Wave 18 Update / Batch E 第 18 波更新 (2026-03-10)

Result / 结果:
- Added `Project Euler` problems `107`, `109`, `112`, and `113`.
- 已新增 `Project Euler` 第 `107`、`109`、`112`、`113` 题。
- Reduced the remaining planned gap to `42`.
- 将剩余计划缺口压到 `42`。
- Current checkpoint accounting after Batch E Wave 18 integration:
- Batch E 第 18 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `882`
  - `build.zig` 已注册算法数：`882`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `874`
  - 按计划分类上限口径的有效完成数：`874`
  - remaining planned gap: `42`
  - 剩余计划缺口：`42`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_107.zig`
  - Symptom / 现象: the first draft failed to compile because `allocator` was incorrectly discarded even though it was still used later, and `best_weight` inherited a `comptime_int` type from the sentinel constant.
  - 中文说明：第一版编译失败，一方面错误地把后续仍要使用的 `allocator` 丢弃了，另一方面 `best_weight` 从哨兵常量推断成了 `comptime_int`。
  - Root Cause / 根因: the scaffold still contained a leftover `_ = allocator;` from an earlier parse-only draft, and the MST loop initialized the mutable best-edge weight without an explicit runtime integer type.
  - 中文说明：脚手架代码残留了早期仅解析版本留下的 `_ = allocator;`；同时最小生成树循环里给可变最优边权赋初值时没有显式写运行期整数类型。
  - Fix Applied / 修复措施: removed the pointless discard and declared `best_weight` as `u32`.
  - 中文说明：已删除无意义的参数丢弃，并把 `best_weight` 显式声明为 `u32`。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_107.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_107.zig` 已通过。

## Batch E Wave 19 Update / Batch E 第 19 波更新 (2026-03-10)

Result / 结果:
- Added `Project Euler` problems `114`, `115`, `116`, `117`, and `120`.
- 已新增 `Project Euler` 第 `114`、`115`、`116`、`117`、`120` 题。
- Reduced the remaining planned gap to `37`.
- 将剩余计划缺口压到 `37`。
- Current checkpoint accounting after Batch E Wave 19 integration:
- Batch E 第 19 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `887`
  - `build.zig` 已注册算法数：`887`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `879`
  - 按计划分类上限口径的有效完成数：`879`
  - remaining planned gap: `37`
  - 剩余计划缺口：`37`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_120.zig`
  - Symptom / 现象: the first edge-case assertion expected `42` for `solution(7)`, but the implementation returned `100`.
  - 中文说明：第一版边界断言把 `solution(7)` 写成了 `42`，而实现返回 `100`。
  - Root Cause / 根因: the test accidentally asserted the maximum remainder for the single value `a = 7` instead of the problem's actual quantity, which sums `r_max` over all `3 <= a <= 7`.
  - 中文说明：测试把题意误写成了单个 `a = 7` 的最大余数，而题目真正要求的是对所有 `3 <= a <= 7` 的 `r_max` 求和。
  - Fix Applied / 修复措施: corrected the edge-case expectation from `42` to the Python-consistent cumulative value `100`.
  - 中文说明：已把该边界断言从 `42` 修正为与 Python 一致的累计结果 `100`。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_120.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_120.zig` 已通过。

## Batch E Wave 20 Update / Batch E 第 20 波更新 (2026-03-10)

Result / 结果:
- Added `Project Euler` problems `121`, `123`, `125`, and `131`.
- 已新增 `Project Euler` 第 `121`、`123`、`125`、`131` 题。
- Reduced the remaining planned gap to `33`.
- 将剩余计划缺口压到 `33`。
- Current checkpoint accounting after Batch E Wave 20 integration:
- Batch E 第 20 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `891`
  - `build.zig` 已注册算法数：`891`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `883`
  - 按计划分类上限口径的有效完成数：`883`
  - remaining planned gap: `33`
  - 剩余计划缺口：`33`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_125.zig`
  - Symptom / 现象: the first implementation returned `2410` for the `< 1000` case instead of the Python-consistent value `4164`.
  - 中文说明：第一版在 `< 1000` 的 case 上返回了 `2410`，而不是与 Python 一致的 `4164`。
  - Root Cause / 根因: the outer consecutive-square window advanced `first_square` in the loop post clause but rebuilt the next starting sum before that increment took effect, so one start position was repeated and later starts were skipped.
  - 中文说明：外层连续平方窗口把 `first_square` 放在循环尾部自增，但下一轮起始和却在自增生效前就重建了，导致某个起点被重复，后续起点被跳过。
  - Fix Applied / 修复措施: moved the `first_square += 1` step into the loop body before recomputing the next starting sum.
  - 中文说明：已把 `first_square += 1` 挪到循环体内部，并在它生效后再重建下一轮的起始和。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_125.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_125.zig` 已通过。

## Batch E Wave 21 Update / Batch E 第 21 波更新 (2026-03-10)

Result / 结果:
- Added `Project Euler` problems `173`, `174`, `191`, and `301`.
- 已新增 `Project Euler` 第 `173`、`174`、`191`、`301` 题。
- Reduced the remaining planned gap to `29`.
- 将剩余计划缺口压到 `29`。
- Current checkpoint accounting after Batch E Wave 21 integration:
- Batch E 第 21 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `895`
  - `build.zig` 已注册算法数：`895`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `887`
  - 按计划分类上限口径的有效完成数：`887`
  - remaining planned gap: `29`
  - 剩余计划缺口：`29`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_173.zig`
  - Symptom / 现象: the first tiny-limit assertion expected `2` for `solution(32)`, but the implementation and Python both returned `9`.
  - 中文说明：第一版极小上限断言把 `solution(32)` 写成了 `2`，但实现和 Python 都返回 `9`。
  - Root Cause / 根因: the added edge case confused the example statement “32 tiles can form two laminae” with the cumulative question “using up to 32 tiles”.
  - 中文说明：新增边界用例把题面中的“32 块砖恰好可形成两种”误当成了“最多 32 块砖时的累计数量”。
  - Fix Applied / 修复措施: corrected the expectation to the cumulative Python result `9`.
  - 中文说明：已把期望值改为与 Python 一致的累计结果 `9`。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_173.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_173.zig` 已通过。
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_174.zig`
  - Symptom / 现象: the first tiny-limit assertion expected `1` for `solution(32, 1)`, but the implementation and Python both returned `5`.
  - 中文说明：第一版极小上限断言把 `solution(32, 1)` 写成了 `1`，但实现和 Python 都返回 `5`。
  - Root Cause / 根因: the test again used the “exactly 32 tiles” reading instead of counting all tile totals up to `32` whose lamina type is `L(1)`.
  - 中文说明：该测试同样把题意误读成“恰好 32 块砖”，而不是“统计所有 `<= 32` 且类型为 `L(1)` 的砖数”。
  - Fix Applied / 修复措施: corrected the expectation to `5` after checking the Python reference.
  - 中文说明：在核对 Python 参考实现后，已把期望值修正为 `5`。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_174.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_174.zig` 已通过。
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_191.zig`
  - Symptom / 现象: the first draft failed to compile because the nested `for` expression used a single-line accumulation form that Zig rejected as an invalid assignment target.
  - 中文说明：第一版编译失败，因为嵌套 `for` 表达式使用了 Zig 不接受的单行累加写法，形成了非法赋值目标。
  - Root Cause / 根因: the final DP total reduction was written too compactly instead of using a normal block loop.
  - 中文说明：最终 DP 求和写得过于紧凑，没有使用普通块体循环。
  - Fix Applied / 修复措施: expanded the nested reduction into explicit loop blocks.
  - 中文说明：已把嵌套求和展开为显式循环块。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_191.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_191.zig` 已通过。
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_301.zig`
  - Symptom / 现象: the first exact-Fibonacci implementation failed Python-alignment cases such as `solution(5)` because the Python reference's floating formula truncates one less on odd exponents.
  - 中文说明：第一版按精确 Fibonacci 实现，在 `solution(5)` 这类 case 上与 Python 不一致，因为 Python 参考实现使用浮点公式，在奇数指数时会少截出 `1`。
  - Root Cause / 根因: I matched the mathematical identity instead of the Python reference behavior required by the repository rules.
  - 中文说明：我先对齐了数学恒等式，而不是仓库规则要求的 Python 参考行为。
  - Fix Applied / 修复措施: kept the stable integer recurrence but adjusted the final value to mirror the Python formula's odd-exponent truncation behavior.
  - 中文说明：保留稳定的整数递推，同时在最终结果上补入与 Python 浮点公式一致的奇数指数截断语义。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_301.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_301.zig` 已通过。

## Batch E Wave 22 Update / Batch E 第 22 波更新 (2026-03-10)

Result / 结果:
- Added `Project Euler` problems `203`, `205`, `206`, and `207`.
- 已新增 `Project Euler` 第 `203`、`205`、`206`、`207` 题。
- Reduced the remaining planned gap to `25`.
- 将剩余计划缺口压到 `25`。
- Current checkpoint accounting after Batch E Wave 22 integration:
- Batch E 第 22 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `899`
  - `build.zig` 已注册算法数：`899`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `891`
  - 按计划分类上限口径的有效完成数：`891`
  - remaining planned gap: `25`
  - 剩余计划缺口：`25`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_203.zig`
  - Symptom / 现象: the first implementation crashed with an integer-overflow panic while building Pascal rows, even for the small `n = 8` test.
  - 中文说明：第一版在构造 Pascal 行时触发了整数溢出 panic，即使是很小的 `n = 8` 测试也会崩。
  - Root Cause / 根因: the current row buffer was assigned into `previous` but was also still scheduled for `deinit` at scope exit, so the next iteration read freed memory and produced bogus overflowing values.
  - 中文说明：当前行缓冲区被赋给了 `previous`，但它仍然会在作用域结束时被 `deinit`，导致下一轮读到已释放内存，并进一步产生伪造的大数溢出。
  - Fix Applied / 修复措施: removed the automatic `defer current.deinit(allocator)` and transferred ownership to `previous` explicitly.
  - 中文说明：已移除 `defer current.deinit(allocator)`，并显式把缓冲区所有权转交给 `previous`。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_203.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_203.zig` 已通过。

## Batch E Wave 23 Update / Batch E 第 23 波更新 (2026-03-10)

Result / 结果:
- Added `Project Euler` problems `145`, `188`, and `190`.
- 已新增 `Project Euler` 第 `145`、`188`、`190` 题。
- Reduced the remaining planned gap to `22`.
- 将剩余计划缺口压到 `22`。
- Current checkpoint accounting after Batch E Wave 23 integration:
- Batch E 第 23 波接入后的当前检查点统计：
  - `build.zig` registered algorithms: `902`
  - `build.zig` 已注册算法数：`902`
  - portable target total: `916`
  - 可移植目标总量：`916`
  - effective completed count under plan-category caps: `894`
  - 按计划分类上限口径的有效完成数：`894`
  - remaining planned gap: `22`
  - 剩余计划缺口：`22`

Failure Log / 失败记录:
- Failing Step/Command / 失败步骤或命令:
  - `zig test project_euler/problem_190.zig`
  - Symptom / 现象: the first draft failed to compile because the running product `p` was inferred as `comptime_float`.
  - 中文说明：第一版编译失败，因为累乘变量 `p` 被推断成了 `comptime_float`。
  - Root Cause / 根因: the mutable floating accumulator was initialized without an explicit runtime type.
  - 中文说明：可变浮点累加器初始化时没有显式写出运行期类型。
  - Fix Applied / 修复措施: declared `p` as `f64`.
  - 中文说明：已把 `p` 显式声明为 `f64`。
  - Post-Fix Verification / 修复后验证: `zig test project_euler/problem_190.zig` passed.
  - 中文说明：修复后 `zig test project_euler/problem_190.zig` 已通过。
