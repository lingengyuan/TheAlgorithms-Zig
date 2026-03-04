# Vibe Coding Experiment Log

This log now tracks algorithm implementation quality with a focus on:
- functional correctness,
- edge-case robustness,
- consistency with Python reference behavior.

Reference project for behavior alignment:
- `https://github.com/TheAlgorithms/Python`

## Ongoing Logging Scope

For each batch/review cycle, only record:
- failing command/step,
- error symptom,
- root cause,
- fix applied,
- post-fix verification result.

## Phase 5 Batch E - Wave 4 (2026-03-04)

Scope:
- `ciphers/cryptomath_module.zig`
- `ciphers/diffie.zig`
- `ciphers/deterministic_miller_rabin.zig`
- `ciphers/rsa_factorization.zig`
- `ciphers/porta_cipher.zig`
- `ciphers/mixed_keyword_cypher.zig`
- `ciphers/simple_keyword_cypher.zig`
- `ciphers/simple_substitution_cipher.zig`

Result:
- 8/8 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test ciphers/cryptomath_module.zig` ✅
- `zig test ciphers/diffie.zig` ✅
- `zig test ciphers/deterministic_miller_rabin.zig` ✅
- `zig test ciphers/rsa_factorization.zig` ✅
- `zig test ciphers/porta_cipher.zig` ✅
- `zig test ciphers/mixed_keyword_cypher.zig` ✅
- `zig test ciphers/simple_keyword_cypher.zig` ✅
- `zig test ciphers/simple_substitution_cipher.zig` ✅

Failure Log:
- Failing step/command:
  - `zig test ciphers/cryptomath_module.zig`
  - Symptom: compile error `name shadows primitive 'u1'`.
  - Root cause: extended-Euclid local variable names (`u1/u2/u3`) conflicted with Zig primitive type names.
  - Fix applied: renamed state variables to non-primitive identifiers (`s1/s2/s3`, `t1/t2/t3`).
  - Post-fix verification: `zig test ciphers/cryptomath_module.zig` passed.
- Failing step/command:
  - `zig test ciphers/deterministic_miller_rabin.zig`
  - Symptom: compile error on shift amount type (`expected type 'u7', found 'u32'`).
  - Root cause: left-shift in exponent assembly used uncast runtime loop counter.
  - Fix applied: cast shift operand to `u7` for Zig's shift-width requirement.
  - Post-fix verification: `zig test ciphers/deterministic_miller_rabin.zig` passed.
- Failing step/command:
  - `zig test ciphers/mixed_keyword_cypher.zig`
  - Symptom: test mismatch in mapping-basics assertion.
  - Root cause: test expected values did not match the vertical-column mapping order defined by Python reference.
  - Fix applied: corrected assertions to the actual Python-order mapping outputs.
  - Post-fix verification: `zig test ciphers/mixed_keyword_cypher.zig` passed.

## Phase 5 Batch E - Wave 3 (2026-03-04)

Scope:
- `ciphers/beaufort_cipher.zig`
- `ciphers/gronsfeld_cipher.zig`
- `ciphers/vernam_cipher.zig`
- `ciphers/running_key_cipher.zig`
- `ciphers/onepad_cipher.zig`
- `ciphers/permutation_cipher.zig`
- `ciphers/mono_alphabetic_ciphers.zig`
- `ciphers/brute_force_caesar_cipher.zig`

Result:
- 8/8 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test ciphers/beaufort_cipher.zig` ✅
- `zig test ciphers/gronsfeld_cipher.zig` ✅
- `zig test ciphers/vernam_cipher.zig` ✅
- `zig test ciphers/running_key_cipher.zig` ✅
- `zig test ciphers/onepad_cipher.zig` ✅
- `zig test ciphers/permutation_cipher.zig` ✅
- `zig test ciphers/mono_alphabetic_ciphers.zig` ✅
- `zig test ciphers/brute_force_caesar_cipher.zig` ✅

Failure Log:
- Failing step/command:
  - `zig test ciphers/running_key_cipher.zig`
  - Symptom: sample test failed with `InvalidCharacter`.
  - Root cause: implementation over-restricted key/plaintext to letters only, but Python reference accepts non-space symbols in key stream.
  - Fix applied: changed normalization to only remove spaces and uppercase characters, preserving Python behavior.
  - Post-fix verification: `zig test ciphers/running_key_cipher.zig` passed.
- Failing step/command:
  - `zig test ciphers/permutation_cipher.zig`
  - Symptom: compile error `root source file struct 'std' has no member named 'BoundedArray'`.
  - Root cause: used unavailable std API for current Zig toolchain (`0.15.2`).
  - Fix applied: replaced with allocator-backed boolean marker array for key-validation checks.
  - Post-fix verification: `zig test ciphers/permutation_cipher.zig` passed.
- Failing step/command:
  - `zig test ciphers/mono_alphabetic_ciphers.zig`
  - Symptom: Python sample mismatch (`expected \"Pcssi Bidsm\", found \"Itssg Vgksr\"`).
  - Root cause: encrypt/decrypt mapping direction was reversed from Python implementation.
  - Fix applied: reworked translation to use Python-equivalent `chars_a/chars_b` mapping direction.
  - Post-fix verification: `zig test ciphers/mono_alphabetic_ciphers.zig` passed.
- Failing step/command:
  - `zig test ciphers/permutation_cipher.zig`
  - Symptom: space-handling test failed with `InvalidBlockSize`.
  - Root cause: test fixture length was not divisible by permutation block size.
  - Fix applied: replaced fixture with block-aligned message while preserving space-case coverage.
  - Post-fix verification: `zig test ciphers/permutation_cipher.zig` passed.

## Phase 5 Batch E - Wave 2 (2026-03-04)

Scope:
- `ciphers/affine_cipher.zig`
- `ciphers/baconian_cipher.zig`
- `ciphers/base16.zig`
- `ciphers/base32.zig`
- `ciphers/base85.zig`
- `ciphers/morse_code.zig`
- `ciphers/polybius.zig`
- `ciphers/autokey.zig`

Result:
- 8/8 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test ciphers/affine_cipher.zig` ✅
- `zig test ciphers/baconian_cipher.zig` ✅
- `zig test ciphers/base16.zig` ✅
- `zig test ciphers/base32.zig` ✅
- `zig test ciphers/base85.zig` ✅
- `zig test ciphers/morse_code.zig` ✅
- `zig test ciphers/polybius.zig` ✅
- `zig test ciphers/autokey.zig` ✅

Failure Log:
- Failing step/command:
  - `zig test ciphers/affine_cipher.zig`
  - Symptom: compile error around absolute-value/gcd type mismatch.
  - Root cause: `@abs` usage inferred an incompatible integer type for gcd operands.
  - Fix applied: replaced with explicit signed absolute-value handling and consistent integer typing.
  - Post-fix verification: `zig test ciphers/affine_cipher.zig` passed.
- Failing step/command:
  - `zig test ciphers/base16.zig`
  - Symptom: compile error for non-existent formatter helper.
  - Root cause: attempted to use unavailable `std.fmt.fmtSliceHexUpper`.
  - Fix applied: implemented manual uppercase hex encoding path.
  - Post-fix verification: `zig test ciphers/base16.zig` passed.
- Failing step/command:
  - `zig test ciphers/base16.zig`
  - Symptom: invalid-input test assertion mismatch.
  - Root cause: test fixture used an unsuitable invalid-case string.
  - Fix applied: corrected test input to an even-length invalid-hex sample that targets alphabet validation.
  - Post-fix verification: `zig test ciphers/base16.zig` passed.
- Failing step/command:
  - `zig test ciphers/base32.zig` and `zig test ciphers/base85.zig`
  - Symptom: compile error from comptime integer literal participation in bit operations.
  - Root cause: untyped literals in bit-accumulation code triggered comptime-only constraints.
  - Fix applied: added explicit typed literals (`@as(u8, 1)`, etc.) in bit paths.
  - Post-fix verification: both file tests passed.
- Failing step/command:
  - `zig test ciphers/autokey.zig`
  - Symptom: compile error from incorrect `std.ascii.lowerString` call shape.
  - Root cause: function requires explicit output buffer argument.
  - Fix applied: updated test to pass destination buffer correctly.
  - Post-fix verification: `zig test ciphers/autokey.zig` passed.

## Phase 5 Batch E - Wave 1 (2026-03-04)

Scope:
- `ciphers/rot13.zig`
- `ciphers/atbash.zig`
- `ciphers/vigenere_cipher.zig`
- `ciphers/rail_fence_cipher.zig`
- `ciphers/xor_cipher.zig`
- `ciphers/base64_cipher.zig`
- `ciphers/transposition_cipher.zig`
- `ciphers/a1z26.zig`

Result:
- 8/8 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test ciphers/rot13.zig` ✅
- `zig test ciphers/atbash.zig` ✅
- `zig test ciphers/vigenere_cipher.zig` ✅
- `zig test ciphers/rail_fence_cipher.zig` ✅
- `zig test ciphers/xor_cipher.zig` ✅
- `zig test ciphers/base64_cipher.zig` ✅
- `zig test ciphers/transposition_cipher.zig` ✅
- `zig test ciphers/a1z26.zig` ✅

Failure Log:
- No implementation/test failures encountered in this wave.

## Phase 5 Batch D - Wave 6 (2026-03-04)

Scope:
- `graphs/karger_min_cut.zig`
- `graphs/markov_chain.zig`
- `graphs/graph_adjacency_list.zig`
- `graphs/graph_adjacency_matrix.zig`

Result:
- 4/4 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test graphs/karger_min_cut.zig` ✅
- `zig test graphs/markov_chain.zig` ✅
- `zig test graphs/graph_adjacency_list.zig` ✅
- `zig test graphs/graph_adjacency_matrix.zig` ✅

Failure Log:
- Failing step/command:
  - `zig fmt graphs/karger_min_cut.zig ...`
  - Symptom: parse error `expected pointer dereference, optional unwrap, or field access, found 'union'`.
  - Root cause: method name `union` conflicted with Zig keyword.
  - Fix applied: renamed method to `unionSets` and updated call sites.
  - Post-fix verification: `zig test graphs/karger_min_cut.zig` passed.
- Failing step/command:
  - `zig test graphs/karger_min_cut.zig`
  - Symptom: compile error `variable of type 'comptime_int' must be const or comptime`.
  - Root cause: sentinel `best` lacked explicit `usize` annotation.
  - Fix applied: annotated `best` as `usize`.
  - Post-fix verification: `zig test graphs/karger_min_cut.zig` passed.
- Failing step/command:
  - `zig test graphs/karger_min_cut.zig`
  - Symptom: extreme-chain test failed (`expected 1, found 0`).
  - Root cause: test fixture built malformed adjacency slices (did not represent a valid chain).
  - Fix applied: rebuilt extreme fixture as one-direction chain edges, which are canonicalized as undirected by the implementation.
  - Post-fix verification: `zig test graphs/karger_min_cut.zig` passed.

## Phase 5 Batch D - Wave 5 (2026-03-04)

Scope:
- `graphs/breadth_first_search_2.zig`
- `graphs/depth_first_search_2.zig`
- `graphs/dijkstra_2.zig`
- `graphs/dijkstra_alternate.zig`
- `graphs/kahn_longest_distance.zig`
- `graphs/greedy_min_vertex_cover.zig`
- `graphs/matching_min_vertex_cover.zig`
- `graphs/random_graph_generator.zig`

Result:
- 8/8 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test graphs/breadth_first_search_2.zig` ✅
- `zig test graphs/depth_first_search_2.zig` ✅
- `zig test graphs/dijkstra_2.zig` ✅
- `zig test graphs/dijkstra_alternate.zig` ✅
- `zig test graphs/kahn_longest_distance.zig` ✅
- `zig test graphs/greedy_min_vertex_cover.zig` ✅
- `zig test graphs/matching_min_vertex_cover.zig` ✅
- `zig test graphs/random_graph_generator.zig` ✅

Failure Log:
- Failing step/command:
  - `zig test graphs/greedy_min_vertex_cover.zig`
  - Symptom: compile error `invalid left-hand side to assignment`.
  - Root cause: single-line `for` with inline `if` and assignment was parsed as invalid syntax in Zig.
  - Fix applied: expanded to block-form `for` loop with explicit braces.
  - Post-fix verification: `zig test graphs/greedy_min_vertex_cover.zig` passed.
- Failing step/command:
  - `zig test graphs/matching_min_vertex_cover.zig`
  - Symptom: sample assertion mismatch (`expected ...4, found ...3`).
  - Root cause: algorithm has multiple valid solutions due arbitrary matching edge selection; original test over-constrained to one specific set.
  - Fix applied: changed assertion to validate vertex-cover correctness + size bound instead of one fixed vertex set.
  - Post-fix verification: `zig test graphs/matching_min_vertex_cover.zig` passed.

## Phase 5 Batch D - Wave 1 (2026-03-04)

Scope:
- `graphs/articulation_points.zig`
- `graphs/kosaraju_scc.zig`
- `graphs/kahn_topological_sort.zig`
- `graphs/breadth_first_search_shortest_path.zig`
- `graphs/boruvka_mst.zig`
- `graphs/zero_one_bfs_shortest_path.zig`
- `graphs/bidirectional_breadth_first_search.zig`
- `graphs/dijkstra_binary_grid.zig`

Result:
- 8/8 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test graphs/articulation_points.zig` ✅
- `zig test graphs/kosaraju_scc.zig` ✅
- `zig test graphs/kahn_topological_sort.zig` ✅
- `zig test graphs/breadth_first_search_shortest_path.zig` ✅
- `zig test graphs/boruvka_mst.zig` ✅
- `zig test graphs/zero_one_bfs_shortest_path.zig` ✅
- `zig test graphs/bidirectional_breadth_first_search.zig` ✅
- `zig test graphs/dijkstra_binary_grid.zig` ✅
- `zig build test` ✅

Failure Log:
- Failing step/command:
  - `zig fmt graphs/boruvka_mst.zig ...`
  - Symptom: parse error `expected pointer dereference, optional unwrap, or field access, found 'union'`.
  - Root cause: method name `union` conflicted with Zig keyword.
  - Fix applied: renamed method to `unionSets` and updated call sites.
  - Post-fix verification: `zig test graphs/boruvka_mst.zig` passed.
- Failing step/command:
  - `zig fmt graphs/dijkstra_binary_grid.zig ...`
  - Symptom: parse error `expected 'an identifier', found 'unreachable'`.
  - Root cause: local test variable used Zig keyword `unreachable`.
  - Fix applied: renamed variable to `unreachable_grid`.
  - Post-fix verification: `zig test graphs/dijkstra_binary_grid.zig` passed.
- Failing step/command:
  - `zig test graphs/dijkstra_binary_grid.zig`
  - Symptom: compile error `variable of type 'comptime_int' must be const or comptime`.
  - Root cause: `current_best` inferred as comptime integer due missing explicit type.
  - Fix applied: added explicit `usize` type for sentinel and runtime variable.
  - Post-fix verification: `zig test graphs/dijkstra_binary_grid.zig` passed.

## Phase 5 Batch D - Wave 2 (2026-03-04)

Scope:
- `graphs/even_tree.zig`
- `graphs/gale_shapley_stable_matching.zig`
- `graphs/page_rank.zig`
- `graphs/bidirectional_dijkstra.zig`

Result:
- 4/4 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test graphs/even_tree.zig` ✅
- `zig test graphs/gale_shapley_stable_matching.zig` ✅
- `zig test graphs/page_rank.zig` ✅
- `zig test graphs/bidirectional_dijkstra.zig` ✅
- `zig build test` ✅

Failure Log:
- Failing step/command:
  - `zig test graphs/bidirectional_dijkstra.zig`
  - Symptom: compile error `variable of type 'comptime_int' must be const or comptime`.
  - Root cause: `inf`/`shortest` sentinel lacked explicit `i64` annotation and was inferred as comptime integer.
  - Fix applied: annotated `inf` and `shortest` as `i64`.
  - Post-fix verification: `zig test graphs/bidirectional_dijkstra.zig` passed.

## Phase 5 Batch D - Wave 3 (2026-03-04)

Scope:
- `graphs/greedy_best_first.zig`
- `graphs/dinic_max_flow.zig`

Result:
- 2/2 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test graphs/greedy_best_first.zig` ✅
- `zig test graphs/dinic_max_flow.zig` ✅
- `zig build test` ✅

Failure Log:
- No implementation/test failures encountered in this wave.

## Phase 5 Batch D - Wave 4 (2026-03-04)

Scope:
- `graphs/bidirectional_search.zig`
- `graphs/minimum_path_sum.zig`
- `graphs/deep_clone_graph.zig`
- `graphs/dijkstra_matrix.zig`

Result:
- 4/4 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test graphs/bidirectional_search.zig` ✅
- `zig test graphs/minimum_path_sum.zig` ✅
- `zig test graphs/deep_clone_graph.zig` ✅
- `zig test graphs/dijkstra_matrix.zig` ✅
- `zig build test` ✅

Failure Log:
- Failing step/command:
  - `zig test graphs/bidirectional_search.zig`
  - Symptom: compile diagnostics (`pointless discard of function parameter`, `local variable is never mutated`).
  - Root cause: redundant `_ = allocator` and mutable declarations for immutable slices.
  - Fix applied: removed redundant discard and changed `var` to `const`.
  - Post-fix verification: file tests compiled and executed.
- Failing step/command:
  - `zig test graphs/bidirectional_search.zig`
  - Symptom: path assertion mismatch (`expected [0,1], found [0,1,1]`) in invalid-neighbor case.
  - Root cause: path stitching logic duplicated intersection node when intersection was goal.
  - Fix applied: in `constructFullPath`, return forward segment directly when `intersection == goal`.
  - Post-fix verification: `zig test graphs/bidirectional_search.zig` passed.
- Failing step/command:
  - `zig test graphs/minimum_path_sum.zig`
  - Symptom: negative-value test expected mismatch (`expected -8, found -7`).
  - Root cause: test oracle was incorrect; DP result `-7` is the true minimum path sum.
  - Fix applied: corrected expected test value to `-7`.
  - Post-fix verification: `zig test graphs/minimum_path_sum.zig` passed.
- Failing step/command:
  - `zig test graphs/deep_clone_graph.zig`
  - Symptom: allocator leak reported in invalid-neighbor error path.
  - Root cause: newly allocated neighbor slice was not freed when early-returning on invalid index.
  - Fix applied: added `errdefer allocator.free(neigh)` for per-node allocation.
  - Post-fix verification: `zig test graphs/deep_clone_graph.zig` passed without leak.
- Failing step/command:
  - `zig test graphs/dijkstra_matrix.zig`
  - Symptom: compile error `variable of type 'comptime_int' must be const or comptime`.
  - Root cause: `inf`/`min_val` inferred as comptime integer due missing explicit type.
  - Fix applied: annotated both as `i64`.
  - Post-fix verification: `zig test graphs/dijkstra_matrix.zig` passed.

## Phase 5 Batch A - Wave 1 (2026-03-04)

Scope:
- `maths/perfect_number.zig`
- `maths/aliquot_sum.zig`
- `maths/fermat_little_theorem.zig`
- `maths/segmented_sieve.zig`
- `maths/odd_sieve.zig`

Result:
- 5/5 implementations completed.
- Python-reference behavior aligned for normal and edge/extreme cases.
- `build.zig` registrations added for all 5 files.

Verification:
- `zig test maths/perfect_number.zig` ✅
- `zig test maths/aliquot_sum.zig` ✅
- `zig test maths/fermat_little_theorem.zig` ✅
- `zig test maths/segmented_sieve.zig` ✅
- `zig test maths/odd_sieve.zig` ✅
- `zig build test` ✅

Failure Log:
- No implementation/test failures encountered in this wave.

## Phase 5 Batch B - Wave 3 (2026-03-04)

Scope:
- `strings/pig_latin.zig`
- `strings/wildcard_pattern_matching.zig`
- `strings/top_k_frequent_words.zig`
- `strings/manacher.zig`
- `strings/min_cost_string_conversion.zig`

Result:
- 5/5 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test strings/pig_latin.zig` ✅
- `zig test strings/wildcard_pattern_matching.zig` ✅
- `zig test strings/top_k_frequent_words.zig` ✅
- `zig test strings/manacher.zig` ✅
- `zig test strings/min_cost_string_conversion.zig` ✅
- `zig build test` ✅

Failure Log:
- No implementation/test failures encountered in this wave.

## Phase 5 Batch C - Wave 2 (2026-03-04)

Scope:
- `sorts/bead_sort.zig`
- `sorts/cyclic_sort.zig`
- `sorts/exchange_sort.zig`
- `sorts/iterative_merge_sort.zig`
- `sorts/pigeon_sort.zig`
- `sorts/pigeonhole_sort.zig`
- `sorts/quick_sort_3_partition.zig`
- `sorts/recursive_quick_sort.zig`
- `sorts/shrink_shell_sort.zig`
- `sorts/stalin_sort.zig`

Result:
- 10/10 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test sorts/bead_sort.zig` ✅
- `zig test sorts/cyclic_sort.zig` ✅
- `zig test sorts/exchange_sort.zig` ✅
- `zig test sorts/iterative_merge_sort.zig` ✅
- `zig test sorts/pigeon_sort.zig` ✅
- `zig test sorts/pigeonhole_sort.zig` ✅
- `zig test sorts/quick_sort_3_partition.zig` ✅
- `zig test sorts/recursive_quick_sort.zig` ✅
- `zig test sorts/shrink_shell_sort.zig` ✅
- `zig test sorts/stalin_sort.zig` ✅
- `zig build test` ✅

Failure Log:
- Failing step/command:
  - `zig test sorts/pigeon_sort.zig`
  - Symptom: compile error `expected type 'error{RangeTooLarge}', found 'error{OutOfMemory}'`.
  - Root cause: function error set did not include allocator error from `allocator.alloc`.
  - Fix applied: expanded `PigeonSortError` to include `std.mem.Allocator.Error`.
  - Post-fix verification: `zig test sorts/pigeon_sort.zig` passed.

## Phase 5 Batch C - Wave 4 (2026-03-04)

Scope:
- `sorts/external_sort.zig`
- `sorts/intro_sort.zig`
- `sorts/msd_radix_sort.zig`
- `sorts/natural_sort.zig`
- `sorts/odd_even_transposition_parallel.zig`
- `sorts/patience_sort.zig`
- `sorts/tim_sort.zig`
- `sorts/topological_sort.zig`

Result:
- 8/8 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test sorts/external_sort.zig` ✅
- `zig test sorts/intro_sort.zig` ✅
- `zig test sorts/msd_radix_sort.zig` ✅
- `zig test sorts/natural_sort.zig` ✅
- `zig test sorts/odd_even_transposition_parallel.zig` ✅
- `zig test sorts/patience_sort.zig` ✅
- `zig test sorts/tim_sort.zig` ✅
- `zig test sorts/topological_sort.zig` ✅
- `zig build test` ✅

Failure Log:
- Failing step/command:
  - `zig test sorts/odd_even_transposition_parallel.zig`
  - Symptom: compile error `operator < not allowed for type 'bool'`.
  - Root cause: generic comparator used direct `<`, but Zig bool type does not support ordering operators.
  - Fix applied: added type-aware `lessThan` helper converting bool to integer order for comparison.
  - Post-fix verification: `zig test sorts/odd_even_transposition_parallel.zig` passed.
- Failing step/command:
  - `zig test sorts/intro_sort.zig`
  - Symptom: compile error `local variable is never mutated`.
  - Root cause: loop-range boundary `start` declared as mutable variable though never reassigned.
  - Fix applied: changed declaration from `var` to `const`.
  - Post-fix verification: `zig test sorts/intro_sort.zig` passed.

## Phase 5 Batch C - Wave 3 (2026-03-04)

Scope:
- `sorts/bitonic_sort.zig`
- `sorts/circle_sort.zig`
- `sorts/dutch_national_flag_sort.zig`
- `sorts/odd_even_transposition_single_threaded.zig`
- `sorts/recursive_mergesort_array.zig`
- `sorts/slowsort.zig`
- `sorts/strand_sort.zig`
- `sorts/tree_sort.zig`
- `sorts/unknown_sort.zig`
- `sorts/merge_insertion_sort.zig`

Result:
- 10/10 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test sorts/bitonic_sort.zig` ✅
- `zig test sorts/circle_sort.zig` ✅
- `zig test sorts/dutch_national_flag_sort.zig` ✅
- `zig test sorts/odd_even_transposition_single_threaded.zig` ✅
- `zig test sorts/recursive_mergesort_array.zig` ✅
- `zig test sorts/slowsort.zig` ✅
- `zig test sorts/strand_sort.zig` ✅
- `zig test sorts/tree_sort.zig` ✅
- `zig test sorts/unknown_sort.zig` ✅
- `zig test sorts/merge_insertion_sort.zig` ✅
- `zig build test` ✅

Failure Log:
- No implementation/test failures encountered in this wave.

## Phase 5 Batch C - Wave 1 (2026-03-04)

Scope:
- `sorts/binary_insertion_sort.zig`
- `sorts/bogo_sort.zig`
- `sorts/comb_sort.zig`
- `sorts/cycle_sort.zig`
- `sorts/double_sort.zig`
- `sorts/odd_even_sort.zig`
- `sorts/pancake_sort.zig`
- `sorts/recursive_insertion_sort.zig`
- `sorts/stooge_sort.zig`
- `sorts/wiggle_sort.zig`

Result:
- 10/10 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test sorts/binary_insertion_sort.zig` ✅
- `zig test sorts/bogo_sort.zig` ✅
- `zig test sorts/comb_sort.zig` ✅
- `zig test sorts/cycle_sort.zig` ✅
- `zig test sorts/double_sort.zig` ✅
- `zig test sorts/odd_even_sort.zig` ✅
- `zig test sorts/pancake_sort.zig` ✅
- `zig test sorts/recursive_insertion_sort.zig` ✅
- `zig test sorts/stooge_sort.zig` ✅
- `zig test sorts/wiggle_sort.zig` ✅
- `zig build test` ✅

Failure Log:
- No implementation/test failures encountered in this wave.

## Phase 5 Batch B - Wave 2 (2026-03-04)

Scope:
- `strings/alternative_string_arrange.zig`
- `strings/boyer_moore_search.zig`
- `strings/bitap_string_match.zig`
- `strings/prefix_function.zig`
- `strings/remove_duplicate.zig`
- `strings/reverse_letters.zig`
- `strings/snake_case_to_camel_pascal_case.zig`
- `strings/strip.zig`
- `strings/title.zig`
- `strings/word_occurrence.zig`

Result:
- 10/10 implementations completed and registered in `build.zig`.
- All files include normal + boundary + extreme-case tests.
- Python-reference behavior aligned for covered input domains.

Verification:
- `zig test strings/alternative_string_arrange.zig` ✅
- `zig test strings/boyer_moore_search.zig` ✅
- `zig test strings/bitap_string_match.zig` ✅
- `zig test strings/prefix_function.zig` ✅
- `zig test strings/remove_duplicate.zig` ✅
- `zig test strings/reverse_letters.zig` ✅
- `zig test strings/snake_case_to_camel_pascal_case.zig` ✅
- `zig test strings/strip.zig` ✅
- `zig test strings/title.zig` ✅
- `zig test strings/word_occurrence.zig` ✅
- `zig build test` ✅

Failure Log:
- Failing step/command:
  - `zig test strings/alternative_string_arrange.zig` (first run in batch loop)
  - Symptom: build failed with `manifest_create AccessDenied` when creating Zig cache.
  - Root cause: default cache location was not writable in current sandbox context.
  - Fix applied: set `ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache` and `ZIG_LOCAL_CACHE_DIR=/tmp/zig-local-cache` before running Zig test commands.
  - Post-fix verification: full wave file tests and `zig build test` passed.
- Failing step/command:
  - `zig test strings/word_occurrence.zig`
  - Symptom: test assertion mismatch (`expected 5, found 6`) for count of `"e"`.
  - Root cause: expected value in test did not match Python sample sentence frequency.
  - Fix applied: corrected expected count of `"e"` from `5` to `6`.
  - Post-fix verification: `zig test strings/word_occurrence.zig` passed.

## Phase 5 Batch B - Wave 1 (2026-03-04)

Scope:
- `strings/camel_case_to_snake_case.zig`
- `strings/can_string_be_rearranged_as_palindrome.zig`
- `strings/capitalize.zig`
- `strings/count_vowels.zig`
- `strings/is_contains_unique_chars.zig`
- `strings/is_isogram.zig`
- `strings/join.zig`
- `strings/lower.zig`
- `strings/split.zig`
- `strings/upper.zig`

Result:
- 10/10 implementations completed and registered in `build.zig`.
- Tests cover normal + boundary + extreme scenarios and align to Python behavior in covered domain.

Verification:
- `zig test strings/camel_case_to_snake_case.zig` ✅
- `zig test strings/can_string_be_rearranged_as_palindrome.zig` ✅
- `zig test strings/capitalize.zig` ✅
- `zig test strings/count_vowels.zig` ✅
- `zig test strings/is_contains_unique_chars.zig` ✅
- `zig test strings/is_isogram.zig` ✅
- `zig test strings/join.zig` ✅
- `zig test strings/lower.zig` ✅
- `zig test strings/split.zig` ✅
- `zig test strings/upper.zig` ✅
- `zig build test` ✅

Failure Log:
- No implementation/test failures encountered in this wave.

## Phase 5 Batch A - Wave 2 (2026-03-04)

Scope:
- `maths/twin_prime.zig`
- `maths/lucas_series.zig`
- `maths/josephus_problem.zig`
- `maths/sum_of_digits.zig`
- `maths/number_of_digits.zig`
- `maths/is_int_palindrome.zig`
- `maths/perfect_square.zig`
- `maths/perfect_cube.zig`
- `maths/quadratic_equations_complex_numbers.zig`
- `maths/decimal_to_fraction.zig`

Result:
- 10/10 implementations completed and registered in `build.zig`.
- Test suites include normal + boundary + extreme scenarios.
- Python-reference behavior aligned for target input domains.

Verification:
- `zig test maths/twin_prime.zig` ✅
- `zig test maths/lucas_series.zig` ✅
- `zig test maths/josephus_problem.zig` ✅
- `zig test maths/sum_of_digits.zig` ✅
- `zig test maths/number_of_digits.zig` ✅
- `zig test maths/is_int_palindrome.zig` ✅
- `zig test maths/perfect_square.zig` ✅
- `zig test maths/perfect_cube.zig` ✅
- `zig test maths/quadratic_equations_complex_numbers.zig` ✅
- `zig test maths/decimal_to_fraction.zig` ✅
- `zig build test` ✅

Failure Log:
- Failing step/command:
  - `zig test maths/number_of_digits.zig`
  - Symptom: compile error `@intFromFloat must have a known result type`.
  - Root cause: result type inference for `@intFromFloat` was ambiguous in expression context.
  - Fix applied: explicit cast added (`@as(u32, @intFromFloat(...))`).
  - Post-fix verification: `zig test maths/number_of_digits.zig` passed.
- Failing step/command:
  - `zig test maths/perfect_cube.zig`
  - Symptom: runtime panic `integer overflow` in `mid * mid * mid`.
  - Root cause: cube computation overflowed `u128` for large search-mid values.
  - Fix applied: switched to overflow-safe cube comparison (`@mulWithOverflow` + division guard) before multiplication.
  - Post-fix verification: `zig test maths/perfect_cube.zig` passed.

## Phase 5 Batch A - Wave 3 (2026-03-04)

Scope:
- `maths/armstrong_numbers.zig`
- `maths/automorphic_number.zig`
- `maths/catalan_number.zig`
- `maths/happy_number.zig`
- `maths/hexagonal_number.zig`
- `maths/pronic_number.zig`
- `maths/proth_number.zig`
- `maths/triangular_numbers.zig`
- `maths/hamming_numbers.zig`
- `maths/polygonal_numbers.zig`

Result:
- 10/10 implementations completed and registered in `build.zig`.
- Test sets cover normal + boundary + extreme scenarios.
- Python-reference behavior aligned for selected scope.

Verification:
- `zig test maths/armstrong_numbers.zig` ✅
- `zig test maths/automorphic_number.zig` ✅
- `zig test maths/catalan_number.zig` ✅
- `zig test maths/happy_number.zig` ✅
- `zig test maths/hexagonal_number.zig` ✅
- `zig test maths/pronic_number.zig` ✅
- `zig test maths/proth_number.zig` ✅
- `zig test maths/triangular_numbers.zig` ✅
- `zig test maths/hamming_numbers.zig` ✅
- `zig test maths/polygonal_numbers.zig` ✅
- `zig build test` ✅

Failure Log:
- Failing step/command:
  - `zig test maths/pronic_number.zig`
  - Symptom: compile error for `%` with signed integer (`i64`) and comptime int.
  - Root cause: Zig requires `@mod/@rem` for signed remainder semantics.
  - Fix applied: replaced `%` expression with `@mod(number, 2)`.
  - Post-fix verification: `zig test maths/pronic_number.zig` passed.
- Failing step/command:
  - `zig test maths/polygonal_numbers.zig`
  - Symptom: test failed with `Overflow` on valid case `polygonalNum(0, 3)`.
  - Root cause: direct unsigned subtraction for `(sides - 4)` underflowed when `sides == 3`.
  - Fix applied: added dedicated triangular-number branch for `sides == 3`; retained generic formula for `sides >= 4`.
  - Post-fix verification: `zig test maths/polygonal_numbers.zig` passed.

## Phase 5 Batch A - Wave 4 (2026-03-04)

Scope:
- `maths/average_mean.zig`
- `maths/average_median.zig`
- `maths/average_mode.zig`
- `maths/find_max.zig`
- `maths/find_min.zig`
- `maths/factors.zig`
- `maths/geometric_mean.zig`
- `maths/line_length.zig`
- `maths/euclidean_distance.zig`
- `maths/manhattan_distance.zig`

Result:
- 10/10 implementations completed and registered in `build.zig`.
- Python-reference behavior aligned for the covered test domains.
- All files include normal + boundary + extreme-case tests.

Verification:
- `zig test maths/average_mean.zig` ✅
- `zig test maths/average_median.zig` ✅
- `zig test maths/average_mode.zig` ✅
- `zig test maths/find_max.zig` ✅
- `zig test maths/find_min.zig` ✅
- `zig test maths/factors.zig` ✅
- `zig test maths/geometric_mean.zig` ✅
- `zig test maths/line_length.zig` ✅
- `zig test maths/euclidean_distance.zig` ✅
- `zig test maths/manhattan_distance.zig` ✅
- `zig build test` ✅

Failure Log:
- No implementation/test failures encountered in this wave.

## Phase 5 Batch A - Wave 5 (2026-03-04)

Scope:
- `maths/abs.zig`
- `maths/average_absolute_deviation.zig`
- `maths/chebyshev_distance.zig`
- `maths/minkowski_distance.zig`
- `maths/jaccard_similarity.zig`
- `maths/decimal_isolate.zig`
- `maths/floor.zig`
- `maths/ceil.zig`
- `maths/signum.zig`
- `maths/remove_digit.zig`

Result:
- 10/10 implementations completed and registered in `build.zig`.
- Python-reference behavior aligned for tested domains, including edge/extreme scenarios.

Verification:
- `zig test maths/abs.zig` ✅
- `zig test maths/average_absolute_deviation.zig` ✅
- `zig test maths/chebyshev_distance.zig` ✅
- `zig test maths/minkowski_distance.zig` ✅
- `zig test maths/jaccard_similarity.zig` ✅
- `zig test maths/decimal_isolate.zig` ✅
- `zig test maths/floor.zig` ✅
- `zig test maths/ceil.zig` ✅
- `zig test maths/signum.zig` ✅
- `zig test maths/remove_digit.zig` ✅
- `zig build test` ✅

Failure Log:
- Failing step/command:
  - `zig fmt maths/jaccard_similarity.zig`
  - Symptom: parse error `expected 'an identifier', found 'union'`.
  - Root cause: used Zig keyword `union` as local variable name.
  - Fix applied: renamed variable to `merged`.
  - Post-fix verification: `zig test maths/jaccard_similarity.zig` passed.
- Failing step/command:
  - `zig test maths/floor.zig` and `zig test maths/ceil.zig`
  - Symptom: extreme-case assertion mismatch at very large integer literals.
  - Root cause: test used values above `f64` exact integer precision (beyond 2^53-1), causing representation drift.
  - Fix applied: replaced extremes with precision-safe boundary (`2^53 - 1`).
  - Post-fix verification: both test files passed.
- Failing step/command:
  - `zig test maths/remove_digit.zig`
  - Symptom: expected value mismatch for `removeDigit(maxInt(i64))`.
  - Root cause: incorrect expected value in test (not Python-aligned).
  - Fix applied: corrected expected value to `923372036854775807` per Python reference behavior.
  - Post-fix verification: `zig test maths/remove_digit.zig` passed.

## Phase 5 Batch A - Wave 6 (2026-03-04)

Scope:
- `maths/addition_without_arithmetic.zig`
- `maths/arc_length.zig`
- `maths/check_polygon.zig`
- `maths/combinations.zig`
- `maths/double_factorial.zig`
- `maths/pythagoras.zig`
- `maths/sum_of_arithmetic_series.zig`
- `maths/sum_of_geometric_progression.zig`
- `maths/sum_of_harmonic_series.zig`
- `maths/sylvester_sequence.zig`

Result:
- 10/10 implementations completed and registered in `build.zig`.
- Tests cover normal + boundary + extreme scenarios and align to Python behavior in covered domain.

Verification:
- `zig test maths/addition_without_arithmetic.zig` ✅
- `zig test maths/arc_length.zig` ✅
- `zig test maths/check_polygon.zig` ✅
- `zig test maths/combinations.zig` ✅
- `zig test maths/double_factorial.zig` ✅
- `zig test maths/pythagoras.zig` ✅
- `zig test maths/sum_of_arithmetic_series.zig` ✅
- `zig test maths/sum_of_geometric_progression.zig` ✅
- `zig test maths/sum_of_harmonic_series.zig` ✅
- `zig test maths/sylvester_sequence.zig` ✅
- `zig build test` ✅

Failure Log:
- Failing step/command:
  - `zig test maths/sum_of_arithmetic_series.zig`
  - Symptom: assertion mismatch on negative-term case.
  - Root cause: expected test value was set incorrectly (`-55`) while Python formula output is `45.0` for `(1, 1, -10)`.
  - Fix applied: corrected expected value to `45.0`.
  - Post-fix verification: `zig test maths/sum_of_arithmetic_series.zig` passed.

## Phase 5 Batch A - Wave 7 (2026-03-04)

Scope:
- `maths/two_sum.zig`
- `maths/two_pointer.zig`
- `maths/three_sum.zig`
- `maths/triplet_sum.zig`
- `maths/sumset.zig`
- `maths/max_sum_sliding_window.zig`
- `maths/sock_merchant.zig`
- `maths/polynomial_evaluation.zig`
- `maths/kth_lexicographic_permutation.zig`
- `maths/largest_of_very_large_numbers.zig`

Result:
- 10/10 implementations completed and registered in `build.zig`.
- All functions include normal + boundary + extreme-case coverage.
- Python-reference behavior aligned for covered scenarios.

Verification:
- `zig test maths/two_sum.zig` ✅
- `zig test maths/two_pointer.zig` ✅
- `zig test maths/three_sum.zig` ✅
- `zig test maths/triplet_sum.zig` ✅
- `zig test maths/sumset.zig` ✅
- `zig test maths/max_sum_sliding_window.zig` ✅
- `zig test maths/sock_merchant.zig` ✅
- `zig test maths/polynomial_evaluation.zig` ✅
- `zig test maths/kth_lexicographic_permutation.zig` ✅
- `zig test maths/largest_of_very_large_numbers.zig` ✅
- `zig build test` ✅

Failure Log:
- No implementation/test failures encountered in this wave.
