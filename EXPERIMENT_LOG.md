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
