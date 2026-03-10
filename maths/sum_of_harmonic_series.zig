//! Sum of Harmonic Progression - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/sum_of_harmonic_series.py

const std = @import("std");
const testing = std.testing;

pub const HarmonicError = error{InvalidFirstTerm};

/// Returns sum of first `number_of_terms` terms in a harmonic progression linked to an arithmetic progression.
/// Time complexity: O(n), Space complexity: O(1)
pub fn sumOfHarmonicProgression(
    first_term: f64,
    common_difference: f64,
    number_of_terms: i64,
) HarmonicError!f64 {
    if (first_term == 0.0) return HarmonicError.InvalidFirstTerm;
    if (number_of_terms == 0) return 0.0;

    var ap_term = 1.0 / first_term;
    var total = 1.0 / ap_term;

    var i: i64 = 0;
    while (i < number_of_terms - 1) : (i += 1) {
        ap_term += common_difference;
        total += 1.0 / ap_term;
    }
    return total;
}

test "sum of harmonic series: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 0.75), try sumOfHarmonicProgression(1.0 / 2.0, 2.0, 2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.45666666666666667), try sumOfHarmonicProgression(1.0 / 5.0, 5.0, 5), 1e-12);
}

test "sum of harmonic series: edge and extreme cases" {
    try testing.expectError(HarmonicError.InvalidFirstTerm, sumOfHarmonicProgression(0.0, 1.0, 5));
    try testing.expectApproxEqAbs(@as(f64, 0.0), try sumOfHarmonicProgression(2.0, 3.0, 0), 1e-12);
    try testing.expect(try sumOfHarmonicProgression(1.0 / 3.0, 1.0, 100_000) > 0.0);
}
