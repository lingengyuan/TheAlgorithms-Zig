//! Resistor Equivalence - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/resistor_equivalence.py

const std = @import("std");
const testing = std.testing;

pub const ResistorError = error{
    EmptyInput,
    NonPositiveParallelResistor,
    NegativeSeriesResistor,
};

/// Computes equivalent resistance of parallel-connected resistors:
/// Req = 1 / (1/R1 + ... + 1/Rn).
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn resistorParallel(resistors: []const f64) ResistorError!f64 {
    if (resistors.len == 0) return ResistorError.EmptyInput;

    var sum_inv: f64 = 0.0;
    for (resistors) |r| {
        if (r <= 0) return ResistorError.NonPositiveParallelResistor;
        sum_inv += 1.0 / r;
    }

    return 1.0 / sum_inv;
}

/// Computes equivalent resistance of series-connected resistors:
/// Req = R1 + ... + Rn.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn resistorSeries(resistors: []const f64) ResistorError!f64 {
    var sum_r: f64 = 0.0;
    for (resistors) |r| {
        sum_r += r;
        if (r < 0) return ResistorError.NegativeSeriesResistor;
    }
    return sum_r;
}

test "resistor equivalence: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 0.8737571620498019), try resistorParallel(&[_]f64{ 3.21389, 2, 3 }), 1e-15);
    try testing.expectError(ResistorError.NonPositiveParallelResistor, resistorParallel(&[_]f64{ 3.21389, 2, -3 }));
    try testing.expectError(ResistorError.NonPositiveParallelResistor, resistorParallel(&[_]f64{ 3.21389, 2, 0.0 }));

    try testing.expectApproxEqAbs(@as(f64, 8.21389), try resistorSeries(&[_]f64{ 3.21389, 2, 3 }), 1e-12);
    try testing.expectError(ResistorError.NegativeSeriesResistor, resistorSeries(&[_]f64{ 3.21389, 2, -3 }));
}

test "resistor equivalence: boundary and extreme values" {
    try testing.expectError(ResistorError.EmptyInput, resistorParallel(&[_]f64{}));
    try testing.expectApproxEqAbs(@as(f64, 0.0), try resistorSeries(&[_]f64{}), 1e-12);

    const huge_parallel = try resistorParallel(&[_]f64{ 1e200, 1e200, 1e200 });
    try testing.expect(std.math.isFinite(huge_parallel));
    try testing.expect(huge_parallel > 0);

    const huge_series = try resistorSeries(&[_]f64{ 1e200, 2e200 });
    try testing.expect(std.math.isFinite(huge_series));
    try testing.expect(huge_series > 0);
}
