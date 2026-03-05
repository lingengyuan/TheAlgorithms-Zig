//! Capacitor Equivalence - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/capacitor_equivalence.py

const std = @import("std");
const testing = std.testing;

pub const CapacitorError = error{
    NegativeParallelCapacitor,
    NonPositiveSeriesCapacitor,
    EmptySeriesInput,
};

/// Computes equivalent capacitance in parallel:
/// Ceq = C1 + C2 + ... + Cn.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn capacitorParallel(capacitors: []const f64) CapacitorError!f64 {
    var sum_c: f64 = 0.0;
    for (capacitors) |capacitor| {
        if (capacitor < 0) return CapacitorError.NegativeParallelCapacitor;
        sum_c += capacitor;
    }
    return sum_c;
}

/// Computes equivalent capacitance in series:
/// Ceq = 1 / (1/C1 + 1/C2 + ... + 1/Cn).
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn capacitorSeries(capacitors: []const f64) CapacitorError!f64 {
    if (capacitors.len == 0) return CapacitorError.EmptySeriesInput;

    var sum_inv: f64 = 0.0;
    for (capacitors) |capacitor| {
        if (capacitor <= 0) return CapacitorError.NonPositiveSeriesCapacitor;
        sum_inv += 1.0 / capacitor;
    }
    return 1.0 / sum_inv;
}

test "capacitor equivalence: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 20.71389), try capacitorParallel(&[_]f64{ 5.71389, 12, 3 }), 1e-12);
    try testing.expectError(CapacitorError.NegativeParallelCapacitor, capacitorParallel(&[_]f64{ 5.71389, 12, -3 }));

    try testing.expectApproxEqAbs(@as(f64, 1.6901062252507735), try capacitorSeries(&[_]f64{ 5.71389, 12, 3 }), 1e-15);
    try testing.expectError(CapacitorError.NonPositiveSeriesCapacitor, capacitorSeries(&[_]f64{ 5.71389, 12, -3 }));
    try testing.expectError(CapacitorError.NonPositiveSeriesCapacitor, capacitorSeries(&[_]f64{ 5.71389, 12, 0.0 }));
}

test "capacitor equivalence: boundary and extreme values" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), try capacitorParallel(&[_]f64{}), 1e-12);
    try testing.expectError(CapacitorError.EmptySeriesInput, capacitorSeries(&[_]f64{}));

    const huge_parallel = try capacitorParallel(&[_]f64{ 1e200, 2e200 });
    try testing.expect(std.math.isFinite(huge_parallel));
    try testing.expect(huge_parallel > 0);

    const tiny_series = try capacitorSeries(&[_]f64{ 1e-50, 1e-50, 1e-50 });
    try testing.expect(std.math.isFinite(tiny_series));
    try testing.expect(tiny_series > 0);
}
