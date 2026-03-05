//! Resonant Frequency - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/resonant_frequency.py

const std = @import("std");
const testing = std.testing;

pub const ResonantFrequencyError = error{
    NonPositiveInductance,
    NonPositiveCapacitance,
};

pub const ResonantFrequencyResult = struct {
    name: []const u8,
    value: f64,
};

/// Computes LC resonant frequency:
/// f = 1 / (2 * pi * sqrt(L * C)).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn resonantFrequency(inductance: f64, capacitance: f64) ResonantFrequencyError!ResonantFrequencyResult {
    if (inductance <= 0) return ResonantFrequencyError.NonPositiveInductance;
    if (capacitance <= 0) return ResonantFrequencyError.NonPositiveCapacitance;

    return ResonantFrequencyResult{
        .name = "Resonant frequency",
        .value = 1.0 / (2.0 * std.math.pi * @sqrt(inductance * capacitance)),
    };
}

test "resonant frequency: python examples" {
    const r = try resonantFrequency(10, 5);
    try testing.expectEqualStrings("Resonant frequency", r.name);
    try testing.expectApproxEqAbs(@as(f64, 0.022507907903927652), r.value, 1e-18);

    try testing.expectError(ResonantFrequencyError.NonPositiveInductance, resonantFrequency(0, 5));
    try testing.expectError(ResonantFrequencyError.NonPositiveCapacitance, resonantFrequency(10, 0));
}

test "resonant frequency: boundary and extreme values" {
    const high_f = try resonantFrequency(1e-12, 1e-12);
    try testing.expect(std.math.isFinite(high_f.value));
    try testing.expect(high_f.value > 0);

    const low_f = try resonantFrequency(1e12, 1e12);
    try testing.expect(std.math.isFinite(low_f.value));
    try testing.expect(low_f.value > 0);
}
