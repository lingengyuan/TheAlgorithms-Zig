//! Doppler Frequency - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/doppler_frequency.py

const std = @import("std");
const testing = std.testing;

pub const DopplerError = error{
    DivisionByZero,
    NonPositiveFrequency,
};

/// Computes observed wave frequency via Doppler effect:
/// f = f0 * (v + v_observer) / (v - v_source).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn dopplerEffect(
    original_frequency: f64,
    wave_velocity: f64,
    observer_velocity: f64,
    source_velocity: f64,
) DopplerError!f64 {
    if (wave_velocity == source_velocity) {
        return DopplerError.DivisionByZero;
    }

    const shifted = (original_frequency * (wave_velocity + observer_velocity)) / (wave_velocity - source_velocity);
    if (shifted <= 0) {
        return DopplerError.NonPositiveFrequency;
    }
    return shifted;
}

test "doppler frequency: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 103.03030303030303), try dopplerEffect(100, 330, 10, 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 96.96969696969697), try dopplerEffect(100, 330, -10, 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 103.125), try dopplerEffect(100, 330, 0, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 97.05882352941177), try dopplerEffect(100, 330, 0, -10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 106.25), try dopplerEffect(100, 330, 10, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 94.11764705882354), try dopplerEffect(100, 330, -10, -10), 1e-12);
}

test "doppler frequency: validation and boundary values" {
    try testing.expectError(DopplerError.DivisionByZero, dopplerEffect(100, 330, 10, 330));
    try testing.expectError(DopplerError.NonPositiveFrequency, dopplerEffect(100, 330, 10, 340));
    try testing.expectError(DopplerError.NonPositiveFrequency, dopplerEffect(100, 330, -340, 10));

    const extreme = try dopplerEffect(1e12, 3e8, 1e5, -1e5);
    try testing.expect(std.math.isFinite(extreme));
    try testing.expect(extreme > 0);
}
