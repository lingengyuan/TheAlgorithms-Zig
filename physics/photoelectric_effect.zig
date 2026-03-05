//! Photoelectric Effect - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/photoelectric_effect.py

const std = @import("std");
const testing = std.testing;

pub const PhotoelectricError = error{
    NegativeFrequency,
};

pub const planck_constant_js: f64 = 6.6261e-34;
pub const planck_constant_evs: f64 = 4.1357e-15;

/// Computes maximum kinetic energy of emitted electrons:
/// K_max = h * f - work_function, clamped at 0.
/// `in_ev` selects whether h is in eV·s (`true`) or J·s (`false`).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn maximumKineticEnergy(
    frequency: f64,
    work_function: f64,
    in_ev: bool,
) PhotoelectricError!f64 {
    if (frequency < 0) {
        return PhotoelectricError.NegativeFrequency;
    }

    const energy = if (in_ev)
        planck_constant_evs * frequency - work_function
    else
        planck_constant_js * frequency - work_function;

    return @max(energy, 0.0);
}

test "photoelectric effect: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), try maximumKineticEnergy(1_000_000, 2, false), 1e-18);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try maximumKineticEnergy(1_000_000, 2, true), 1e-18);
    try testing.expectApproxEqAbs(@as(f64, 39.357000000000006), try maximumKineticEnergy(10_000_000_000_000_000, 2, true), 1e-12);
}

test "photoelectric effect: validation and extreme values" {
    try testing.expectError(PhotoelectricError.NegativeFrequency, maximumKineticEnergy(-9, 20, false));

    const zero_floor = try maximumKineticEnergy(1000, 1e10, true);
    try testing.expectApproxEqAbs(@as(f64, 0.0), zero_floor, 1e-18);

    const extreme = try maximumKineticEnergy(1e22, 0.1, true);
    try testing.expect(std.math.isFinite(extreme));
    try testing.expect(extreme > 0);
}
