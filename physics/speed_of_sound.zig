//! Speed of Sound in a Fluid - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/speed_of_sound.py

const std = @import("std");
const testing = std.testing;

pub const SpeedOfSoundError = error{
    ImpossibleDensity,
    ImpossibleBulkModulus,
};

/// Computes speed of sound in a fluid:
/// c = sqrt(bulk_modulus / density).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn speedOfSoundInAFluid(density: f64, bulk_modulus: f64) SpeedOfSoundError!f64 {
    if (density <= 0) return SpeedOfSoundError.ImpossibleDensity;
    if (bulk_modulus <= 0) return SpeedOfSoundError.ImpossibleBulkModulus;
    return @sqrt(bulk_modulus / density);
}

test "speed of sound: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 1467.7563207952705), try speedOfSoundInAFluid(998, 2.15e9), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 1447.614670861731), try speedOfSoundInAFluid(13600, 28.5e9), 1e-9);
}

test "speed of sound: invalid inputs" {
    try testing.expectError(SpeedOfSoundError.ImpossibleDensity, speedOfSoundInAFluid(0, 2.15e9));
    try testing.expectError(SpeedOfSoundError.ImpossibleDensity, speedOfSoundInAFluid(-1, 2.15e9));
    try testing.expectError(SpeedOfSoundError.ImpossibleBulkModulus, speedOfSoundInAFluid(998, 0));
    try testing.expectError(SpeedOfSoundError.ImpossibleBulkModulus, speedOfSoundInAFluid(998, -1));
}

test "speed of sound: boundary and extreme values" {
    const tiny = try speedOfSoundInAFluid(1e-9, 1e-3);
    try testing.expect(std.math.isFinite(tiny));
    try testing.expect(tiny > 0);

    const huge = try speedOfSoundInAFluid(1e-12, 1e20);
    try testing.expect(std.math.isFinite(huge));
    try testing.expect(huge > 0);
}
