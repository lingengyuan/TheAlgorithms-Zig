//! Hubble Parameter - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/hubble_parameter.py

const std = @import("std");
const testing = std.testing;

pub const HubbleError = error{
    NegativeParameter,
    DensityGreaterThanOne,
};

/// Computes Hubble parameter at redshift z:
/// H(z) = H0 * sqrt(Ωr(1+z)^4 + Ωm(1+z)^3 + Ωk(1+z)^2 + ΩΛ),
/// with Ωk = 1 - (Ωm + Ωr + ΩΛ).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn hubbleParameter(
    hubble_constant: f64,
    radiation_density: f64,
    matter_density: f64,
    dark_energy: f64,
    redshift: f64,
) HubbleError!f64 {
    if (redshift < 0 or radiation_density < 0 or matter_density < 0 or dark_energy < 0) {
        return HubbleError.NegativeParameter;
    }
    if (radiation_density > 1 or matter_density > 1 or dark_energy > 1) {
        return HubbleError.DensityGreaterThanOne;
    }

    const curvature = 1.0 - (matter_density + radiation_density + dark_energy);
    const one_plus_z = redshift + 1.0;
    const e_squared =
        radiation_density * std.math.pow(f64, one_plus_z, 4) +
        matter_density * std.math.pow(f64, one_plus_z, 3) +
        curvature * std.math.pow(f64, one_plus_z, 2) +
        dark_energy;

    return hubble_constant * @sqrt(e_squared);
}

test "hubble parameter: python examples" {
    try testing.expectError(HubbleError.NegativeParameter, hubbleParameter(68.3, 1e-4, -0.3, 0.7, 1));
    try testing.expectError(HubbleError.DensityGreaterThanOne, hubbleParameter(68.3, 1e-4, 1.2, 0.7, 1));
    try testing.expectApproxEqAbs(@as(f64, 68.3), try hubbleParameter(68.3, 1e-4, 0.3, 0.7, 0), 1e-12);
}

test "hubble parameter: boundary and extreme values" {
    const lcdm = try hubbleParameter(68.3, 1e-4, 0.3, 1 - 0.3, 0);
    try testing.expectApproxEqAbs(@as(f64, 68.3), lcdm, 1e-12);

    const high_redshift = try hubbleParameter(70.0, 8e-5, 0.3, 0.69992, 1e3);
    try testing.expect(std.math.isFinite(high_redshift));
    try testing.expect(high_redshift > 0);
}
