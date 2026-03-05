//! Electric Conductivity - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/electric_conductivity.py

const std = @import("std");
const testing = std.testing;

pub const ElectricConductivityError = error{
    InvalidKnownValuesCount,
    NegativeConductivity,
    NegativeElectronConcentration,
    NegativeMobility,
};

pub const electricCharge: f64 = 1.6021e-19;

pub const ElectricConductivityResult = union(enum) {
    conductivity: f64,
    electron_conc: f64,
    mobility: f64,
};

fn countZeros(conductivity: f64, electron_conc: f64, mobility: f64) u8 {
    var count: u8 = 0;
    if (conductivity == 0) count += 1;
    if (electron_conc == 0) count += 1;
    if (mobility == 0) count += 1;
    return count;
}

/// Solves one missing variable in conductivity relation:
/// conductivity = electron_conc * mobility * electron_charge.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn electricConductivity(
    conductivity: f64,
    electron_conc: f64,
    mobility: f64,
) ElectricConductivityError!ElectricConductivityResult {
    if (countZeros(conductivity, electron_conc, mobility) != 1) {
        return ElectricConductivityError.InvalidKnownValuesCount;
    }
    if (conductivity < 0) return ElectricConductivityError.NegativeConductivity;
    if (electron_conc < 0) return ElectricConductivityError.NegativeElectronConcentration;
    if (mobility < 0) return ElectricConductivityError.NegativeMobility;

    if (conductivity == 0) {
        return ElectricConductivityResult{ .conductivity = mobility * electron_conc * electricCharge };
    } else if (electron_conc == 0) {
        return ElectricConductivityResult{ .electron_conc = conductivity / (mobility * electricCharge) };
    } else {
        return ElectricConductivityResult{ .mobility = conductivity / (electron_conc * electricCharge) };
    }
}

test "electric conductivity: python examples" {
    const m = try electricConductivity(25, 100, 0);
    switch (m) {
        .mobility => |value| try testing.expectApproxEqAbs(@as(f64, 1.5604519068722301e18), value, 1e6),
        else => try testing.expect(false),
    }

    const c = try electricConductivity(0, 1600, 200);
    switch (c) {
        .conductivity => |value| try testing.expectApproxEqAbs(@as(f64, 5.12672e-14), value, 1e-24),
        else => try testing.expect(false),
    }

    const e = try electricConductivity(1000, 0, 1200);
    switch (e) {
        .electron_conc => |value| try testing.expectApproxEqAbs(@as(f64, 5.201506356240767e18), value, 1e6),
        else => try testing.expect(false),
    }

    try testing.expectError(ElectricConductivityError.NegativeConductivity, electricConductivity(-10, 100, 0));
    try testing.expectError(ElectricConductivityError.NegativeElectronConcentration, electricConductivity(50, -10, 0));
    try testing.expectError(ElectricConductivityError.NegativeMobility, electricConductivity(50, 0, -10));
    try testing.expectError(ElectricConductivityError.InvalidKnownValuesCount, electricConductivity(50, 0, 0));
    try testing.expectError(ElectricConductivityError.InvalidKnownValuesCount, electricConductivity(50, 200, 300));
}

test "electric conductivity: boundary and extreme values" {
    const huge = try electricConductivity(0, 1e30, 1e30);
    switch (huge) {
        .conductivity => |value| {
            try testing.expect(std.math.isFinite(value));
            try testing.expect(value > 0);
        },
        else => try testing.expect(false),
    }
}
