//! Carrier Concentration - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/carrier_concentration.py

const std = @import("std");
const testing = std.testing;

pub const CarrierConcentrationError = error{
    InvalidKnownValuesCount,
    NegativeElectronConcentration,
    NegativeHoleConcentration,
    NegativeIntrinsicConcentration,
};

pub const CarrierConcentrationResult = union(enum) {
    electron_conc: f64,
    hole_conc: f64,
    intrinsic_conc: f64,
};

fn countZeros(electron_conc: f64, hole_conc: f64, intrinsic_conc: f64) u8 {
    var count: u8 = 0;
    if (electron_conc == 0) count += 1;
    if (hole_conc == 0) count += 1;
    if (intrinsic_conc == 0) count += 1;
    return count;
}

/// Solves one missing semiconductor carrier concentration using n * p = n_i^2.
/// Exactly one of the three values must be zero.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn carrierConcentration(
    electron_conc: f64,
    hole_conc: f64,
    intrinsic_conc: f64,
) CarrierConcentrationError!CarrierConcentrationResult {
    if (countZeros(electron_conc, hole_conc, intrinsic_conc) != 1) {
        return CarrierConcentrationError.InvalidKnownValuesCount;
    }
    if (electron_conc < 0) {
        return CarrierConcentrationError.NegativeElectronConcentration;
    }
    if (hole_conc < 0) {
        return CarrierConcentrationError.NegativeHoleConcentration;
    }
    if (intrinsic_conc < 0) {
        return CarrierConcentrationError.NegativeIntrinsicConcentration;
    }

    if (electron_conc == 0) {
        return CarrierConcentrationResult{ .electron_conc = (intrinsic_conc * intrinsic_conc) / hole_conc };
    }
    if (hole_conc == 0) {
        return CarrierConcentrationResult{ .hole_conc = (intrinsic_conc * intrinsic_conc) / electron_conc };
    }
    return CarrierConcentrationResult{ .intrinsic_conc = @sqrt(electron_conc * hole_conc) };
}

test "carrier concentration: python examples" {
    const r1 = try carrierConcentration(25, 100, 0);
    switch (r1) {
        .intrinsic_conc => |value| try testing.expectApproxEqAbs(@as(f64, 50.0), value, 1e-12),
        else => try testing.expect(false),
    }

    const r2 = try carrierConcentration(0, 1600, 200);
    switch (r2) {
        .electron_conc => |value| try testing.expectApproxEqAbs(@as(f64, 25.0), value, 1e-12),
        else => try testing.expect(false),
    }

    const r3 = try carrierConcentration(1000, 0, 1200);
    switch (r3) {
        .hole_conc => |value| try testing.expectApproxEqAbs(@as(f64, 1440.0), value, 1e-9),
        else => try testing.expect(false),
    }
}

test "carrier concentration: validation and extreme values" {
    try testing.expectError(CarrierConcentrationError.InvalidKnownValuesCount, carrierConcentration(1000, 400, 1200));
    try testing.expectError(CarrierConcentrationError.NegativeElectronConcentration, carrierConcentration(-1000, 0, 1200));
    try testing.expectError(CarrierConcentrationError.NegativeHoleConcentration, carrierConcentration(0, -400, 1200));
    try testing.expectError(CarrierConcentrationError.NegativeIntrinsicConcentration, carrierConcentration(0, 400, -1200));

    const extreme = try carrierConcentration(0, 1e-300, 1e-150);
    switch (extreme) {
        .electron_conc => |value| try testing.expectApproxEqAbs(@as(f64, 1.0), value, 1e-12),
        else => try testing.expect(false),
    }
}
