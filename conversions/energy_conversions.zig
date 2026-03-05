//! Energy Conversions - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/energy_conversions.py

const std = @import("std");
const testing = std.testing;

pub const EnergyConversionError = error{InvalidUnit};

const Unit = enum {
    joule,
    kilojoule,
    megajoule,
    gigajoule,
    wattsecond,
    watthour,
    kilowatthour,
    newtonmeter,
    calorie_nutr,
    kilocalorie_nutr,
    electronvolt,
    britishthermalunit_it,
    footpound,
};

fn parseUnit(input: []const u8) ?Unit {
    if (std.mem.eql(u8, input, "joule")) return .joule;
    if (std.mem.eql(u8, input, "kilojoule")) return .kilojoule;
    if (std.mem.eql(u8, input, "megajoule")) return .megajoule;
    if (std.mem.eql(u8, input, "gigajoule")) return .gigajoule;
    if (std.mem.eql(u8, input, "wattsecond")) return .wattsecond;
    if (std.mem.eql(u8, input, "watthour")) return .watthour;
    if (std.mem.eql(u8, input, "kilowatthour")) return .kilowatthour;
    if (std.mem.eql(u8, input, "newtonmeter")) return .newtonmeter;
    if (std.mem.eql(u8, input, "calorie_nutr")) return .calorie_nutr;
    if (std.mem.eql(u8, input, "kilocalorie_nutr")) return .kilocalorie_nutr;
    if (std.mem.eql(u8, input, "electronvolt")) return .electronvolt;
    if (std.mem.eql(u8, input, "britishthermalunit_it")) return .britishthermalunit_it;
    if (std.mem.eql(u8, input, "footpound")) return .footpound;
    return null;
}

fn factor(unit: Unit) f64 {
    return switch (unit) {
        .joule => 1.0,
        .kilojoule => 1_000.0,
        .megajoule => 1_000_000.0,
        .gigajoule => 1_000_000_000.0,
        .wattsecond => 1.0,
        .watthour => 3_600.0,
        .kilowatthour => 3_600_000.0,
        .newtonmeter => 1.0,
        .calorie_nutr => 4_186.8,
        .kilocalorie_nutr => 4_186_800.0,
        .electronvolt => 1.602_176_634e-19,
        .britishthermalunit_it => 1_055.055_85,
        .footpound => 1.355_818,
    };
}

/// Converts energy values between supported units.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn convertEnergy(
    from_type: []const u8,
    to_type: []const u8,
    value: f64,
) EnergyConversionError!f64 {
    const from = parseUnit(from_type) orelse return EnergyConversionError.InvalidUnit;
    const to = parseUnit(to_type) orelse return EnergyConversionError.InvalidUnit;
    return value * factor(from) / factor(to);
}

test "energy conversions: selected python examples" {
    try testing.expectApproxEqAbs(@as(f64, 1.0), try convertEnergy("joule", "joule", 1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.001), try convertEnergy("joule", "kilojoule", 1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1e-06), try convertEnergy("joule", "megajoule", 1), 1e-15);
    try testing.expectApproxEqAbs(@as(f64, 1e-09), try convertEnergy("joule", "gigajoule", 1), 1e-18);
    try testing.expectApproxEqAbs(@as(f64, 0.0002777777777777778), try convertEnergy("joule", "watthour", 1), 1e-18);
    try testing.expectApproxEqAbs(@as(f64, 2.7777777777777776e-07), try convertEnergy("joule", "kilowatthour", 1), 1e-21);
    try testing.expectApproxEqAbs(@as(f64, 6.241509074460763e+18), try convertEnergy("joule", "electronvolt", 1), 1e6);
    try testing.expectApproxEqAbs(@as(f64, 0.7375621211696556), try convertEnergy("joule", "footpound", 1), 1e-15);
    try testing.expectApproxEqAbs(@as(f64, 0.001), try convertEnergy("joule", "megajoule", 1000), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), try convertEnergy("calorie_nutr", "kilocalorie_nutr", 1000), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 36_000_000.0), try convertEnergy("kilowatthour", "joule", 10), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 778.1692306784539), try convertEnergy("britishthermalunit_it", "footpound", 1), 1e-9);
}

test "energy conversions: invalid units" {
    try testing.expectError(EnergyConversionError.InvalidUnit, convertEnergy("wrongunit", "joule", 1));
    try testing.expectError(EnergyConversionError.InvalidUnit, convertEnergy("joule", "wrongunit", 1));
}

test "energy conversions: boundary and extreme values" {
    try testing.expectApproxEqAbs(@as(f64, 0), try convertEnergy("joule", "kilojoule", 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.001), try convertEnergy("joule", "kilojoule", -1), 1e-12);

    const huge = try convertEnergy("gigajoule", "electronvolt", 1e6);
    try testing.expect(huge > 1e30);
}
