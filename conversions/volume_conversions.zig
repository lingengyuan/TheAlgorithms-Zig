//! Volume Conversions - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/volume_conversions.py

const std = @import("std");
const testing = std.testing;

pub const VolumeConversionError = error{ InvalidFromType, InvalidToType };

const Unit = enum {
    cubic_meter,
    litre,
    kilolitre,
    gallon,
    cubic_yard,
    cubic_foot,
    cup,
};

const FromTo = struct {
    from_factor: f64,
    to_factor: f64,
};

fn parseUnit(input: []const u8) ?Unit {
    if (std.mem.eql(u8, input, "cubic meter")) return .cubic_meter;
    if (std.mem.eql(u8, input, "litre")) return .litre;
    if (std.mem.eql(u8, input, "kilolitre")) return .kilolitre;
    if (std.mem.eql(u8, input, "gallon")) return .gallon;
    if (std.mem.eql(u8, input, "cubic yard")) return .cubic_yard;
    if (std.mem.eql(u8, input, "cubic foot")) return .cubic_foot;
    if (std.mem.eql(u8, input, "cup")) return .cup;
    return null;
}

fn factors(unit: Unit) FromTo {
    return switch (unit) {
        .cubic_meter => .{ .from_factor = 1.0, .to_factor = 1.0 },
        .litre => .{ .from_factor = 0.001, .to_factor = 1000.0 },
        .kilolitre => .{ .from_factor = 1.0, .to_factor = 1.0 },
        .gallon => .{ .from_factor = 0.00454, .to_factor = 264.172 },
        .cubic_yard => .{ .from_factor = 0.76455, .to_factor = 1.30795 },
        .cubic_foot => .{ .from_factor = 0.028, .to_factor = 35.3147 },
        .cup => .{ .from_factor = 0.000236588, .to_factor = 4226.75 },
    };
}

/// Converts volume values between supported units.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn convertVolume(
    value: f64,
    from_type: []const u8,
    to_type: []const u8,
) VolumeConversionError!f64 {
    const from = parseUnit(from_type) orelse return VolumeConversionError.InvalidFromType;
    const to = parseUnit(to_type) orelse return VolumeConversionError.InvalidToType;
    return value * factors(from).from_factor * factors(to).to_factor;
}

test "volume conversions: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 4000), try convertVolume(4, "cubic meter", "litre"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.264172), try convertVolume(1, "litre", "gallon"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1), try convertVolume(1, "kilolitre", "cubic meter"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.017814279), try convertVolume(3, "gallon", "cubic yard"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1529.1), try convertVolume(2, "cubic yard", "litre"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 473.396), try convertVolume(4, "cubic foot", "cup"), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.000236588), try convertVolume(1, "cup", "kilolitre"), 1e-12);
}

test "volume conversions: invalid unit handling" {
    try testing.expectError(VolumeConversionError.InvalidFromType, convertVolume(4, "wrongUnit", "litre"));
    try testing.expectError(VolumeConversionError.InvalidToType, convertVolume(4, "litre", "wrongUnit"));
}

test "volume conversions: boundary and extreme values" {
    try testing.expectApproxEqAbs(@as(f64, 0), try convertVolume(0, "cup", "litre"), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -1000), try convertVolume(-1, "cubic meter", "litre"), 1e-9);

    const huge = try convertVolume(1e9, "cubic meter", "cup");
    try testing.expect(huge > 1e12);
}
