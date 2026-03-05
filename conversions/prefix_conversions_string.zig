//! SI/Binary Prefix String Conversion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/prefix_conversions_string.py

const std = @import("std");
const testing = std.testing;

const Prefix = struct {
    name: []const u8,
    exponent: i8,
};

const si_positive = [_]Prefix{
    .{ .name = "yotta", .exponent = 24 },
    .{ .name = "zetta", .exponent = 21 },
    .{ .name = "exa", .exponent = 18 },
    .{ .name = "peta", .exponent = 15 },
    .{ .name = "tera", .exponent = 12 },
    .{ .name = "giga", .exponent = 9 },
    .{ .name = "mega", .exponent = 6 },
    .{ .name = "kilo", .exponent = 3 },
    .{ .name = "hecto", .exponent = 2 },
    .{ .name = "deca", .exponent = 1 },
};

const si_negative = [_]Prefix{
    .{ .name = "deci", .exponent = -1 },
    .{ .name = "centi", .exponent = -2 },
    .{ .name = "milli", .exponent = -3 },
    .{ .name = "micro", .exponent = -6 },
    .{ .name = "nano", .exponent = -9 },
    .{ .name = "pico", .exponent = -12 },
    .{ .name = "femto", .exponent = -15 },
    .{ .name = "atto", .exponent = -18 },
    .{ .name = "zepto", .exponent = -21 },
    .{ .name = "yocto", .exponent = -24 },
};

const binary_prefixes = [_]Prefix{
    .{ .name = "yotta", .exponent = 80 },
    .{ .name = "zetta", .exponent = 70 },
    .{ .name = "exa", .exponent = 60 },
    .{ .name = "peta", .exponent = 50 },
    .{ .name = "tera", .exponent = 40 },
    .{ .name = "giga", .exponent = 30 },
    .{ .name = "mega", .exponent = 20 },
    .{ .name = "kilo", .exponent = 10 },
};

fn formatPythonLikeNumber(allocator: std.mem.Allocator, value: f64) std.mem.Allocator.Error![]u8 {
    if (std.math.isFinite(value) and std.math.floor(value) == value) {
        return std.fmt.allocPrint(allocator, "{d:.1}", .{value});
    }
    return std.fmt.allocPrint(allocator, "{d}", .{value});
}

/// Converts a value to SI-prefixed string using the reference's selection logic.
///
/// Time complexity: O(p), p = number of prefixes
/// Space complexity: O(1) auxiliary, O(output) returned string
pub fn addSiPrefix(allocator: std.mem.Allocator, value: f64) std.mem.Allocator.Error![]u8 {
    const prefixes = if (value > 0) &si_positive else &si_negative;
    for (prefixes) |prefix| {
        const part = value / std.math.pow(f64, 10.0, @as(f64, @floatFromInt(prefix.exponent)));
        if (part > 1) {
            const num = try formatPythonLikeNumber(allocator, part);
            defer allocator.free(num);
            return std.fmt.allocPrint(allocator, "{s} {s}", .{ num, prefix.name });
        }
    }
    return formatPythonLikeNumber(allocator, value);
}

/// Converts a value to binary-prefixed string using the reference's selection logic.
///
/// Time complexity: O(p), p = number of prefixes
/// Space complexity: O(1) auxiliary, O(output) returned string
pub fn addBinaryPrefix(allocator: std.mem.Allocator, value: f64) std.mem.Allocator.Error![]u8 {
    for (binary_prefixes) |prefix| {
        const part = value / std.math.pow(f64, 2.0, @as(f64, @floatFromInt(prefix.exponent)));
        if (part > 1) {
            const num = try formatPythonLikeNumber(allocator, part);
            defer allocator.free(num);
            return std.fmt.allocPrint(allocator, "{s} {s}", .{ num, prefix.name });
        }
    }
    return formatPythonLikeNumber(allocator, value);
}

test "prefix conversions string: python examples" {
    const alloc = testing.allocator;

    const si = try addSiPrefix(alloc, 10000);
    defer alloc.free(si);
    try testing.expectEqualStrings("10.0 kilo", si);

    const binary = try addBinaryPrefix(alloc, 65536);
    defer alloc.free(binary);
    try testing.expectEqualStrings("64.0 kilo", binary);
}

test "prefix conversions string: boundary behavior" {
    const alloc = testing.allocator;

    const small = try addSiPrefix(alloc, 0.5);
    defer alloc.free(small);
    try testing.expectEqualStrings("0.5", small);

    const negative = try addSiPrefix(alloc, -3.0);
    defer alloc.free(negative);
    try testing.expectEqualStrings("-3.0", negative);

    const one = try addBinaryPrefix(alloc, 1.0);
    defer alloc.free(one);
    try testing.expectEqualStrings("1.0", one);
}

test "prefix conversions string: extreme values" {
    const alloc = testing.allocator;

    const huge_si = try addSiPrefix(alloc, 2.0 * std.math.pow(f64, 10.0, 24.0));
    defer alloc.free(huge_si);
    try testing.expectEqualStrings("2.0 yotta", huge_si);

    const huge_binary = try addBinaryPrefix(alloc, 2.0 * std.math.pow(f64, 2.0, 80.0));
    defer alloc.free(huge_binary);
    try testing.expectEqualStrings("2.0 yotta", huge_binary);
}
