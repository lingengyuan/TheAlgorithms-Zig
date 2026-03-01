//! Integer to Roman - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/roman_numerals.py

const std = @import("std");
const testing = std.testing;

pub const RomanError = error{OutOfRange};

const TABLE = [_]struct { value: u16, numeral: []const u8 }{
    .{ .value = 1000, .numeral = "M" },
    .{ .value = 900, .numeral = "CM" },
    .{ .value = 500, .numeral = "D" },
    .{ .value = 400, .numeral = "CD" },
    .{ .value = 100, .numeral = "C" },
    .{ .value = 90, .numeral = "XC" },
    .{ .value = 50, .numeral = "L" },
    .{ .value = 40, .numeral = "XL" },
    .{ .value = 10, .numeral = "X" },
    .{ .value = 9, .numeral = "IX" },
    .{ .value = 5, .numeral = "V" },
    .{ .value = 4, .numeral = "IV" },
    .{ .value = 1, .numeral = "I" },
};

/// Converts an integer to canonical Roman numeral form.
/// Valid domain is [1, 3999]. Caller owns returned slice.
/// Time complexity: O(1) for bounded domain.
pub fn integerToRoman(allocator: std.mem.Allocator, number: u32) (RomanError || std.mem.Allocator.Error)![]u8 {
    if (number == 0 or number > 3999) return RomanError.OutOfRange;

    var n: u32 = number;
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    for (TABLE) |entry| {
        const value: u32 = entry.value;
        while (n >= value) {
            try out.appendSlice(allocator, entry.numeral);
            n -= value;
        }
        if (n == 0) break;
    }

    return try out.toOwnedSlice(allocator);
}

test "integer to roman: known values" {
    const alloc = testing.allocator;

    const cases = [_]struct { number: u32, roman: []const u8 }{
        .{ .number = 3, .roman = "III" },
        .{ .number = 154, .roman = "CLIV" },
        .{ .number = 1009, .roman = "MIX" },
        .{ .number = 2500, .roman = "MMD" },
        .{ .number = 3999, .roman = "MMMCMXCIX" },
    };

    for (cases) |c| {
        const s = try integerToRoman(alloc, c.number);
        defer alloc.free(s);
        try testing.expectEqualStrings(c.roman, s);
    }
}

test "integer to roman: out of range" {
    const alloc = testing.allocator;
    try testing.expectError(RomanError.OutOfRange, integerToRoman(alloc, 0));
    try testing.expectError(RomanError.OutOfRange, integerToRoman(alloc, 4000));
}

test "integer to roman: extreme boundaries" {
    const alloc = testing.allocator;

    const min = try integerToRoman(alloc, 1);
    defer alloc.free(min);
    try testing.expectEqualStrings("I", min);

    const max = try integerToRoman(alloc, 3999);
    defer alloc.free(max);
    try testing.expectEqualStrings("MMMCMXCIX", max);
}
