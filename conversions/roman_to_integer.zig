//! Roman to Integer - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/roman_numerals.py

const std = @import("std");
const testing = std.testing;

pub const RomanError = error{ EmptyInput, InvalidCharacter, InvalidFormat, OutOfRange };

const TABLE = [_]struct { value: u32, numeral: []const u8 }{
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

fn valueOf(c: u8) RomanError!u32 {
    return switch (c) {
        'I' => 1,
        'V' => 5,
        'X' => 10,
        'L' => 50,
        'C' => 100,
        'D' => 500,
        'M' => 1000,
        else => RomanError.InvalidCharacter,
    };
}

fn validSubtractivePair(a: u8, b: u8) bool {
    return (a == 'I' and (b == 'V' or b == 'X')) or
        (a == 'X' and (b == 'L' or b == 'C')) or
        (a == 'C' and (b == 'D' or b == 'M'));
}

fn toCanonicalRoman(number: u32, out: *[16]u8) []const u8 {
    var n = number;
    var pos: usize = 0;

    for (TABLE) |entry| {
        while (n >= entry.value) {
            @memcpy(out[pos .. pos + entry.numeral.len], entry.numeral);
            pos += entry.numeral.len;
            n -= entry.value;
        }
        if (n == 0) break;
    }

    return out[0..pos];
}

/// Converts canonical Roman numeral in range [I, MMMCMXCIX] to integer.
/// Non-canonical forms are rejected (e.g. `IIV`, `IL`, `IIII`).
/// Time complexity: O(n), space complexity: O(1)
pub fn romanToInteger(roman: []const u8) RomanError!u32 {
    if (roman.len == 0) return RomanError.EmptyInput;

    var total: u32 = 0;
    var i: usize = 0;
    while (i < roman.len) {
        const cur_char = roman[i];
        const cur = try valueOf(cur_char);

        if (i + 1 < roman.len) {
            const next_char = roman[i + 1];
            const next = try valueOf(next_char);

            if (cur < next) {
                if (!validSubtractivePair(cur_char, next_char)) return RomanError.InvalidFormat;
                total += next - cur;
                i += 2;
                continue;
            }
        }

        total += cur;
        i += 1;
    }

    if (total == 0 or total > 3999) return RomanError.OutOfRange;

    var canonical_buf: [16]u8 = undefined;
    const canonical = toCanonicalRoman(total, &canonical_buf);
    if (!std.mem.eql(u8, canonical, roman)) return RomanError.InvalidFormat;

    return total;
}

test "roman to integer: known values" {
    const cases = [_]struct { roman: []const u8, number: u32 }{
        .{ .roman = "III", .number = 3 },
        .{ .roman = "CLIV", .number = 154 },
        .{ .roman = "MIX", .number = 1009 },
        .{ .roman = "MMD", .number = 2500 },
        .{ .roman = "MMMCMXCIX", .number = 3999 },
    };

    for (cases) |c| {
        try testing.expectEqual(c.number, try romanToInteger(c.roman));
    }
}

test "roman to integer: invalid characters and empty" {
    try testing.expectError(RomanError.EmptyInput, romanToInteger(""));
    try testing.expectError(RomanError.InvalidCharacter, romanToInteger("ABCD"));
    try testing.expectError(RomanError.InvalidCharacter, romanToInteger("xiv"));
}

test "roman to integer: invalid numeral forms" {
    const invalid = [_][]const u8{ "IIII", "IIV", "IL", "IC", "VX", "IM", "MCMC" };
    for (invalid) |s| {
        try testing.expectError(RomanError.InvalidFormat, romanToInteger(s));
    }
}

test "roman to integer: extreme boundaries" {
    try testing.expectEqual(@as(u32, 1), try romanToInteger("I"));
    try testing.expectEqual(@as(u32, 3999), try romanToInteger("MMMCMXCIX"));
}
