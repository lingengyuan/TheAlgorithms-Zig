//! Decimal to Hexadecimal - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/decimal_to_hexadecimal.py

const std = @import("std");
const testing = std.testing;

const HEX_DIGITS = "0123456789abcdef";

/// Converts u64 to lowercase hex string without "0x" prefix.
/// Caller owns the returned slice.
pub fn decimalToHex(allocator: std.mem.Allocator, n: u64) ![]u8 {
    if (n == 0) {
        const s = try allocator.alloc(u8, 1);
        s[0] = '0';
        return s;
    }
    var tmp = n;
    var len: usize = 0;
    while (tmp > 0) : (tmp >>= 4) len += 1;

    const s = try allocator.alloc(u8, len);
    tmp = n;
    var i: usize = len;
    while (tmp > 0) {
        i -= 1;
        s[i] = HEX_DIGITS[@intCast(tmp & 0xF)];
        tmp >>= 4;
    }
    return s;
}

test "decimal to hex: known values" {
    const alloc = testing.allocator;
    const cases = [_]struct { n: u64, expected: []const u8 }{
        .{ .n = 0, .expected = "0" },
        .{ .n = 5, .expected = "5" },
        .{ .n = 15, .expected = "f" },
        .{ .n = 37, .expected = "25" },
        .{ .n = 255, .expected = "ff" },
        .{ .n = 256, .expected = "100" },
        .{ .n = 4096, .expected = "1000" },
    };
    for (cases) |c| {
        const s = try decimalToHex(alloc, c.n);
        defer alloc.free(s);
        try testing.expectEqualStrings(c.expected, s);
    }
}
