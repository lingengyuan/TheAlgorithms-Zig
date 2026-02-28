//! Decimal to Binary - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/decimal_to_binary.py

const std = @import("std");
const testing = std.testing;

/// Converts a u64 to its binary string representation (no prefix).
/// Caller owns the returned slice.
pub fn decimalToBinary(allocator: std.mem.Allocator, n: u64) ![]u8 {
    if (n == 0) {
        const s = try allocator.alloc(u8, 1);
        s[0] = '0';
        return s;
    }
    // Count digits
    var tmp = n;
    var len: usize = 0;
    while (tmp > 0) : (tmp >>= 1) len += 1;

    const s = try allocator.alloc(u8, len);
    tmp = n;
    var i: usize = len;
    while (tmp > 0) {
        i -= 1;
        s[i] = if (tmp & 1 == 1) '1' else '0';
        tmp >>= 1;
    }
    return s;
}

test "decimal to binary: zero" {
    const alloc = testing.allocator;
    const s = try decimalToBinary(alloc, 0);
    defer alloc.free(s);
    try testing.expectEqualStrings("0", s);
}

test "decimal to binary: known values" {
    const alloc = testing.allocator;
    const cases = [_]struct { n: u64, expected: []const u8 }{
        .{ .n = 1, .expected = "1" },
        .{ .n = 2, .expected = "10" },
        .{ .n = 7, .expected = "111" },
        .{ .n = 35, .expected = "100011" },
        .{ .n = 255, .expected = "11111111" },
    };
    for (cases) |c| {
        const s = try decimalToBinary(alloc, c.n);
        defer alloc.free(s);
        try testing.expectEqualStrings(c.expected, s);
    }
}
