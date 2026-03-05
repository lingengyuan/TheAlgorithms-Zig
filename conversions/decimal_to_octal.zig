//! Decimal to Octal - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/decimal_to_octal.py

const std = @import("std");
const testing = std.testing;

/// Converts decimal integer to octal string with `0o` prefix.
///
/// API note: Python reference returns `"0o0"` for non-positive inputs due
/// loop condition (`while num > 0`). This implementation keeps that behavior.
///
/// Time complexity: O(log_8 n)
/// Space complexity: O(log_8 n)
pub fn decimalToOctal(
    allocator: std.mem.Allocator,
    num: i64,
) std.mem.Allocator.Error![]u8 {
    if (num <= 0) {
        const out = try allocator.alloc(u8, 3);
        out[0] = '0';
        out[1] = 'o';
        out[2] = '0';
        return out;
    }

    var value: u64 = @intCast(num);
    var rev_digits = std.ArrayListUnmanaged(u8){};
    defer rev_digits.deinit(allocator);

    while (value > 0) {
        const digit: u8 = @intCast(value % 8);
        try rev_digits.append(allocator, '0' + digit);
        value /= 8;
    }

    const out = try allocator.alloc(u8, 2 + rev_digits.items.len);
    out[0] = '0';
    out[1] = 'o';
    for (0..rev_digits.items.len) |i| {
        out[2 + i] = rev_digits.items[rev_digits.items.len - 1 - i];
    }
    return out;
}

test "decimal to octal: python style examples" {
    const alloc = testing.allocator;

    const cases = [_]struct { n: i64, expected: []const u8 }{
        .{ .n = 0, .expected = "0o0" },
        .{ .n = 2, .expected = "0o2" },
        .{ .n = 8, .expected = "0o10" },
        .{ .n = 64, .expected = "0o100" },
        .{ .n = 65, .expected = "0o101" },
        .{ .n = 216, .expected = "0o330" },
        .{ .n = 255, .expected = "0o377" },
        .{ .n = 256, .expected = "0o400" },
        .{ .n = 512, .expected = "0o1000" },
    };

    for (cases) |c| {
        const s = try decimalToOctal(alloc, c.n);
        defer alloc.free(s);
        try testing.expectEqualStrings(c.expected, s);
    }
}

test "decimal to octal: edge and extreme" {
    const alloc = testing.allocator;

    const neg = try decimalToOctal(alloc, -10);
    defer alloc.free(neg);
    try testing.expectEqualStrings("0o0", neg);

    const big = try decimalToOctal(alloc, std.math.maxInt(i64));
    defer alloc.free(big);
    try testing.expect(big.len > 3);
    try testing.expect(std.mem.startsWith(u8, big, "0o"));
}
