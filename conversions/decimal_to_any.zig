//! Decimal to Any Base - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/decimal_to_any.py

const std = @import("std");
const testing = std.testing;

pub const DecimalToAnyError = error{ NegativeNumber, InvalidBase };

/// Converts a non-negative decimal integer into a string representation in `base` (2..36).
///
/// Time complexity: O(log_base(n))
/// Space complexity: O(log_base(n))
pub fn decimalToAny(
    allocator: std.mem.Allocator,
    num: i64,
    base: u8,
) (DecimalToAnyError || std.mem.Allocator.Error)![]u8 {
    if (num < 0) return DecimalToAnyError.NegativeNumber;
    if (base < 2 or base > 36) return DecimalToAnyError.InvalidBase;

    if (num == 0) return allocator.dupe(u8, "0");

    var n: u64 = @intCast(num);
    var reversed = std.ArrayListUnmanaged(u8){};
    defer reversed.deinit(allocator);

    while (n > 0) {
        const digit: u8 = @intCast(n % base);
        const ch: u8 = if (digit < 10) ('0' + digit) else ('A' + (digit - 10));
        try reversed.append(allocator, ch);
        n /= base;
    }

    const out = try allocator.alloc(u8, reversed.items.len);
    for (reversed.items, 0..) |ch, i| {
        out[i] = reversed.items[reversed.items.len - 1 - i];
        _ = ch;
    }
    return out;
}

test "decimal to any: python examples" {
    const alloc = testing.allocator;

    const c0 = try decimalToAny(alloc, 0, 2);
    defer alloc.free(c0);
    try testing.expectEqualStrings("0", c0);

    const c1 = try decimalToAny(alloc, 5, 4);
    defer alloc.free(c1);
    try testing.expectEqualStrings("11", c1);

    const c2 = try decimalToAny(alloc, 20, 3);
    defer alloc.free(c2);
    try testing.expectEqualStrings("202", c2);

    const c3 = try decimalToAny(alloc, 58, 16);
    defer alloc.free(c3);
    try testing.expectEqualStrings("3A", c3);

    const c4 = try decimalToAny(alloc, 243, 17);
    defer alloc.free(c4);
    try testing.expectEqualStrings("E5", c4);

    const c5 = try decimalToAny(alloc, 34_923, 36);
    defer alloc.free(c5);
    try testing.expectEqualStrings("QY3", c5);

    const c6 = try decimalToAny(alloc, 36, 36);
    defer alloc.free(c6);
    try testing.expectEqualStrings("10", c6);
}

test "decimal to any: validation and extreme" {
    const alloc = testing.allocator;

    try testing.expectError(DecimalToAnyError.NegativeNumber, decimalToAny(alloc, -45, 8));
    try testing.expectError(DecimalToAnyError.InvalidBase, decimalToAny(alloc, 7, 1));
    try testing.expectError(DecimalToAnyError.InvalidBase, decimalToAny(alloc, 34, 37));

    const max_bin = try decimalToAny(alloc, std.math.maxInt(i64), 2);
    defer alloc.free(max_bin);
    try testing.expectEqual(@as(usize, 63), max_bin.len);
    try testing.expectEqual(@as(u8, '1'), max_bin[0]);
    for (max_bin) |ch| {
        try testing.expect(ch == '0' or ch == '1');
    }
}
