//! Base -2 Conversion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/base_neg2_conversion.py

const std = @import("std");
const testing = std.testing;

/// Converts an integer to its negative-base-2 representation.
/// Caller owns the returned string.
/// Time complexity: O(log |n|), Space complexity: O(log |n|)
pub fn decimalToNegativeBase2(allocator: std.mem.Allocator, num: i64) ![]u8 {
    if (num == 0) return allocator.dupe(u8, "0");

    var n = num;
    var digits = std.ArrayListUnmanaged(u8){};
    defer digits.deinit(allocator);

    while (n != 0) {
        var q = @divFloor(n, -2);
        var rem = n - q * -2;
        if (rem < 0) {
            rem += 2;
            q += 1;
        }
        try digits.append(allocator, @as(u8, @intCast(rem)) + '0');
        n = q;
    }

    std.mem.reverse(u8, digits.items);
    return digits.toOwnedSlice(allocator);
}

test "base neg2 conversion: python reference examples" {
    const alloc = testing.allocator;
    const c1 = try decimalToNegativeBase2(alloc, 0);
    defer alloc.free(c1);
    try testing.expectEqualStrings("0", c1);

    const c2 = try decimalToNegativeBase2(alloc, -19);
    defer alloc.free(c2);
    try testing.expectEqualStrings("111101", c2);

    const c3 = try decimalToNegativeBase2(alloc, 4);
    defer alloc.free(c3);
    try testing.expectEqualStrings("100", c3);

    const c4 = try decimalToNegativeBase2(alloc, 7);
    defer alloc.free(c4);
    try testing.expectEqualStrings("11011", c4);
}

test "base neg2 conversion: edge and extreme cases" {
    const alloc = testing.allocator;
    const c1 = try decimalToNegativeBase2(alloc, -1);
    defer alloc.free(c1);
    try testing.expectEqualStrings("11", c1);

    const c2 = try decimalToNegativeBase2(alloc, 1);
    defer alloc.free(c2);
    try testing.expectEqualStrings("1", c2);

    const c3 = try decimalToNegativeBase2(alloc, -2);
    defer alloc.free(c3);
    try testing.expectEqualStrings("10", c3);
}
