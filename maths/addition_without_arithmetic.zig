//! Addition Without Arithmetic - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/addition_without_arithmetic.py

const std = @import("std");
const testing = std.testing;

pub const AddError = error{Overflow};

/// Adds two integers using bitwise operations.
/// Time complexity: O(1) for fixed-width integers, Space complexity: O(1)
pub fn add(first: i64, second: i64) AddError!i64 {
    const checked_sum_i128 = @as(i128, first) + @as(i128, second);
    if (checked_sum_i128 > std.math.maxInt(i64) or checked_sum_i128 < std.math.minInt(i64)) {
        return AddError.Overflow;
    }

    var a: u64 = @bitCast(first);
    var b: u64 = @bitCast(second);
    while (b != 0) {
        const carry = a & b;
        a ^= b;
        b = carry << 1;
    }
    return @bitCast(a);
}

test "addition without arithmetic: python reference examples" {
    try testing.expectEqual(@as(i64, 8), try add(3, 5));
    try testing.expectEqual(@as(i64, 18), try add(13, 5));
    try testing.expectEqual(@as(i64, -5), try add(-7, 2));
    try testing.expectEqual(@as(i64, -7), try add(0, -7));
    try testing.expectEqual(@as(i64, -321), try add(-321, 0));
}

test "addition without arithmetic: edge and extreme cases" {
    try testing.expectEqual(@as(i64, 0), try add(0, 0));
    try testing.expectEqual(@as(i64, std.math.maxInt(i64) - 1), try add(std.math.maxInt(i64) - 2, 1));
    try testing.expectError(AddError.Overflow, add(std.math.maxInt(i64), 1));
    try testing.expectError(AddError.Overflow, add(std.math.minInt(i64), -1));
}
