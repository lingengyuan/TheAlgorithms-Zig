//! Project Euler Problem 16: Power Digit Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_016/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns sum of decimal digits of 2^power.
///
/// Time complexity: O(power * digits)
/// Space complexity: O(digits)
pub fn solution(power: u32, allocator: std.mem.Allocator) !u64 {
    var digits = std.ArrayListUnmanaged(u8){};
    defer digits.deinit(allocator);

    try digits.append(allocator, 1); // little-endian decimal digits for current value

    var p: u32 = 0;
    while (p < power) : (p += 1) {
        var carry: u16 = 0;
        for (digits.items) |*digit| {
            const value: u16 = @as(u16, digit.*) * 2 + carry;
            digit.* = @intCast(value % 10);
            carry = value / 10;
        }

        while (carry > 0) {
            try digits.append(allocator, @intCast(carry % 10));
            carry /= 10;
        }
    }

    var sum: u64 = 0;
    for (digits.items) |digit| {
        sum += digit;
    }
    return sum;
}

test "problem 016: python reference" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 1366), try solution(1000, allocator));
    try testing.expectEqual(@as(u64, 76), try solution(50, allocator));
    try testing.expectEqual(@as(u64, 31), try solution(20, allocator));
    try testing.expectEqual(@as(u64, 26), try solution(15, allocator));
}

test "problem 016: boundaries and large powers" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 1), try solution(0, allocator));
    try testing.expectEqual(@as(u64, 2), try solution(1, allocator));
    try testing.expectEqual(@as(u64, 4), try solution(2, allocator));
    try testing.expectEqual(@as(u64, 115), try solution(100, allocator));
    try testing.expectEqual(@as(u64, 6790), try solution(5000, allocator));
    try testing.expectEqual(@as(u64, 13_561), try solution(10_000, allocator));
}
