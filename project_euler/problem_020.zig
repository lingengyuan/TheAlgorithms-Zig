//! Project Euler Problem 20: Factorial Digit Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_020/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the sum of decimal digits of n!.
///
/// Time complexity: O(n * digits(n!))
/// Space complexity: O(digits(n!))
pub fn solution(n: u32, allocator: std.mem.Allocator) !u64 {
    var digits = std.ArrayListUnmanaged(u8){};
    defer digits.deinit(allocator);

    // Decimal digits for current factorial value, little-endian.
    try digits.append(allocator, 1);

    if (n <= 1) return 1;

    var factor: u32 = 2;
    while (factor <= n) : (factor += 1) {
        var carry: u64 = 0;

        for (digits.items) |*digit| {
            const product = @as(u64, digit.*) * factor + carry;
            digit.* = @intCast(product % 10);
            carry = product / 10;
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

test "problem 020: python reference" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 648), try solution(100, allocator));
    try testing.expectEqual(@as(u64, 216), try solution(50, allocator));
    try testing.expectEqual(@as(u64, 27), try solution(10, allocator));
    try testing.expectEqual(@as(u64, 3), try solution(5, allocator));
    try testing.expectEqual(@as(u64, 6), try solution(3, allocator));
    try testing.expectEqual(@as(u64, 2), try solution(2, allocator));
    try testing.expectEqual(@as(u64, 1), try solution(1, allocator));
}

test "problem 020: boundaries and larger input" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 1), try solution(0, allocator));
    try testing.expectEqual(@as(u64, 1), try solution(1, allocator));
    try testing.expectEqual(@as(u64, 1404), try solution(200, allocator));

    // Extreme-case stress to verify big-decimal multiplication path.
    try testing.expectEqual(@as(u64, 10_539), try solution(1000, allocator));
}
