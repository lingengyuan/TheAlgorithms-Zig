//! Twin Prime - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/twin_prime.py

const std = @import("std");
const testing = std.testing;
const prime_check = @import("prime_check.zig");

/// Returns `number + 2` if both `number` and `number + 2` are prime; otherwise returns -1.
/// Time complexity: O(sqrt(n)), Space complexity: O(1)
pub fn twinPrime(number: i64) i64 {
    if (number < 2) return -1;
    if (number > std.math.maxInt(i64) - 2) return -1;

    const n: u64 = @intCast(number);
    const n_plus_two: u64 = @intCast(number + 2);
    if (prime_check.isPrime(n) and prime_check.isPrime(n_plus_two)) {
        return number + 2;
    }
    return -1;
}

test "twin prime: python reference examples" {
    try testing.expectEqual(@as(i64, 5), twinPrime(3));
    try testing.expectEqual(@as(i64, -1), twinPrime(4));
    try testing.expectEqual(@as(i64, 7), twinPrime(5));
    try testing.expectEqual(@as(i64, 19), twinPrime(17));
    try testing.expectEqual(@as(i64, -1), twinPrime(0));
}

test "twin prime: edge and extreme cases" {
    try testing.expectEqual(@as(i64, -1), twinPrime(-11));
    try testing.expectEqual(@as(i64, -1), twinPrime(2));
    try testing.expectEqual(@as(i64, -1), twinPrime(std.math.maxInt(i64)));
    try testing.expectEqual(@as(i64, -1), twinPrime(std.math.maxInt(i64) - 1));
}
