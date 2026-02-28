//! Euler's Totient Function - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/eulers_totient.py

const std = @import("std");
const testing = std.testing;

/// Returns φ(n): number of integers in [1, n] that are coprime with n.
/// Time complexity: O(sqrt(n)), Space complexity: O(1)
pub fn eulersTotient(n: u64) u64 {
    if (n == 0) return 0;
    var result = n;
    var x = n;
    var p: u64 = 2;
    while (p * p <= x) : (p += 1) {
        if (x % p == 0) {
            while (x % p == 0) x /= p;
            result -= result / p;
        }
    }
    if (x > 1) {
        result -= result / x;
    }
    return result;
}

test "totient: basic values" {
    try testing.expectEqual(@as(u64, 1), eulersTotient(1));
    try testing.expectEqual(@as(u64, 1), eulersTotient(2));
    try testing.expectEqual(@as(u64, 2), eulersTotient(3));
    try testing.expectEqual(@as(u64, 2), eulersTotient(4));
    try testing.expectEqual(@as(u64, 4), eulersTotient(5));
}

test "totient: composite numbers" {
    try testing.expectEqual(@as(u64, 4), eulersTotient(8));
    try testing.expectEqual(@as(u64, 6), eulersTotient(9));
    try testing.expectEqual(@as(u64, 4), eulersTotient(10));
    try testing.expectEqual(@as(u64, 12), eulersTotient(21));
}

test "totient: prime number" {
    try testing.expectEqual(@as(u64, 96), eulersTotient(97));
}

test "totient: zero" {
    try testing.expectEqual(@as(u64, 0), eulersTotient(0));
}
