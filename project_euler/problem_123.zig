//! Project Euler Problem 123: Prime Square Remainders - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_123/sol1.py

const std = @import("std");
const testing = std.testing;

fn isPrime(primes: []const u64, candidate: u64) bool {
    if (candidate < 2) return false;
    for (primes) |prime| {
        if (prime * prime > candidate) break;
        if (candidate % prime == 0) return false;
    }
    return true;
}

/// Returns the least prime index `n` for which the remainder first exceeds `limit`.
/// Time complexity: roughly O(answer · pi(sqrt(p_n)))
/// Space complexity: O(pi(p_n))
pub fn solution(allocator: std.mem.Allocator, limit: u64) !u32 {
    var primes = std.ArrayListUnmanaged(u64){};
    defer primes.deinit(allocator);

    var candidate: u64 = 2;
    var n: u32 = 0;
    while (true) : (candidate += if (candidate == 2) 1 else 2) {
        if (!isPrime(primes.items, candidate)) continue;
        try primes.append(allocator, candidate);
        n += 1;
        if ((n & 1) == 1 and 2 * candidate * @as(u64, n) > limit) return n;
    }
}

test "problem 123: python reference" {
    try testing.expectEqual(@as(u32, 2371), try solution(testing.allocator, 100_000_000));
    try testing.expectEqual(@as(u32, 7037), try solution(testing.allocator, 1_000_000_000));
    try testing.expectEqual(@as(u32, 21035), try solution(testing.allocator, 10_000_000_000));
}

test "problem 123: small limit" {
    try testing.expectEqual(@as(u32, 1), try solution(testing.allocator, 3));
    try testing.expectEqual(@as(u32, 3), try solution(testing.allocator, 5));
}
