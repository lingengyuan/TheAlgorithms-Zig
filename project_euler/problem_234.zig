//! Project Euler Problem 234: Semidivisible Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_234/sol1.py

const std = @import("std");
const testing = std.testing;

const Allocator = std.mem.Allocator;

fn primeSieve(allocator: Allocator, limit: usize) ![]u64 {
    if (limit <= 2) return allocator.alloc(u64, 0);

    var is_prime = try allocator.alloc(bool, limit);
    defer allocator.free(is_prime);
    @memset(is_prime, true);
    is_prime[0] = false;
    is_prime[1] = false;

    var i: usize = 2;
    while (i * i < limit) : (i += 1) {
        if (!is_prime[i]) continue;
        var j = i * i;
        while (j < limit) : (j += i) is_prime[j] = false;
    }

    var count: usize = 0;
    for (2..limit) |value| {
        if (is_prime[value]) count += 1;
    }

    var primes = try allocator.alloc(u64, count);
    var index: usize = 0;
    for (2..limit) |value| {
        if (!is_prime[value]) continue;
        primes[index] = value;
        index += 1;
    }
    return primes;
}

fn sumMultiplesInRange(divisor: u64, lower_exclusive: u64, upper_exclusive: u64) u128 {
    if (upper_exclusive <= lower_exclusive + 1) return 0;

    const start = lower_exclusive / divisor + 1;
    const end = (upper_exclusive - 1) / divisor;
    if (start > end) return 0;

    const count: u128 = end - start + 1;
    return @as(u128, divisor) * count * @as(u128, start + end) / 2;
}

/// Returns the sum of all semidivisible numbers not exceeding `limit`.
/// Time complexity: roughly O(pi(sqrt(limit)))
/// Space complexity: O(sqrt(limit))
pub fn solution(allocator: Allocator, limit: u64) !u64 {
    if (limit < 4) return 0;

    const upper_bound = std.math.sqrt(limit) + 100;
    const primes = try primeSieve(allocator, @intCast(upper_bound));
    defer allocator.free(primes);

    var total: u128 = 0;
    var index: usize = 0;
    while (index + 1 < primes.len) : (index += 1) {
        const p = primes[index];
        const q = primes[index + 1];
        const lower = p * p;
        if (lower > limit) break;

        const upper_exclusive = @min(limit + 1, q * q);
        total += sumMultiplesInRange(p, lower, upper_exclusive);
        total += sumMultiplesInRange(q, lower, upper_exclusive);
        total -= 2 * sumMultiplesInRange(p * q, lower, upper_exclusive);
    }

    return @intCast(total);
}

test "problem 234: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 34825), try solution(alloc, 1000));
    try testing.expectEqual(@as(u64, 1134942), try solution(alloc, 10000));
    try testing.expectEqual(@as(u64, 36393008), try solution(alloc, 100000));
    try testing.expectEqual(@as(u64, 1259187438574927161), try solution(alloc, 999966663333));
}

test "problem 234: sieve and arithmetic-range helpers" {
    const alloc = testing.allocator;
    const primes = try primeSieve(alloc, 50);
    defer alloc.free(primes);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47 }, primes);
    try testing.expectEqual(@as(u128, 50), sumMultiplesInRange(2, 4, 15));
}
