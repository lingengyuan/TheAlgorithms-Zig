//! Project Euler Problem 77: Prime Summations - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_077/sol1.py

const std = @import("std");
const testing = std.testing;

fn sievePrimes(allocator: std.mem.Allocator, limit: usize) ![]usize {
    var is_prime = try allocator.alloc(bool, limit + 1);
    defer allocator.free(is_prime);
    @memset(is_prime, true);
    if (limit >= 0) is_prime[0] = false;
    if (limit >= 1) is_prime[1] = false;

    var p: usize = 2;
    while (p <= limit / p) : (p += 1) {
        if (!is_prime[p]) continue;
        var multiple = p * p;
        while (multiple <= limit) : (multiple += p) is_prime[multiple] = false;
    }

    var primes = std.ArrayListUnmanaged(usize){};
    errdefer primes.deinit(allocator);
    for (is_prime, 0..) |prime, value| {
        if (prime) try primes.append(allocator, value);
    }
    return primes.toOwnedSlice(allocator);
}

fn countPrimePartitions(allocator: std.mem.Allocator, target: usize) !u64 {
    const primes = try sievePrimes(allocator, target);
    defer allocator.free(primes);

    var ways = try allocator.alloc(u64, target + 1);
    defer allocator.free(ways);
    @memset(ways, 0);
    ways[0] = 1;

    for (primes) |prime| {
        var total = prime;
        while (total <= target) : (total += 1) {
            ways[total] += ways[total - prime];
        }
    }
    return ways[target];
}

/// Returns the smallest integer that can be written as a sum of primes in over
/// `number_unique_partitions` distinct ways.
/// Time complexity: roughly O(answer^2 / log answer)
/// Space complexity: O(answer)
pub fn solution(allocator: std.mem.Allocator, number_unique_partitions: u64) !?usize {
    var number_to_partition: usize = 1;
    while (number_to_partition < 256) : (number_to_partition += 1) {
        if (try countPrimePartitions(allocator, number_to_partition) > number_unique_partitions) {
            return number_to_partition;
        }
    }
    return null;
}

test "problem 077: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(?usize, 10), try solution(alloc, 4));
    try testing.expectEqual(@as(?usize, 45), try solution(alloc, 500));
    try testing.expectEqual(@as(?usize, 53), try solution(alloc, 1000));
    try testing.expectEqual(@as(?usize, 71), try solution(alloc, 5000));
}

test "problem 077: partition helper and tiny thresholds" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 5), try countPrimePartitions(alloc, 10));
    try testing.expectEqual(@as(u64, 26), try countPrimePartitions(alloc, 20));
    try testing.expectEqual(@as(?usize, 2), try solution(alloc, 0));
}
