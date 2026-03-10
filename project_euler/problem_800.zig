//! Project Euler Problem 800: Hybrid Integers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_800/sol1.py

const std = @import("std");
const testing = std.testing;

const Allocator = std.mem.Allocator;

fn calculatePrimeNumbers(allocator: Allocator, max_number: usize) ![]u32 {
    if (max_number <= 2) return allocator.alloc(u32, 0);

    var is_prime = try allocator.alloc(bool, max_number);
    defer allocator.free(is_prime);
    @memset(is_prime, true);
    is_prime[0] = false;
    is_prime[1] = false;

    var i: usize = 2;
    while (i * i < max_number) : (i += 1) {
        if (!is_prime[i]) continue;
        var j = i * i;
        while (j < max_number) : (j += i) is_prime[j] = false;
    }

    var count: usize = 0;
    for (2..max_number) |value| {
        if (is_prime[value]) count += 1;
    }

    var primes = try allocator.alloc(u32, count);
    var index: usize = 0;
    for (2..max_number) |value| {
        if (!is_prime[value]) continue;
        primes[index] = @intCast(value);
        index += 1;
    }
    return primes;
}

/// Returns the count of hybrid integers not exceeding `base^degree`.
/// Time complexity: O(pi(limit))
/// Space complexity: O(limit)
pub fn solution(allocator: Allocator, base: u64, degree: u64) !u64 {
    if (base < 2 or degree == 0) return 0;

    const upper_bound = @as(f64, @floatFromInt(degree)) * std.math.log2(@as(f64, @floatFromInt(base)));
    const primes = try calculatePrimeNumbers(allocator, @as(usize, @intFromFloat(upper_bound)) + 1);
    defer allocator.free(primes);

    if (primes.len < 2) return 0;

    var count: u64 = 0;
    var left: usize = 0;
    var right: usize = primes.len - 1;

    while (left < right) : (left += 1) {
        while (left < right) {
            const lhs = @as(f64, @floatFromInt(primes[right])) * std.math.log2(@as(f64, @floatFromInt(primes[left])));
            const rhs = @as(f64, @floatFromInt(primes[left])) * std.math.log2(@as(f64, @floatFromInt(primes[right])));
            if (lhs + rhs <= upper_bound) break;
            right -= 1;
        }
        if (left >= right) break;
        count += right - left;
    }

    return count;
}

test "problem 800: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 2), try solution(alloc, 800, 1));
    try testing.expectEqual(@as(u64, 10790), try solution(alloc, 800, 800));
    try testing.expectEqual(@as(u64, 1412403576), try solution(alloc, 800800, 800800));
}

test "problem 800: prime helper and edge cases" {
    const alloc = testing.allocator;
    const primes = try calculatePrimeNumbers(alloc, 10);
    defer alloc.free(primes);
    try testing.expectEqualSlices(u32, &[_]u32{ 2, 3, 5, 7 }, primes);
    try testing.expectEqual(@as(u64, 0), try solution(alloc, 1, 10));
}
