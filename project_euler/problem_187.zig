//! Project Euler Problem 187: Semiprimes - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_187/sol1.py

const std = @import("std");
const testing = std.testing;

const Allocator = std.mem.Allocator;

fn calculatePrimeNumbers(allocator: Allocator, max_number: usize) ![]usize {
    if (max_number <= 2) return allocator.alloc(usize, 0);

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

    var primes = try allocator.alloc(usize, count);
    var index: usize = 0;
    for (2..max_number) |value| {
        if (!is_prime[value]) continue;
        primes[index] = value;
        index += 1;
    }
    return primes;
}

/// Counts composite integers below `max_number` with exactly two prime factors,
/// not necessarily distinct.
/// Time complexity: O(pi(max_number / 2))
/// Space complexity: O(max_number)
pub fn solution(allocator: Allocator, max_number: usize) !usize {
    if (max_number <= 4) return 0;

    const primes = try calculatePrimeNumbers(allocator, max_number / 2 + 1);
    defer allocator.free(primes);

    if (primes.len == 0) return 0;

    var semiprimes_count: usize = 0;
    var right: usize = primes.len - 1;
    for (primes, 0..) |left_prime, left| {
        if (left > right) break;
        while (left <= right and @as(u128, left_prime) * primes[right] >= max_number) {
            if (right == 0) break;
            right -= 1;
        }
        if (left > right or @as(u128, left_prime) * primes[right] >= max_number) break;
        semiprimes_count += right - left + 1;
    }

    return semiprimes_count;
}

test "problem 187: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 10), try solution(alloc, 30));
    try testing.expectEqual(@as(usize, 34), try solution(alloc, 100));
    try testing.expectEqual(@as(usize, 2625), try solution(alloc, 10000));
    try testing.expectEqual(@as(usize, 17427258), try solution(alloc, 100000000));
}

test "problem 187: prime generation and edge bounds" {
    const alloc = testing.allocator;

    const primes_under_ten = try calculatePrimeNumbers(alloc, 10);
    defer alloc.free(primes_under_ten);
    try testing.expectEqualSlices(usize, &[_]usize{ 2, 3, 5, 7 }, primes_under_ten);

    const primes_under_two = try calculatePrimeNumbers(alloc, 2);
    defer alloc.free(primes_under_two);
    try testing.expectEqual(@as(usize, 0), primes_under_two.len);

    try testing.expectEqual(@as(usize, 0), try solution(alloc, 4));
    try testing.expectEqual(@as(usize, 1), try solution(alloc, 5));
}
