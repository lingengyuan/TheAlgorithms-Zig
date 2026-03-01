//! Sieve of Eratosthenes - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/sieve_of_eratosthenes.py

const std = @import("std");
const testing = std.testing;

pub const SieveError = error{Overflow};

/// Returns all prime numbers up to and including `limit`.
/// Caller owns the returned slice.
/// Time complexity: O(n log log n), Space complexity: O(n)
pub fn primeSieve(allocator: std.mem.Allocator, limit: usize) (SieveError || std.mem.Allocator.Error)![]u64 {
    if (limit < 2) {
        return try allocator.alloc(u64, 0);
    }

    // Create sieve
    const with_one = @addWithOverflow(limit, @as(usize, 1));
    if (with_one[1] != 0) return SieveError.Overflow;
    const sieve = try allocator.alloc(bool, with_one[0]);
    defer allocator.free(sieve);
    @memset(sieve, true);
    sieve[0] = false;
    sieve[1] = false;

    var i: usize = 2;
    while (i <= limit / i) : (i += 1) {
        if (sieve[i]) {
            const start = @mulWithOverflow(i, i);
            if (start[1] != 0) return SieveError.Overflow;
            var j: usize = start[0];
            while (j <= limit) {
                sieve[j] = false;
                const next = @addWithOverflow(j, i);
                if (next[1] != 0) break;
                j = next[0];
            }
        }
    }

    // Count primes
    var count: usize = 0;
    for (sieve) |is_prime| {
        if (is_prime) count += 1;
    }

    // Collect primes
    const primes = try allocator.alloc(u64, count);
    var idx: usize = 0;
    for (sieve, 0..) |is_prime, n| {
        if (is_prime) {
            primes[idx] = @intCast(n);
            idx += 1;
        }
    }
    return primes;
}

test "sieve: primes up to 50" {
    const alloc = testing.allocator;
    const primes = try primeSieve(alloc, 50);
    defer alloc.free(primes);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47 }, primes);
}

test "sieve: primes up to 10" {
    const alloc = testing.allocator;
    const primes = try primeSieve(alloc, 10);
    defer alloc.free(primes);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 5, 7 }, primes);
}

test "sieve: primes up to 2" {
    const alloc = testing.allocator;
    const primes = try primeSieve(alloc, 2);
    defer alloc.free(primes);
    try testing.expectEqualSlices(u64, &[_]u64{2}, primes);
}

test "sieve: limit 1 returns empty" {
    const alloc = testing.allocator;
    const primes = try primeSieve(alloc, 1);
    defer alloc.free(primes);
    try testing.expectEqual(@as(usize, 0), primes.len);
}

test "sieve: limit 0 returns empty" {
    const alloc = testing.allocator;
    const primes = try primeSieve(alloc, 0);
    defer alloc.free(primes);
    try testing.expectEqual(@as(usize, 0), primes.len);
}

test "sieve: oversize limit returns overflow" {
    try testing.expectError(SieveError.Overflow, primeSieve(testing.allocator, std.math.maxInt(usize)));
}
