//! Prime Sieve Eratosthenes - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/prime_sieve_eratosthenes.py

const std = @import("std");
const testing = std.testing;
const sieve = @import("sieve_of_eratosthenes.zig");

pub const PrimeSieveError = error{InvalidInput};

/// Returns all primes up to and including `num`.
/// Caller owns the returned slice.
/// Time complexity: O(n log log n), Space complexity: O(n)
pub fn primeSieveEratosthenes(allocator: std.mem.Allocator, num: i64) ![]u64 {
    if (num <= 0) return error.InvalidInput;
    if (num == 1) return allocator.alloc(u64, 0);
    return sieve.primeSieve(allocator, @intCast(num));
}

test "prime sieve eratosthenes: python reference examples" {
    const alloc = testing.allocator;
    const p10 = try primeSieveEratosthenes(alloc, 10);
    defer alloc.free(p10);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 5, 7 }, p10);

    const p20 = try primeSieveEratosthenes(alloc, 20);
    defer alloc.free(p20);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 5, 7, 11, 13, 17, 19 }, p20);
}

test "prime sieve eratosthenes: edge cases" {
    const alloc = testing.allocator;
    const p2 = try primeSieveEratosthenes(alloc, 2);
    defer alloc.free(p2);
    try testing.expectEqualSlices(u64, &[_]u64{2}, p2);

    const p1 = try primeSieveEratosthenes(alloc, 1);
    defer alloc.free(p1);
    try testing.expectEqual(@as(usize, 0), p1.len);

    try testing.expectError(error.InvalidInput, primeSieveEratosthenes(alloc, -1));
}
