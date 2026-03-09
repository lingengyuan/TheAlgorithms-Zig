//! Prime Numbers Generators - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/prime_numbers.py

const std = @import("std");
const testing = std.testing;

fn isPrimeSlow(n: i64) bool {
    if (n <= 1) return false;
    var d: i64 = 2;
    while (d < n) : (d += 1) {
        if (@rem(n, d) == 0) return false;
    }
    return true;
}

fn isPrimeBounded(n: i64) bool {
    if (n <= 1) return false;
    var d: i64 = 2;
    const bound = std.math.sqrt(@as(u64, @intCast(n))) + 1;
    while (d < @as(i64, @intCast(bound))) : (d += 1) {
        if (@rem(n, d) == 0) return false;
    }
    return true;
}

fn appendPrimes(
    allocator: std.mem.Allocator,
    max_n: i64,
    comptime checker: fn (i64) bool,
    odd_only: bool,
) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    defer out.deinit(allocator);

    if (odd_only and max_n > 2) {
        try out.append(allocator, 2);
    }

    var start: i64 = if (odd_only) 1 else 1;
    const step: i64 = if (odd_only) 2 else 1;
    while (start <= max_n) : (start += step) {
        if (start > 1 and checker(start)) {
            if (!(odd_only and start == 2)) {
                try out.append(allocator, start);
            }
        }
    }
    return out.toOwnedSlice(allocator);
}

/// Returns primes up to `max_n` with the slow reference predicate.
pub fn slowPrimes(allocator: std.mem.Allocator, max_n: i64) ![]i64 {
    return appendPrimes(allocator, max_n, isPrimeSlow, false);
}

/// Returns primes up to `max_n` using sqrt-bounded trial division.
pub fn primes(allocator: std.mem.Allocator, max_n: i64) ![]i64 {
    return appendPrimes(allocator, max_n, isPrimeBounded, false);
}

/// Returns primes up to `max_n`, skipping even candidates.
pub fn fastPrimes(allocator: std.mem.Allocator, max_n: i64) ![]i64 {
    return appendPrimes(allocator, max_n, isPrimeBounded, true);
}

test "prime numbers: python reference examples" {
    const alloc = testing.allocator;
    const expected = [_]i64{ 2, 3, 5, 7, 11, 13, 17, 19, 23 };

    const s = try slowPrimes(alloc, 25);
    defer alloc.free(s);
    try testing.expectEqualSlices(i64, &expected, s);

    const p = try primes(alloc, 25);
    defer alloc.free(p);
    try testing.expectEqualSlices(i64, &expected, p);

    const f = try fastPrimes(alloc, 25);
    defer alloc.free(f);
    try testing.expectEqualSlices(i64, &expected, f);
}

test "prime numbers: edge and extreme cases" {
    const alloc = testing.allocator;

    const e1 = try primes(alloc, 0);
    defer alloc.free(e1);
    try testing.expectEqual(@as(usize, 0), e1.len);

    const e2 = try fastPrimes(alloc, 1000);
    defer alloc.free(e2);
    try testing.expectEqual(@as(i64, 997), e2[e2.len - 1]);
}
