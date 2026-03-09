//! Mobius Function - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/mobius_function.py

const std = @import("std");
const testing = std.testing;
const prime_factors_mod = @import("prime_factors.zig");

fn hasRepeatedFactor(factors: []const i64) bool {
    if (factors.len < 2) return false;
    for (factors[1..], factors[0 .. factors.len - 1]) |current, previous| {
        if (current == previous) return true;
    }
    return false;
}

/// Returns the Mobius value with the same semantics as the Python reference.
/// Non-positive integers yield 1 because the Python factorization returns an empty list.
/// Time complexity: O(sqrt(n)), Space complexity: O(log n)
pub fn mobius(allocator: std.mem.Allocator, n: i64) !i8 {
    const factors = try prime_factors_mod.primeFactors(allocator, n);
    defer allocator.free(factors);
    if (hasRepeatedFactor(factors)) return 0;
    return if (factors.len % 2 == 0) 1 else -1;
}

test "mobius function: python reference examples" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(i8, 0), try mobius(alloc, 24));
    try testing.expectEqual(@as(i8, 1), try mobius(alloc, -1));
    try testing.expectEqual(@as(i8, 1), try mobius(alloc, -1424));
}

test "mobius function: edge and extreme cases" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(i8, 1), try mobius(alloc, 1));
    try testing.expectEqual(@as(i8, -1), try mobius(alloc, 13));
    try testing.expectEqual(@as(i8, 0), try mobius(alloc, 1_000_000));
}
