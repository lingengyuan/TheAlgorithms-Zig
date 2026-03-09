//! Prime Factors - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/prime_factors.py

const std = @import("std");
const testing = std.testing;

fn appendFactor(list: *std.ArrayListUnmanaged(i64), allocator: std.mem.Allocator, factor: i64) !void {
    try list.append(allocator, factor);
}

/// Returns prime factors of `n` in ascending order.
/// Caller owns the returned slice.
/// Time complexity: O(sqrt(n)), Space complexity: O(log n)
pub fn primeFactors(allocator: std.mem.Allocator, n: i64) ![]i64 {
    if (n <= 1) return allocator.alloc(i64, 0);

    var factors = std.ArrayListUnmanaged(i64){};
    defer factors.deinit(allocator);

    var value = n;
    var divisor: i64 = 2;
    while (divisor * divisor <= value) {
        if (@rem(value, divisor) != 0) {
            divisor += 1;
        } else {
            value = @divTrunc(value, divisor);
            try appendFactor(&factors, allocator, divisor);
        }
    }
    if (value > 1) {
        try appendFactor(&factors, allocator, value);
    }
    return factors.toOwnedSlice(allocator);
}

/// Returns unique prime factors of `n` in ascending order.
/// Caller owns the returned slice.
/// Time complexity: O(sqrt(n)), Space complexity: O(log n)
pub fn uniquePrimeFactors(allocator: std.mem.Allocator, n: i64) ![]i64 {
    if (n <= 1) return allocator.alloc(i64, 0);

    var factors = std.ArrayListUnmanaged(i64){};
    defer factors.deinit(allocator);

    var value = n;
    var divisor: i64 = 2;
    while (divisor * divisor <= value) {
        if (@rem(value, divisor) == 0) {
            while (@rem(value, divisor) == 0) {
                value = @divTrunc(value, divisor);
            }
            try appendFactor(&factors, allocator, divisor);
        }
        divisor += 1;
    }
    if (value > 1) {
        try appendFactor(&factors, allocator, value);
    }
    return factors.toOwnedSlice(allocator);
}

test "prime factors: python reference examples" {
    const alloc = testing.allocator;

    const f1 = try primeFactors(alloc, 0);
    defer alloc.free(f1);
    try testing.expectEqual(@as(usize, 0), f1.len);

    const f2 = try primeFactors(alloc, 100);
    defer alloc.free(f2);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 2, 5, 5 }, f2);

    const f3 = try uniquePrimeFactors(alloc, 2560);
    defer alloc.free(f3);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 5 }, f3);
}

test "prime factors: edge and extreme cases" {
    const alloc = testing.allocator;
    const f1 = try primeFactors(alloc, 1);
    defer alloc.free(f1);
    try testing.expectEqual(@as(usize, 0), f1.len);

    const f2 = try primeFactors(alloc, 1_000_000);
    defer alloc.free(f2);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5 }, f2);
}
