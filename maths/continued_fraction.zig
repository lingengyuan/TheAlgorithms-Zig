//! Continued Fraction - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/continued_fraction.py

const std = @import("std");
const testing = std.testing;

pub const ContinuedFractionError = error{ZeroDenominator};

/// Returns the continued fraction of the rational number `numerator / denominator`.
/// Caller owns the returned slice.
/// Time complexity: O(k), Space complexity: O(k)
pub fn continuedFraction(
    allocator: std.mem.Allocator,
    numerator: i128,
    denominator: i128,
) (ContinuedFractionError || std.mem.Allocator.Error)![]i64 {
    if (denominator == 0) return error.ZeroDenominator;

    var num = numerator;
    var den = denominator;
    var out = std.ArrayListUnmanaged(i64){};
    defer out.deinit(allocator);

    while (true) {
        const integer_part = @divFloor(num, den);
        try out.append(allocator, @intCast(integer_part));
        num -= integer_part * den;
        if (num == 0) break;
        const tmp = num;
        num = den;
        den = tmp;
    }
    return out.toOwnedSlice(allocator);
}

test "continued fraction: python reference examples" {
    const alloc = testing.allocator;
    const c1 = try continuedFraction(alloc, 2, 1);
    defer alloc.free(c1);
    try testing.expectEqualSlices(i64, &[_]i64{2}, c1);

    const c2 = try continuedFraction(alloc, 649, 200);
    defer alloc.free(c2);
    try testing.expectEqualSlices(i64, &[_]i64{ 3, 4, 12, 4 }, c2);

    const c3 = try continuedFraction(alloc, 415, 93);
    defer alloc.free(c3);
    try testing.expectEqualSlices(i64, &[_]i64{ 4, 2, 6, 7 }, c3);
}

test "continued fraction: edge and extreme cases" {
    const alloc = testing.allocator;
    const c1 = try continuedFraction(alloc, 0, 1);
    defer alloc.free(c1);
    try testing.expectEqualSlices(i64, &[_]i64{0}, c1);

    const c2 = try continuedFraction(alloc, -9, 4);
    defer alloc.free(c2);
    try testing.expectEqualSlices(i64, &[_]i64{ -3, 1, 3 }, c2);

    try testing.expectError(error.ZeroDenominator, continuedFraction(alloc, 1, 0));
}
