//! Factors of a Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/factors.py

const std = @import("std");
const testing = std.testing;

/// Returns sorted positive factors of `num`.
/// For `num < 1`, returns empty slice.
/// Caller owns returned slice.
/// Time complexity: O(sqrt(n)), Space complexity: O(k)
pub fn factorsOfANumber(allocator: std.mem.Allocator, num: i64) std.mem.Allocator.Error![]u64 {
    if (num < 1) return try allocator.alloc(u64, 0);

    const n: u64 = @intCast(num);
    var factors = std.ArrayListUnmanaged(u64){};
    errdefer factors.deinit(allocator);

    try factors.append(allocator, 1);
    if (n == 1) return try factors.toOwnedSlice(allocator);

    try factors.append(allocator, n);

    const root = integerSqrt(n);
    var i: u64 = 2;
    while (i <= root) : (i += 1) {
        if (n % i == 0) {
            try factors.append(allocator, i);
            const d = n / i;
            if (d != i) try factors.append(allocator, d);
        }
    }

    std.mem.sort(u64, factors.items, {}, std.sort.asc(u64));
    return try factors.toOwnedSlice(allocator);
}

fn integerSqrt(value: u64) u64 {
    if (value < 2) return value;
    var x = value;
    var y = (x + value / x) / 2;
    while (y < x) {
        x = y;
        y = (x + value / x) / 2;
    }
    return x;
}

test "factors: python reference examples" {
    const alloc = testing.allocator;

    const f1 = try factorsOfANumber(alloc, 1);
    defer alloc.free(f1);
    try testing.expectEqualSlices(u64, &[_]u64{1}, f1);

    const f5 = try factorsOfANumber(alloc, 5);
    defer alloc.free(f5);
    try testing.expectEqualSlices(u64, &[_]u64{ 1, 5 }, f5);

    const f24 = try factorsOfANumber(alloc, 24);
    defer alloc.free(f24);
    try testing.expectEqualSlices(u64, &[_]u64{ 1, 2, 3, 4, 6, 8, 12, 24 }, f24);
}

test "factors: edge and extreme cases" {
    const neg = try factorsOfANumber(testing.allocator, -24);
    defer testing.allocator.free(neg);
    try testing.expectEqual(@as(usize, 0), neg.len);

    const prime = try factorsOfANumber(testing.allocator, 1_000_003);
    defer testing.allocator.free(prime);
    try testing.expectEqualSlices(u64, &[_]u64{ 1, 1_000_003 }, prime);
}
