//! Square-Free Factor List Check - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/is_square_free.py

const std = @import("std");
const testing = std.testing;

/// Returns true when no value appears more than once in the factor list.
/// Time complexity: O(n), Space complexity: O(n)
pub fn isSquareFree(allocator: std.mem.Allocator, factors: []const i64) !bool {
    var seen = std.AutoHashMap(i64, void).init(allocator);
    defer seen.deinit();

    for (factors) |factor| {
        if (seen.contains(factor)) return false;
        try seen.put(factor, {});
    }
    return true;
}

test "square free: python reference examples" {
    const alloc = testing.allocator;
    try testing.expect(!(try isSquareFree(alloc, &[_]i64{ 1, 1, 2, 3, 4 })));
    try testing.expect(try isSquareFree(alloc, &[_]i64{ 1, 3, 4, 0, -2 }));
    try testing.expect(try isSquareFree(alloc, &[_]i64{ 1, 0, 2, -5 }));
    try testing.expect(!(try isSquareFree(alloc, &[_]i64{ 1, 2, 2, 5 })));
}

test "square free: edge and extreme cases" {
    const alloc = testing.allocator;
    try testing.expect(try isSquareFree(alloc, &[_]i64{}));
    try testing.expect(try isSquareFree(alloc, &[_]i64{42}));
    try testing.expect(!(try isSquareFree(alloc, &[_]i64{ std.math.maxInt(i64), std.math.maxInt(i64) })));
}
