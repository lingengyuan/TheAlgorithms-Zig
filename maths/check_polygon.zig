//! Check Polygon - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/check_polygon.py

const std = @import("std");
const testing = std.testing;

pub const PolygonCheckError = error{ InvalidPolygon, NonPositiveSide };

/// Returns true when side lengths can form a Euclidean polygon.
/// Time complexity: O(n log n), Space complexity: O(n)
pub fn checkPolygon(
    allocator: std.mem.Allocator,
    nums: []const f64,
) (PolygonCheckError || std.mem.Allocator.Error)!bool {
    if (nums.len < 2) return PolygonCheckError.InvalidPolygon;
    for (nums) |value| {
        if (value <= 0.0) return PolygonCheckError.NonPositiveSide;
    }

    const copy_nums = try allocator.alloc(f64, nums.len);
    defer allocator.free(copy_nums);
    @memcpy(copy_nums, nums);
    std.mem.sort(f64, copy_nums, {}, std.sort.asc(f64));

    var rest_sum: f64 = 0.0;
    for (copy_nums[0 .. copy_nums.len - 1]) |value| rest_sum += value;
    return copy_nums[copy_nums.len - 1] < rest_sum;
}

test "check polygon: python reference examples" {
    try testing.expect(try checkPolygon(testing.allocator, &[_]f64{ 6, 10, 5 }));
    try testing.expect(!(try checkPolygon(testing.allocator, &[_]f64{ 3, 7, 13, 2 })));
    try testing.expect(!(try checkPolygon(testing.allocator, &[_]f64{ 1, 4.3, 5.2, 12.2 })));
}

test "check polygon: input immutability and validation" {
    var nums = [_]f64{ 3, 7, 13, 2 };
    _ = try checkPolygon(testing.allocator, &nums);
    try testing.expectEqualSlices(f64, &[_]f64{ 3, 7, 13, 2 }, &nums);

    try testing.expectError(PolygonCheckError.InvalidPolygon, checkPolygon(testing.allocator, &[_]f64{}));
    try testing.expectError(PolygonCheckError.NonPositiveSide, checkPolygon(testing.allocator, &[_]f64{ -2, 5, 6 }));
}

test "check polygon: edge and extreme cases" {
    try testing.expect(!(try checkPolygon(testing.allocator, &[_]f64{ 1, 2 })));
    try testing.expect(try checkPolygon(testing.allocator, &[_]f64{ 1_000_000_000, 1_000_000_000, 1_000_000_000 }));
}
