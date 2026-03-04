//! Largest Rectangle Histogram - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/stacks/largest_rectangle_histogram.py

const std = @import("std");
const testing = std.testing;

/// Computes largest rectangle area in histogram.
/// Time complexity: O(n), Space complexity: O(n)
pub fn largestRectangleArea(allocator: std.mem.Allocator, heights: []const i64) !i64 {
    if (heights.len == 0) return 0;

    for (heights) |h| {
        if (h < 0) return error.InvalidHeight;
    }

    var stack = std.ArrayListUnmanaged(usize){};
    defer stack.deinit(allocator);

    var max_area: i64 = 0;

    var i: usize = 0;
    while (i <= heights.len) : (i += 1) {
        const current_height: i64 = if (i == heights.len) 0 else heights[i];

        while (stack.items.len > 0 and current_height < heights[stack.items[stack.items.len - 1]]) {
            const top_index = stack.pop().?;
            const h = heights[top_index];
            const width: usize = if (stack.items.len == 0) i else i - stack.items[stack.items.len - 1] - 1;

            const mul = @mulWithOverflow(h, @as(i64, @intCast(width)));
            if (mul[1] != 0) return error.Overflow;
            if (mul[0] > max_area) max_area = mul[0];
        }

        try stack.append(allocator, i);
    }

    return max_area;
}

test "largest rectangle histogram: python samples" {
    try testing.expectEqual(@as(i64, 10), try largestRectangleArea(testing.allocator, &[_]i64{ 2, 1, 5, 6, 2, 3 }));
    try testing.expectEqual(@as(i64, 4), try largestRectangleArea(testing.allocator, &[_]i64{ 2, 4 }));
    try testing.expectEqual(@as(i64, 12), try largestRectangleArea(testing.allocator, &[_]i64{ 6, 2, 5, 4, 5, 1, 6 }));
    try testing.expectEqual(@as(i64, 1), try largestRectangleArea(testing.allocator, &[_]i64{1}));
}

test "largest rectangle histogram: invalid and empty" {
    try testing.expectEqual(@as(i64, 0), try largestRectangleArea(testing.allocator, &[_]i64{}));
    try testing.expectError(error.InvalidHeight, largestRectangleArea(testing.allocator, &[_]i64{ 1, -1, 2 }));
}

test "largest rectangle histogram: extreme" {
    const n: usize = 50_000;

    const flat = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(flat);
    @memset(flat, 1);
    try testing.expectEqual(@as(i64, @intCast(n)), try largestRectangleArea(testing.allocator, flat));

    var increasing = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(increasing);
    for (0..n) |i| increasing[i] = @intCast(i + 1);

    const area = try largestRectangleArea(testing.allocator, increasing);
    // For strictly increasing heights, area should be positive and at least n.
    try testing.expect(area >= @as(i64, @intCast(n)));
}
