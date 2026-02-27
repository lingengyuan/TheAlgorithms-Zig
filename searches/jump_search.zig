//! Jump Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/jump_search.py

const std = @import("std");
const math = std.math;
const testing = std.testing;

/// Jump search on a sorted slice. Returns the index of target, or null if not found.
/// Time complexity: O(√n), Space complexity: O(1)
pub fn jumpSearch(comptime T: type, items: []const T, target: T) ?usize {
    if (items.len == 0) return null;

    const n = items.len;
    const block_size = math.sqrt(n);

    var prev: usize = 0;
    var step: usize = block_size;

    // Jump forward until we overshoot or reach the end
    while (items[@min(step, n) - 1] < target) {
        prev = step;
        step += block_size;
        if (prev >= n) return null;
    }

    // Linear search within the block
    while (prev < @min(step, n)) {
        if (items[prev] == target) return prev;
        prev += 1;
    }
    return null;
}

test "jump search: found" {
    const arr = [_]i32{ 0, 1, 2, 3, 4, 5 };
    try testing.expectEqual(@as(?usize, 3), jumpSearch(i32, &arr, 3));
}

test "jump search: found negative" {
    const arr = [_]i32{ -5, -2, -1 };
    try testing.expectEqual(@as(?usize, 2), jumpSearch(i32, &arr, -1));
}

test "jump search: not found" {
    const arr = [_]i32{ 0, 5, 10, 20 };
    try testing.expectEqual(@as(?usize, null), jumpSearch(i32, &arr, 8));
}

test "jump search: large array" {
    const arr = [_]i32{ 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610 };
    try testing.expectEqual(@as(?usize, 10), jumpSearch(i32, &arr, 55));
}

test "jump search: empty array" {
    const arr = [_]i32{};
    try testing.expectEqual(@as(?usize, null), jumpSearch(i32, &arr, 1));
}

test "jump search: single element found" {
    const arr = [_]i32{5};
    try testing.expectEqual(@as(?usize, 0), jumpSearch(i32, &arr, 5));
}
