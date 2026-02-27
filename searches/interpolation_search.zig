//! Interpolation Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/interpolation_search.py

const std = @import("std");
const testing = std.testing;

/// Interpolation search on a sorted i32 slice. Returns the index of target, or null.
/// Works best for uniformly distributed values.
/// Time complexity: O(log log n) average, O(n) worst, Space complexity: O(1)
pub fn interpolationSearch(items: []const i32, target: i32) ?usize {
    if (items.len == 0) return null;

    var low: usize = 0;
    var high: usize = items.len - 1;

    while (low <= high and target >= items[low] and target <= items[high]) {
        if (items[high] == items[low]) {
            if (items[low] == target) return low;
            return null;
        }

        const low_val = @as(i64, items[low]);
        const high_val = @as(i64, items[high]);
        const target_val = @as(i64, target);

        const numerator = @as(i64, @intCast(high - low)) * (target_val - low_val);
        const denominator = high_val - low_val;
        const offset: usize = @intCast(@divTrunc(numerator, denominator));
        const pos = low + offset;
        if (pos > high) return null;

        if (items[pos] == target) {
            return pos;
        } else if (items[pos] < target) {
            low = pos + 1;
        } else {
            if (pos == 0) break;
            high = pos - 1;
        }
    }

    return null;
}

test "interpolation search: found in middle" {
    const arr = [_]i32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    try testing.expectEqual(@as(?usize, 3), interpolationSearch(&arr, 40));
}

test "interpolation search: found at boundaries" {
    const arr = [_]i32{ 5, 10, 15, 20, 25 };
    try testing.expectEqual(@as(?usize, 0), interpolationSearch(&arr, 5));
    try testing.expectEqual(@as(?usize, 4), interpolationSearch(&arr, 25));
}

test "interpolation search: not found" {
    const arr = [_]i32{ 10, 20, 30, 40, 50 };
    try testing.expectEqual(@as(?usize, null), interpolationSearch(&arr, 35));
}

test "interpolation search: duplicate values" {
    const arr = [_]i32{ 2, 2, 2, 2, 2 };
    const result = interpolationSearch(&arr, 2);
    try testing.expect(result != null);
    try testing.expectEqual(@as(i32, 2), arr[result.?]);
}

test "interpolation search: empty array" {
    const arr = [_]i32{};
    try testing.expectEqual(@as(?usize, null), interpolationSearch(&arr, 1));
}

test "interpolation search: out of range" {
    const arr = [_]i32{ 10, 20, 30, 40, 50 };
    try testing.expectEqual(@as(?usize, null), interpolationSearch(&arr, -10));
    try testing.expectEqual(@as(?usize, null), interpolationSearch(&arr, 100));
}
