//! Quick Select - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/quick_select.py

const std = @import("std");
const testing = std.testing;

pub const QuickSelectError = error{
    OutOfMemory,
};

fn partition(items: []const i64, pivot: i64, allocator: std.mem.Allocator) QuickSelectError!struct {
    less: []i64,
    equal: []i64,
    greater: []i64,
} {
    var less_count: usize = 0;
    var equal_count: usize = 0;
    var greater_count: usize = 0;

    for (items) |item| {
        if (item < pivot) {
            less_count += 1;
        } else if (item > pivot) {
            greater_count += 1;
        } else {
            equal_count += 1;
        }
    }

    const less = try allocator.alloc(i64, less_count);
    errdefer allocator.free(less);
    const equal = try allocator.alloc(i64, equal_count);
    errdefer allocator.free(equal);
    const greater = try allocator.alloc(i64, greater_count);
    errdefer allocator.free(greater);

    var less_idx: usize = 0;
    var equal_idx: usize = 0;
    var greater_idx: usize = 0;
    for (items) |item| {
        if (item < pivot) {
            less[less_idx] = item;
            less_idx += 1;
        } else if (item > pivot) {
            greater[greater_idx] = item;
            greater_idx += 1;
        } else {
            equal[equal_idx] = item;
            equal_idx += 1;
        }
    }

    return .{
        .less = less,
        .equal = equal,
        .greater = greater,
    };
}

/// Returns the value that would appear at `index` in sorted order, or null if
/// the index is invalid.
///
/// Time complexity: O(n) average
/// Space complexity: O(n)
pub fn quickSelect(items: []const i64, index: usize, allocator: std.mem.Allocator) QuickSelectError!?i64 {
    if (index >= items.len) return null;

    const pivot = items[items.len / 2];
    const parts = try partition(items, pivot, allocator);
    defer allocator.free(parts.less);
    defer allocator.free(parts.equal);
    defer allocator.free(parts.greater);

    if (parts.less.len <= index and index < parts.less.len + parts.equal.len) {
        return pivot;
    } else if (index < parts.less.len) {
        return quickSelect(parts.less, index, allocator);
    } else {
        return quickSelect(parts.greater, index - parts.less.len - parts.equal.len, allocator);
    }
}

pub fn median(items: []const i64, allocator: std.mem.Allocator) QuickSelectError!?f64 {
    if (items.len == 0) return null;

    const mid = items.len / 2;
    if (items.len % 2 == 1) {
        return @floatFromInt((try quickSelect(items, mid, allocator)).?);
    }

    const low = (try quickSelect(items, mid - 1, allocator)).?;
    const high = (try quickSelect(items, mid, allocator)).?;
    return (@as(f64, @floatFromInt(low)) + @as(f64, @floatFromInt(high))) / 2.0;
}

test "quick select: examples" {
    const allocator = testing.allocator;
    try testing.expectEqual(@as(?i64, 54), try quickSelect(&[_]i64{ 2, 4, 5, 7, 899, 54, 32 }, 5, allocator));
    try testing.expectEqual(@as(?i64, 4), try quickSelect(&[_]i64{ 2, 4, 5, 7, 899, 54, 32 }, 1, allocator));
    try testing.expectEqual(@as(?i64, 4), try quickSelect(&[_]i64{ 5, 4, 3, 2 }, 2, allocator));
    try testing.expectEqual(@as(?i64, 7), try quickSelect(&[_]i64{ 3, 5, 7, 10, 2, 12 }, 3, allocator));
}

test "quick select: median and boundaries" {
    const allocator = testing.allocator;
    try testing.expectEqual(@as(?f64, 3.0), try median(&[_]i64{ 3, 2, 2, 9, 9 }, allocator));
    try testing.expectEqual(@as(?f64, 6.0), try median(&[_]i64{ 2, 2, 9, 9, 9, 3 }, allocator));
    try testing.expectEqual(@as(?i64, null), try quickSelect(&[_]i64{ 1, 2, 3 }, 3, allocator));
    try testing.expectEqual(@as(?f64, null), try median(&[_]i64{}, allocator));
}
