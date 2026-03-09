//! Median of Medians - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/median_of_medians.py

const std = @import("std");
const testing = std.testing;

pub const MedianOfMediansError = error{
    OutOfMemory,
};

fn sortCopy(items: []const i64, allocator: std.mem.Allocator) MedianOfMediansError![]i64 {
    const copy = try allocator.dupe(i64, items);
    std.mem.sort(i64, copy, {}, std.sort.asc(i64));
    return copy;
}

pub fn medianOfFive(items: []const i64, allocator: std.mem.Allocator) MedianOfMediansError!i64 {
    const copy = try sortCopy(items, allocator);
    defer allocator.free(copy);
    return copy[copy.len / 2];
}

pub fn medianOfMedians(items: []const i64, allocator: std.mem.Allocator) MedianOfMediansError!i64 {
    if (items.len <= 5) return medianOfFive(items, allocator);

    var medians = std.ArrayListUnmanaged(i64){};
    defer medians.deinit(allocator);

    var i: usize = 0;
    while (i < items.len) : (i += 5) {
        const end = @min(i + 5, items.len);
        try medians.append(allocator, try medianOfFive(items[i..end], allocator));
    }

    return medianOfMedians(medians.items, allocator);
}

/// Returns the 1-based rank target using the Python reference semantics.
pub fn quickSelect(items: []const i64, target: usize, allocator: std.mem.Allocator) MedianOfMediansError!i64 {
    if (target == 0 or target > items.len) return -1;

    const pivot = try medianOfMedians(items, allocator);

    var left = std.ArrayListUnmanaged(i64){};
    var right = std.ArrayListUnmanaged(i64){};
    defer left.deinit(allocator);
    defer right.deinit(allocator);

    var used_pivot = false;
    for (items) |item| {
        if (item < pivot) {
            try left.append(allocator, item);
        } else if (item > pivot) {
            try right.append(allocator, item);
        } else if (!used_pivot) {
            used_pivot = true;
        } else {
            try right.append(allocator, item);
        }
    }

    const rank = left.items.len + 1;
    if (rank == target) return pivot;
    if (rank > target) return quickSelect(left.items, target, allocator);
    return quickSelect(right.items, target - rank, allocator);
}

test "median of medians: examples" {
    const allocator = testing.allocator;
    try testing.expectEqual(@as(i64, 5), try medianOfFive(&[_]i64{ 2, 4, 5, 7, 899 }, allocator));
    try testing.expectEqual(@as(i64, 32), try medianOfFive(&[_]i64{ 5, 7, 899, 54, 32 }, allocator));
    try testing.expectEqual(@as(i64, 54), try medianOfMedians(&[_]i64{ 2, 4, 5, 7, 899, 54, 32 }, allocator));
    try testing.expectEqual(@as(i64, 32), try quickSelect(&[_]i64{ 2, 4, 5, 7, 899, 54, 32 }, 5, allocator));
}

test "median of medians: boundaries" {
    const allocator = testing.allocator;
    try testing.expectEqual(@as(i64, 2), try quickSelect(&[_]i64{ 2, 4, 5, 7, 899, 54, 32 }, 1, allocator));
    try testing.expectEqual(@as(i64, 3), try quickSelect(&[_]i64{ 5, 4, 3, 2 }, 2, allocator));
    try testing.expectEqual(@as(i64, 5), try quickSelect(&[_]i64{ 3, 5, 7, 10, 2, 12 }, 3, allocator));
    try testing.expectEqual(@as(i64, -1), try quickSelect(&[_]i64{ 1, 2, 3 }, 4, allocator));
}
