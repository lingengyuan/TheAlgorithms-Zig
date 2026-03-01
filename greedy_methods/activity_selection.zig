//! Activity Selection - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/other/activity_selection.py

const std = @import("std");
const testing = std.testing;

pub const ActivitySelectionError = error{ LengthMismatch, NotSortedByFinish };

/// Selects a maximal set of non-overlapping activities.
/// `start` and `finish` must be aligned and sorted by non-decreasing finish time.
/// Returns selected activity indices (in scan order). Caller owns returned slice.
/// Time complexity: O(n), space complexity: O(k)
pub fn activitySelection(
    allocator: std.mem.Allocator,
    start: []const i64,
    finish: []const i64,
) (ActivitySelectionError || std.mem.Allocator.Error)![]usize {
    if (start.len != finish.len) return ActivitySelectionError.LengthMismatch;
    if (start.len == 0) return try allocator.alloc(usize, 0);

    for (1..finish.len) |i| {
        if (finish[i] < finish[i - 1]) return ActivitySelectionError.NotSortedByFinish;
    }

    var out = std.ArrayListUnmanaged(usize){};
    errdefer out.deinit(allocator);

    var last_finish = finish[0];
    try out.append(allocator, 0);

    for (1..start.len) |i| {
        if (start[i] >= last_finish) {
            try out.append(allocator, i);
            last_finish = finish[i];
        }
    }

    return try out.toOwnedSlice(allocator);
}

test "activity selection: reference example" {
    const alloc = testing.allocator;
    const start = [_]i64{ 1, 3, 0, 5, 8, 5 };
    const finish = [_]i64{ 2, 4, 6, 7, 9, 9 };

    const chosen = try activitySelection(alloc, &start, &finish);
    defer alloc.free(chosen);

    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 3, 4 }, chosen);
}

test "activity selection: empty input" {
    const alloc = testing.allocator;
    const chosen = try activitySelection(alloc, &[_]i64{}, &[_]i64{});
    defer alloc.free(chosen);
    try testing.expectEqual(@as(usize, 0), chosen.len);
}

test "activity selection: single activity" {
    const alloc = testing.allocator;
    const start = [_]i64{42};
    const finish = [_]i64{43};

    const chosen = try activitySelection(alloc, &start, &finish);
    defer alloc.free(chosen);
    try testing.expectEqualSlices(usize, &[_]usize{0}, chosen);
}

test "activity selection: length mismatch" {
    const alloc = testing.allocator;
    try testing.expectError(
        ActivitySelectionError.LengthMismatch,
        activitySelection(alloc, &[_]i64{ 1, 2 }, &[_]i64{1}),
    );
}

test "activity selection: unsorted finish is rejected" {
    const alloc = testing.allocator;
    const start = [_]i64{ 0, 1, 2 };
    const finish = [_]i64{ 3, 2, 4 };

    try testing.expectError(
        ActivitySelectionError.NotSortedByFinish,
        activitySelection(alloc, &start, &finish),
    );
}

test "activity selection: extreme dense overlap" {
    const alloc = testing.allocator;
    var start: [512]i64 = undefined;
    var finish: [512]i64 = undefined;

    for (0..512) |i| {
        start[i] = @as(i64, @intCast(i));
        finish[i] = @as(i64, @intCast(i + 1));
    }

    const chosen = try activitySelection(alloc, &start, &finish);
    defer alloc.free(chosen);

    try testing.expectEqual(@as(usize, 512), chosen.len);
    try testing.expectEqual(@as(usize, 0), chosen[0]);
    try testing.expectEqual(@as(usize, 511), chosen[chosen.len - 1]);
}
