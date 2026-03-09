//! Sentinel Linear Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/sentinel_linear_search.py

const std = @import("std");
const testing = std.testing;

pub const SentinelSearchError = error{
    OutOfMemory,
};

/// Sentinel linear search. Uses an internal copied buffer because Zig callers
/// often pass immutable slices.
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn sentinelLinearSearch(comptime T: type, items: []const T, target: T, allocator: std.mem.Allocator) SentinelSearchError!?usize {
    var buffer = try allocator.alloc(T, items.len + 1);
    defer allocator.free(buffer);

    @memcpy(buffer[0..items.len], items);
    buffer[items.len] = target;

    var index: usize = 0;
    while (buffer[index] != target) : (index += 1) {}

    if (index == items.len) return null;
    return index;
}

test "sentinel linear search: examples" {
    const allocator = testing.allocator;
    const arr = [_]i32{ 0, 5, 7, 10, 15 };
    try testing.expectEqual(@as(?usize, 0), try sentinelLinearSearch(i32, &arr, 0, allocator));
    try testing.expectEqual(@as(?usize, 4), try sentinelLinearSearch(i32, &arr, 15, allocator));
    try testing.expectEqual(@as(?usize, 1), try sentinelLinearSearch(i32, &arr, 5, allocator));
    try testing.expectEqual(@as(?usize, null), try sentinelLinearSearch(i32, &arr, 6, allocator));
}

test "sentinel linear search: boundaries" {
    const allocator = testing.allocator;
    try testing.expectEqual(@as(?usize, null), try sentinelLinearSearch(i32, &[_]i32{}, 1, allocator));
    try testing.expectEqual(@as(?usize, 0), try sentinelLinearSearch(i32, &[_]i32{42}, 42, allocator));
}
