//! Minimum Waiting Time - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/greedy_methods/minimum_waiting_time.py

const std = @import("std");
const testing = std.testing;

/// Computes minimum total waiting time when queries are sorted in ascending order.
/// Each query must finish before the next begins. Query i waits for sum of all
/// queries before it, so sort ascending and multiply position by remaining count.
/// Time complexity: O(n log n)
pub fn minimumWaitingTime(allocator: std.mem.Allocator, queries: []const u64) !u64 {
    if (queries.len <= 1) return 0;

    // Sort a copy
    const sorted = try allocator.alloc(u64, queries.len);
    defer allocator.free(sorted);
    @memcpy(sorted, queries);
    std.mem.sort(u64, sorted, {}, std.sort.asc(u64));

    const n = sorted.len;
    var total: u64 = 0;
    for (sorted, 0..) |q, i| {
        total += q * (n - i - 1);
    }
    return total;
}

test "minimum waiting: examples" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 17), try minimumWaitingTime(alloc, &[_]u64{ 3, 2, 1, 2, 6 }));
    try testing.expectEqual(@as(u64, 4), try minimumWaitingTime(alloc, &[_]u64{ 3, 2, 1 }));
    try testing.expectEqual(@as(u64, 10), try minimumWaitingTime(alloc, &[_]u64{ 1, 2, 3, 4 }));
    try testing.expectEqual(@as(u64, 30), try minimumWaitingTime(alloc, &[_]u64{ 5, 5, 5, 5 }));
}

test "minimum waiting: empty and single" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 0), try minimumWaitingTime(alloc, &[_]u64{}));
    try testing.expectEqual(@as(u64, 0), try minimumWaitingTime(alloc, &[_]u64{7}));
}
