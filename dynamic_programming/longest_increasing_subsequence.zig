//! Longest Increasing Subsequence (LIS) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/longest_increasing_subsequence.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Returns the length of the longest strictly increasing subsequence.
/// Uses patience-sorting style tails array with binary search.
/// Time complexity: O(n log n), Space complexity: O(n)
pub fn longestIncreasingSubsequenceLength(allocator: Allocator, items: []const i64) !usize {
    if (items.len == 0) return 0;

    const tails = try allocator.alloc(i64, items.len);
    defer allocator.free(tails);
    var size: usize = 0;

    for (items) |value| {
        var lo: usize = 0;
        var hi: usize = size;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (tails[mid] < value) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        tails[lo] = value;
        if (lo == size) size += 1;
    }

    return size;
}

test "lis: classic example" {
    const alloc = testing.allocator;
    const arr = [_]i64{ 10, 22, 9, 33, 21, 50, 41, 60 };
    try testing.expectEqual(@as(usize, 5), try longestIncreasingSubsequenceLength(alloc, &arr));
}

test "lis: strictly decreasing" {
    const alloc = testing.allocator;
    const arr = [_]i64{ 9, 7, 5, 3, 1 };
    try testing.expectEqual(@as(usize, 1), try longestIncreasingSubsequenceLength(alloc, &arr));
}

test "lis: duplicates" {
    const alloc = testing.allocator;
    const arr = [_]i64{ 2, 2, 2, 2 };
    try testing.expectEqual(@as(usize, 1), try longestIncreasingSubsequenceLength(alloc, &arr));
}

test "lis: mixed negatives and positives" {
    const alloc = testing.allocator;
    const arr = [_]i64{ -7, -3, -5, 0, 2, 1, 3 };
    try testing.expectEqual(@as(usize, 5), try longestIncreasingSubsequenceLength(alloc, &arr)); // -7,-5,0,1,3
}

test "lis: empty input" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 0), try longestIncreasingSubsequenceLength(alloc, &[_]i64{}));
}
