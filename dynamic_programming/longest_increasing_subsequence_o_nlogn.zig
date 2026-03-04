//! Longest Increasing Subsequence Length O(n log n) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/longest_increasing_subsequence_o_nlogn.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Returns LIS length using strict increase relation.
/// Time complexity: O(n log n), Space complexity: O(n)
pub fn longestIncreasingSubsequenceLengthNlogN(
    allocator: Allocator,
    values: []const i64,
) !usize {
    if (values.len == 0) return 0;

    const tail = try allocator.alloc(i64, values.len);
    defer allocator.free(tail);

    var length: usize = 1;
    tail[0] = values[0];

    for (values[1..]) |value| {
        if (value < tail[0]) {
            tail[0] = value;
        } else if (value > tail[length - 1]) {
            tail[length] = value;
            length += 1;
        } else {
            const idx = ceilIndex(tail[0..length], value);
            tail[idx] = value;
        }
    }

    return length;
}

fn ceilIndex(tail: []const i64, key: i64) usize {
    var left: isize = -1;
    var right: isize = @intCast(tail.len - 1);

    while (right - left > 1) {
        const middle: isize = @intCast(@divTrunc(left + right, @as(isize, 2)));
        if (tail[@intCast(middle)] >= key) {
            right = middle;
        } else {
            left = middle;
        }
    }
    return @intCast(right);
}

test "lis nlogn: python examples" {
    try testing.expectEqual(@as(usize, 6), try longestIncreasingSubsequenceLengthNlogN(
        testing.allocator,
        &[_]i64{ 2, 5, 3, 7, 11, 8, 10, 13, 6 },
    ));
    try testing.expectEqual(@as(usize, 0), try longestIncreasingSubsequenceLengthNlogN(testing.allocator, &[_]i64{}));
    try testing.expectEqual(@as(usize, 6), try longestIncreasingSubsequenceLengthNlogN(
        testing.allocator,
        &[_]i64{ 0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15 },
    ));
    try testing.expectEqual(@as(usize, 1), try longestIncreasingSubsequenceLengthNlogN(
        testing.allocator,
        &[_]i64{ 5, 4, 3, 2, 1 },
    ));
}

test "lis nlogn: extreme long descending array" {
    var arr: [20000]i64 = undefined;
    for (0..arr.len) |i| arr[i] = @intCast(arr.len - i);
    try testing.expectEqual(@as(usize, 1), try longestIncreasingSubsequenceLengthNlogN(testing.allocator, &arr));
}
