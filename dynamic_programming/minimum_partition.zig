//! Minimum Partition Difference - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/minimum_partition.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const MinimumPartitionError = error{
    NegativeElement,
    ElementTooLarge,
    Overflow,
};

/// Partitions a non-negative integer set into two subsets and returns
/// the minimum absolute difference of subset sums.
/// Time complexity: O(n * total_sum), Space complexity: O(total_sum)
pub fn minimumPartitionDifference(
    allocator: Allocator,
    numbers: []const i64,
) (MinimumPartitionError || Allocator.Error)!usize {
    if (numbers.len == 0) return 0;

    var total: usize = 0;
    for (numbers) |value| {
        if (value < 0) return MinimumPartitionError.NegativeElement;
        if (value > std.math.maxInt(usize)) return MinimumPartitionError.ElementTooLarge;

        const add = @addWithOverflow(total, @as(usize, @intCast(value)));
        if (add[1] != 0) return MinimumPartitionError.Overflow;
        total = add[0];
    }

    const target = total / 2;
    const len_plus = @addWithOverflow(target, @as(usize, 1));
    if (len_plus[1] != 0) return MinimumPartitionError.Overflow;

    const dp = try allocator.alloc(bool, len_plus[0]);
    defer allocator.free(dp);
    @memset(dp, false);
    dp[0] = true;

    for (numbers) |value| {
        const item: usize = @intCast(value);
        if (item > target) continue;

        var s = target;
        while (true) {
            if (s >= item and dp[s - item]) dp[s] = true;
            if (s == 0) break;
            s -= 1;
        }
    }

    var j = target;
    while (true) {
        if (dp[j]) {
            const twice = @mulWithOverflow(j, @as(usize, 2));
            if (twice[1] != 0) return MinimumPartitionError.Overflow;
            return total - twice[0];
        }

        if (j == 0) break;
        j -= 1;
    }

    return total;
}

test "minimum partition: python examples on non-negative domain" {
    try testing.expectEqual(@as(usize, 1), try minimumPartitionDifference(testing.allocator, &[_]i64{ 1, 2, 3, 4, 5 }));
    try testing.expectEqual(@as(usize, 5), try minimumPartitionDifference(testing.allocator, &[_]i64{ 5, 5, 5, 5, 5 }));
    try testing.expectEqual(@as(usize, 0), try minimumPartitionDifference(testing.allocator, &[_]i64{ 5, 5, 5, 5 }));
    try testing.expectEqual(@as(usize, 3), try minimumPartitionDifference(testing.allocator, &[_]i64{3}));
    try testing.expectEqual(@as(usize, 0), try minimumPartitionDifference(testing.allocator, &[_]i64{}));
    try testing.expectEqual(@as(usize, 0), try minimumPartitionDifference(testing.allocator, &[_]i64{ 1, 2, 3, 4 }));
    try testing.expectEqual(@as(usize, 0), try minimumPartitionDifference(testing.allocator, &[_]i64{ 0, 0, 0, 0 }));
    try testing.expectEqual(@as(usize, 9), try minimumPartitionDifference(testing.allocator, &[_]i64{ 9, 9, 9, 9, 9 }));
    try testing.expectEqual(@as(usize, 1), try minimumPartitionDifference(testing.allocator, &[_]i64{ 1, 5, 10, 3 }));
}

test "minimum partition: descending range sample" {
    const values = [_]i64{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    try testing.expectEqual(@as(usize, 1), try minimumPartitionDifference(testing.allocator, &values));
}

test "minimum partition: negative values are rejected" {
    try testing.expectError(MinimumPartitionError.NegativeElement, minimumPartitionDifference(testing.allocator, &[_]i64{ -1, 0, 1 }));
}

test "minimum partition: extreme large list" {
    var values: [200]i64 = undefined;
    for (0..100) |i| values[i] = 1000;
    for (100..200) |i| values[i] = 1000;
    try testing.expectEqual(@as(usize, 0), try minimumPartitionDifference(testing.allocator, &values));
}

test "minimum partition: accumulation overflow is detected" {
    const huge = [_]i64{ std.math.maxInt(i64), std.math.maxInt(i64), std.math.maxInt(i64) };
    try testing.expectError(MinimumPartitionError.Overflow, minimumPartitionDifference(testing.allocator, &huge));
}
