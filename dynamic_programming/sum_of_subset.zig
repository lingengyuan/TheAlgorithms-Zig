//! Sum Of Subset (compatibility wrapper) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/sum_of_subset.py

const std = @import("std");
const testing = std.testing;
const existing = @import("subset_sum.zig");

pub const SumOfSubsetError = existing.SubsetSumError;

/// Returns whether a subset with the required sum exists.
/// Time complexity: O(n * target), Space complexity: O(target)
pub fn isSumSubset(
    allocator: std.mem.Allocator,
    numbers: []const i64,
    required_sum: i64,
) (SumOfSubsetError || std.mem.Allocator.Error)!bool {
    return existing.isSubsetSum(allocator, numbers, required_sum);
}

test "sum of subset: python examples" {
    try testing.expect(!(try isSumSubset(testing.allocator, &[_]i64{ 2, 4, 6, 8 }, 5)));
    try testing.expect(try isSumSubset(testing.allocator, &[_]i64{ 2, 4, 6, 8 }, 14));
}

test "sum of subset: boundaries" {
    try testing.expect(try isSumSubset(testing.allocator, &[_]i64{}, 0));
    try testing.expect(!(try isSumSubset(testing.allocator, &[_]i64{}, 1)));
}

test "sum of subset: extreme repeated values" {
    var values: [256]i64 = undefined;
    for (&values, 0..) |*item, index| item.* = @intCast((index % 9) + 1);
    try testing.expect(try isSumSubset(testing.allocator, &values, 500));
}
