//! Sum of Subsets - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/sum_of_subsets.py

const std = @import("std");
const testing = std.testing;

pub const SubsetError = error{NegativeElement};

fn createStateSpaceTree(
    allocator: std.mem.Allocator,
    nums: []const i32,
    max_sum: i32,
    num_index: usize,
    path: *std.ArrayListUnmanaged(i32),
    result: *std.ArrayListUnmanaged([]i32),
    remaining_nums_sum: i32,
    current_sum: i32,
) std.mem.Allocator.Error!void {
    if (current_sum > max_sum or (remaining_nums_sum + current_sum) < max_sum) return;

    if (current_sum == max_sum) {
        const subset = try allocator.dupe(i32, path.items);
        errdefer allocator.free(subset);
        try result.append(allocator, subset);
        return;
    }

    var index = num_index;
    while (index < nums.len) : (index += 1) {
        try path.append(allocator, nums[index]);
        try createStateSpaceTree(
            allocator,
            nums,
            max_sum,
            index + 1,
            path,
            result,
            remaining_nums_sum - nums[index],
            current_sum + nums[index],
        );
        _ = path.pop();
    }
}

/// Generates all subsets that sum to `max_sum`.
///
/// API note: input is expected to contain non-negative integers, matching the
/// reference problem definition. Negative values are rejected.
///
/// Time complexity: exponential in `nums.len`
/// Space complexity: O(n) recursion depth (excluding output)
pub fn generateSumOfSubsetsSolutions(
    allocator: std.mem.Allocator,
    nums: []const i32,
    max_sum: i32,
    result: *std.ArrayListUnmanaged([]i32),
) (SubsetError || std.mem.Allocator.Error)!void {
    var remaining: i32 = 0;
    for (nums) |num| {
        if (num < 0) return SubsetError.NegativeElement;
        remaining += num;
    }

    var path = std.ArrayListUnmanaged(i32){};
    defer path.deinit(allocator);

    try createStateSpaceTree(allocator, nums, max_sum, 0, &path, result, remaining, 0);
}

test "sum of subsets: python examples" {
    const alloc = testing.allocator;

    var result1 = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result1.items) |subset| alloc.free(subset);
        result1.deinit(alloc);
    }

    try generateSumOfSubsetsSolutions(alloc, &[_]i32{ 3, 34, 4, 12, 5, 2 }, 9, &result1);
    try testing.expectEqual(@as(usize, 2), result1.items.len);
    try testing.expectEqualSlices(i32, &[_]i32{ 3, 4, 2 }, result1.items[0]);
    try testing.expectEqualSlices(i32, &[_]i32{ 4, 5 }, result1.items[1]);

    var result2 = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result2.items) |subset| alloc.free(subset);
        result2.deinit(alloc);
    }

    try generateSumOfSubsetsSolutions(alloc, &[_]i32{ 3, 34, 4, 12, 5, 2 }, 3, &result2);
    try testing.expectEqual(@as(usize, 1), result2.items.len);
    try testing.expectEqualSlices(i32, &[_]i32{3}, result2.items[0]);
}

test "sum of subsets: empty and zero target" {
    const alloc = testing.allocator;

    var result_empty_zero = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result_empty_zero.items) |subset| alloc.free(subset);
        result_empty_zero.deinit(alloc);
    }
    try generateSumOfSubsetsSolutions(alloc, &[_]i32{}, 0, &result_empty_zero);
    try testing.expectEqual(@as(usize, 1), result_empty_zero.items.len);
    try testing.expectEqual(@as(usize, 0), result_empty_zero.items[0].len);

    var result_empty_non_zero = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result_empty_non_zero.items) |subset| alloc.free(subset);
        result_empty_non_zero.deinit(alloc);
    }
    try generateSumOfSubsetsSolutions(alloc, &[_]i32{}, 1, &result_empty_non_zero);
    try testing.expectEqual(@as(usize, 0), result_empty_non_zero.items.len);
}

test "sum of subsets: invalid negative element and extreme" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]i32){};
    defer result.deinit(alloc);

    try testing.expectError(
        SubsetError.NegativeElement,
        generateSumOfSubsetsSolutions(alloc, &[_]i32{ 1, -2, 3 }, 2, &result),
    );

    var result_extreme = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result_extreme.items) |subset| alloc.free(subset);
        result_extreme.deinit(alloc);
    }

    try generateSumOfSubsetsSolutions(
        alloc,
        &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
        15,
        &result_extreme,
    );
    try testing.expect(result_extreme.items.len > 0);
}
