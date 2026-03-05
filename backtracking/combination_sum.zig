//! Combination Sum (Backtracking) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/combination_sum.py

const std = @import("std");
const testing = std.testing;

pub const CombinationSumError = error{ EmptyCandidates, NonPositiveCandidate };

fn backtrack(
    allocator: std.mem.Allocator,
    candidates: []const i32,
    start_index: usize,
    remaining: i32,
    path: *std.ArrayListUnmanaged(i32),
    result: *std.ArrayListUnmanaged([]i32),
) std.mem.Allocator.Error!void {
    if (remaining == 0) {
        const combination = try allocator.dupe(i32, path.items);
        errdefer allocator.free(combination);
        try result.append(allocator, combination);
        return;
    }

    var idx = start_index;
    while (idx < candidates.len) : (idx += 1) {
        const value = candidates[idx];
        if (remaining >= value) {
            try path.append(allocator, value);
            try backtrack(allocator, candidates, idx, remaining - value, path, result);
            _ = path.pop();
        }
    }
}

/// Computes all non-decreasing combinations that sum to `target`.
///
/// API note: Python reference only rejects negatives, but candidate `0` leads to
/// non-terminating recursion. This Zig version rejects non-positive candidates.
///
/// Time complexity: exponential in target/candidate domain
/// Space complexity: O(target/min_candidate) recursion depth (excluding output)
pub fn combinationSum(
    allocator: std.mem.Allocator,
    candidates: []const i32,
    target: i32,
    result: *std.ArrayListUnmanaged([]i32),
) (CombinationSumError || std.mem.Allocator.Error)!void {
    if (candidates.len == 0) return CombinationSumError.EmptyCandidates;

    for (candidates) |value| {
        if (value <= 0) return CombinationSumError.NonPositiveCandidate;
    }

    var path = std.ArrayListUnmanaged(i32){};
    defer path.deinit(allocator);

    if (target < 0) return;
    try backtrack(allocator, candidates, 0, target, &path, result);
}

test "combination sum: python examples" {
    const alloc = testing.allocator;

    var result1 = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result1.items) |comb| alloc.free(comb);
        result1.deinit(alloc);
    }
    try combinationSum(alloc, &[_]i32{ 2, 3, 5 }, 8, &result1);
    try testing.expectEqual(@as(usize, 3), result1.items.len);
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 2, 2, 2 }, result1.items[0]);
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 3, 3 }, result1.items[1]);
    try testing.expectEqualSlices(i32, &[_]i32{ 3, 5 }, result1.items[2]);

    var result2 = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result2.items) |comb| alloc.free(comb);
        result2.deinit(alloc);
    }
    try combinationSum(alloc, &[_]i32{ 2, 3, 6, 7 }, 7, &result2);
    try testing.expectEqual(@as(usize, 2), result2.items.len);
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 2, 3 }, result2.items[0]);
    try testing.expectEqualSlices(i32, &[_]i32{7}, result2.items[1]);
}

test "combination sum: validation errors" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]i32){};
    defer result.deinit(alloc);

    try testing.expectError(CombinationSumError.EmptyCandidates, combinationSum(alloc, &[_]i32{}, 1, &result));
    try testing.expectError(CombinationSumError.NonPositiveCandidate, combinationSum(alloc, &[_]i32{ -8, 2, 3 }, 1, &result));
    try testing.expectError(CombinationSumError.NonPositiveCandidate, combinationSum(alloc, &[_]i32{ 0, 2, 3 }, 7, &result));
}

test "combination sum: negative target returns empty" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result.items) |comb| alloc.free(comb);
        result.deinit(alloc);
    }

    try combinationSum(alloc, &[_]i32{ 2, 3, 5 }, -1, &result);
    try testing.expectEqual(@as(usize, 0), result.items.len);
}

test "combination sum: extreme target single candidate" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result.items) |comb| alloc.free(comb);
        result.deinit(alloc);
    }

    try combinationSum(alloc, &[_]i32{2}, 40, &result);
    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqual(@as(usize, 20), result.items[0].len);
    for (result.items[0]) |v| try testing.expectEqual(@as(i32, 2), v);
}
