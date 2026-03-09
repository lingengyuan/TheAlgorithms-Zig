//! Knapsack - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/knapsack/knapsack.py

const std = @import("std");
const testing = std.testing;

pub const KnapsackError = error{
    OutOfMemory,
    InvalidInput,
};

fn recur(capacity: i32, weights: []const i32, values: []const i32, counter: usize, allow_repetition: bool, memo: []i32) i32 {
    if (counter == 0 or capacity <= 0) return 0;

    const cap_usize: usize = @intCast(capacity);
    const key = counter * (memo.len / (weights.len + 1)) + cap_usize;
    if (memo[key] != -1) return memo[key];

    const result = if (weights[counter - 1] > capacity)
        recur(capacity, weights, values, counter - 1, allow_repetition, memo)
    else blk: {
        const left_capacity = capacity - weights[counter - 1];
        const next_counter = if (allow_repetition) counter else counter - 1;
        const include = values[counter - 1] + recur(left_capacity, weights, values, next_counter, allow_repetition, memo);
        const exclude = recur(capacity, weights, values, counter - 1, allow_repetition, memo);
        break :blk @max(include, exclude);
    };

    memo[key] = result;
    return result;
}

/// 0-1 / unbounded knapsack by recursive memoization.
///
/// Time complexity: O(counter * capacity)
/// Space complexity: O(counter * capacity)
pub fn knapsack(capacity: i32, weights: []const i32, values: []const i32, counter: usize, allow_repetition: bool, allocator: std.mem.Allocator) KnapsackError!i32 {
    if (weights.len != values.len or counter > weights.len or capacity < 0) return KnapsackError.InvalidInput;

    const cols: usize = @intCast(capacity + 1);
    const memo = try allocator.alloc(i32, (weights.len + 1) * cols);
    defer allocator.free(memo);
    @memset(memo, -1);

    return recur(capacity, weights, values, counter, allow_repetition, memo);
}

test "knapsack: python reference" {
    const allocator = testing.allocator;
    const cap = 50;
    const val = [_]i32{ 60, 100, 120 };
    const w = [_]i32{ 10, 20, 30 };
    try testing.expectEqual(@as(i32, 220), try knapsack(cap, &w, &val, val.len, false, allocator));
    try testing.expectEqual(@as(i32, 300), try knapsack(cap, &w, &val, val.len, true, allocator));
}

test "knapsack: boundaries" {
    const allocator = testing.allocator;
    try testing.expectEqual(@as(i32, 0), try knapsack(0, &[_]i32{ 1, 2 }, &[_]i32{ 3, 4 }, 2, false, allocator));
    try testing.expectEqual(@as(i32, 0), try knapsack(5, &[_]i32{}, &[_]i32{}, 0, false, allocator));
    try testing.expectError(KnapsackError.InvalidInput, knapsack(-1, &[_]i32{1}, &[_]i32{1}, 1, false, allocator));
}
