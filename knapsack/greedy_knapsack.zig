//! Greedy Knapsack - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/knapsack/greedy_knapsack.py

const std = @import("std");
const testing = std.testing;

pub const GreedyKnapsackError = error{
    InvalidInput,
    OutOfMemory,
};

const Item = struct {
    profit: f64,
    weight: f64,
    ratio: f64,
};

fn byRatioDesc(_: void, a: Item, b: Item) bool {
    return a.ratio > b.ratio;
}

/// Fractional greedy knapsack following the Python reference validation rules.
///
/// Time complexity: O(n log n)
/// Space complexity: O(n)
pub fn calcProfit(profit: []const f64, weight: []const f64, max_weight: f64, allocator: std.mem.Allocator) GreedyKnapsackError!f64 {
    if (profit.len != weight.len) return GreedyKnapsackError.InvalidInput;
    if (max_weight <= 0) return GreedyKnapsackError.InvalidInput;
    for (profit) |p| if (p < 0) return GreedyKnapsackError.InvalidInput;
    for (weight) |w| if (w <= 0) return GreedyKnapsackError.InvalidInput;

    const items = try allocator.alloc(Item, profit.len);
    defer allocator.free(items);
    for (profit, weight, 0..) |p, w, i| {
        items[i] = .{ .profit = p, .weight = w, .ratio = p / w };
    }
    std.mem.sort(Item, items, {}, byRatioDesc);

    var used: f64 = 0;
    var gain: f64 = 0;
    for (items) |item| {
        if (used >= max_weight) break;
        if (max_weight - used >= item.weight) {
            used += item.weight;
            gain += item.profit;
        } else {
            gain += (max_weight - used) / item.weight * item.profit;
            break;
        }
    }
    return gain;
}

test "greedy knapsack: python reference" {
    const allocator = testing.allocator;
    try testing.expectApproxEqAbs(@as(f64, 6), try calcProfit(&[_]f64{ 1, 2, 3 }, &[_]f64{ 3, 4, 5 }, 15, allocator), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 27), try calcProfit(&[_]f64{ 10, 9, 8 }, &[_]f64{ 3, 4, 5 }, 25, allocator), 1e-9);
}

test "greedy knapsack: boundaries" {
    const allocator = testing.allocator;
    try testing.expectError(GreedyKnapsackError.InvalidInput, calcProfit(&[_]f64{ 1, 2 }, &[_]f64{1}, 5, allocator));
    try testing.expectError(GreedyKnapsackError.InvalidInput, calcProfit(&[_]f64{ 1, -2 }, &[_]f64{ 1, 2 }, 5, allocator));
    try testing.expectError(GreedyKnapsackError.InvalidInput, calcProfit(&[_]f64{ 1, 2 }, &[_]f64{ 1, 0 }, 5, allocator));
}
