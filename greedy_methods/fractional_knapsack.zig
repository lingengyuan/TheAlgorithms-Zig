//! Fractional Knapsack - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/greedy_methods/fractional_knapsack.py

const std = @import("std");
const testing = std.testing;

pub const KnapsackError = error{LengthMismatch};

const Item = struct { value: f64, weight: f64 };

fn byRatioDesc(_: void, a: Item, b: Item) bool {
    return (a.value / a.weight) > (b.value / b.weight);
}

/// Solves the fractional knapsack problem.
/// Returns the maximum value obtainable with the given weight capacity.
/// Items can be taken fractionally.
/// Time complexity: O(n log n)
pub fn fractionalKnapsack(
    allocator: std.mem.Allocator,
    values: []const f64,
    weights: []const f64,
    capacity: f64,
) (KnapsackError || std.mem.Allocator.Error)!f64 {
    if (values.len != weights.len) return KnapsackError.LengthMismatch;
    if (capacity <= 0 or values.len == 0) return 0;

    const items = try allocator.alloc(Item, values.len);
    defer allocator.free(items);
    for (items, 0..) |*item, i| {
        item.* = .{ .value = values[i], .weight = weights[i] };
    }
    std.mem.sort(Item, items, {}, byRatioDesc);

    var remaining = capacity;
    var total: f64 = 0;
    for (items) |item| {
        if (remaining <= 0) break;
        if (item.weight <= remaining) {
            total += item.value;
            remaining -= item.weight;
        } else {
            total += item.value * (remaining / item.weight);
            remaining = 0;
        }
    }
    return total;
}

test "fractional knapsack: basic" {
    const alloc = testing.allocator;
    const v = [_]f64{ 60, 100, 120 };
    const w = [_]f64{ 10, 20, 30 };
    try testing.expectApproxEqAbs(@as(f64, 240), try fractionalKnapsack(alloc, &v, &w, 50), 1e-9);
}

test "fractional knapsack: partial item" {
    const alloc = testing.allocator;
    const v = [_]f64{ 10, 40, 30, 50 };
    const w = [_]f64{ 5, 4, 6, 3 };
    const result = try fractionalKnapsack(alloc, &v, &w, 10);
    try testing.expectApproxEqAbs(@as(f64, 105), result, 1e-9);
}

test "fractional knapsack: zero capacity" {
    const alloc = testing.allocator;
    const v = [_]f64{ 10, 20 };
    const w = [_]f64{ 5, 10 };
    try testing.expectApproxEqAbs(@as(f64, 0), try fractionalKnapsack(alloc, &v, &w, 0), 1e-9);
}

test "fractional knapsack: length mismatch" {
    const alloc = testing.allocator;
    const v = [_]f64{ 10, 20 };
    const w = [_]f64{5};
    try testing.expectError(KnapsackError.LengthMismatch, fractionalKnapsack(alloc, &v, &w, 10));
}
