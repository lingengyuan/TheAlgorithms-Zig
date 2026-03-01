//! 0/1 Knapsack Problem - Zig implementation (bottom-up DP)
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/knapsack.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const KnapsackError = error{ LengthMismatch, Overflow };

/// Solves the 0/1 knapsack problem.
/// Returns the maximum total value achievable within the given capacity.
/// Returns error.LengthMismatch if weights.len != values.len.
/// Time complexity: O(n × W), Space complexity: O(n × W)
pub fn knapsack(
    allocator: Allocator,
    capacity: usize,
    weights: []const usize,
    values: []const usize,
) (KnapsackError || Allocator.Error)!usize {
    if (weights.len != values.len) return KnapsackError.LengthMismatch;
    const n = weights.len;
    const n_plus = @addWithOverflow(n, @as(usize, 1));
    if (n_plus[1] != 0) return KnapsackError.Overflow;
    const capacity_plus = @addWithOverflow(capacity, @as(usize, 1));
    if (capacity_plus[1] != 0) return KnapsackError.Overflow;
    const elem_count = @mulWithOverflow(n_plus[0], capacity_plus[0]);
    if (elem_count[1] != 0) return KnapsackError.Overflow;

    // Allocate flattened 2D table of size (n+1) × (capacity+1)
    const dp = try allocator.alloc(usize, elem_count[0]);
    defer allocator.free(dp);
    @memset(dp, 0);

    const cols = capacity_plus[0];

    for (1..n_plus[0]) |i| {
        for (1..capacity_plus[0]) |w| {
            if (weights[i - 1] <= w) {
                const take = @addWithOverflow(values[i - 1], dp[(i - 1) * cols + (w - weights[i - 1])]);
                if (take[1] != 0) return KnapsackError.Overflow;
                const skip = dp[(i - 1) * cols + w];
                dp[i * cols + w] = @max(take[0], skip);
            } else {
                dp[i * cols + w] = dp[(i - 1) * cols + w];
            }
        }
    }

    return dp[n * cols + capacity];
}

// ===== Tests =====

test "knapsack: basic example" {
    const alloc = testing.allocator;
    const weights = [_]usize{ 4, 3, 2, 3 };
    const values = [_]usize{ 3, 2, 4, 4 };
    try testing.expectEqual(@as(usize, 8), try knapsack(alloc, 6, &weights, &values));
}

test "knapsack: example 2" {
    const alloc = testing.allocator;
    const weights = [_]usize{ 1, 3, 5, 2 };
    const values = [_]usize{ 10, 20, 100, 22 };
    try testing.expectEqual(@as(usize, 142), try knapsack(alloc, 10, &weights, &values));
}

test "knapsack: zero capacity" {
    const alloc = testing.allocator;
    const weights = [_]usize{ 1, 2, 3 };
    const values = [_]usize{ 10, 20, 30 };
    try testing.expectEqual(@as(usize, 0), try knapsack(alloc, 0, &weights, &values));
}

test "knapsack: no items" {
    const alloc = testing.allocator;
    const weights = [_]usize{};
    const values = [_]usize{};
    try testing.expectEqual(@as(usize, 0), try knapsack(alloc, 10, &weights, &values));
}

test "knapsack: single item fits" {
    const alloc = testing.allocator;
    const weights = [_]usize{5};
    const values = [_]usize{100};
    try testing.expectEqual(@as(usize, 100), try knapsack(alloc, 5, &weights, &values));
}

test "knapsack: single item too heavy" {
    const alloc = testing.allocator;
    const weights = [_]usize{10};
    const values = [_]usize{100};
    try testing.expectEqual(@as(usize, 0), try knapsack(alloc, 5, &weights, &values));
}

test "knapsack: length mismatch returns error" {
    const alloc = testing.allocator;
    const weights = [_]usize{ 4, 3, 2, 3 };
    const values = [_]usize{ 3, 2, 4 }; // 3 values vs 4 weights
    const result = knapsack(alloc, 6, &weights, &values);
    try testing.expectError(KnapsackError.LengthMismatch, result);
}

test "knapsack: oversize dimensions return overflow" {
    const fake_ptr: [*]const usize = @ptrFromInt(@alignOf(usize));
    const fake = fake_ptr[0..1];
    try testing.expectError(KnapsackError.Overflow, knapsack(testing.allocator, std.math.maxInt(usize), fake, fake));
}
