//! 0/1 Knapsack Problem - Zig implementation (bottom-up DP)
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/knapsack.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const KnapsackError = error{LengthMismatch};

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

    // Allocate flattened 2D table of size (n+1) × (capacity+1)
    const dp = try allocator.alloc(usize, (n + 1) * (capacity + 1));
    defer allocator.free(dp);
    @memset(dp, 0);

    const cols = capacity + 1;

    for (1..n + 1) |i| {
        for (1..capacity + 1) |w| {
            if (weights[i - 1] <= w) {
                const take = values[i - 1] + dp[(i - 1) * cols + (w - weights[i - 1])];
                const skip = dp[(i - 1) * cols + w];
                dp[i * cols + w] = @max(take, skip);
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
