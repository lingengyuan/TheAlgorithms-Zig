//! Optimal Binary Search Tree (Cost) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/optimal_binary_search_tree.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const OptimalBstError = error{
    LengthMismatch,
    Overflow,
};

const Node = struct {
    key: i64,
    freq: u64,
};

fn index(cols: usize, row: usize, col: usize) usize {
    return row * cols + col;
}

/// Computes the minimal successful-search cost for an optimal BST.
/// Keys are sorted internally together with their frequencies, matching
/// the Python reference preprocessing.
/// Time complexity: O(n^2) with Knuth optimization, Space complexity: O(n^2)
pub fn optimalBinarySearchTreeCost(
    allocator: Allocator,
    keys: []const i64,
    frequencies: []const u64,
) (OptimalBstError || Allocator.Error)!u64 {
    if (keys.len != frequencies.len) return OptimalBstError.LengthMismatch;
    const n = keys.len;
    if (n == 0) return 0;

    const nodes = try allocator.alloc(Node, n);
    defer allocator.free(nodes);
    for (0..n) |i| {
        nodes[i] = .{ .key = keys[i], .freq = frequencies[i] };
    }
    std.mem.sort(Node, nodes, {}, struct {
        fn lessThan(_: void, a: Node, b: Node) bool {
            return a.key < b.key;
        }
    }.lessThan);

    const cells = @mulWithOverflow(n, n);
    if (cells[1] != 0) return OptimalBstError.Overflow;

    const dp = try allocator.alloc(u64, cells[0]);
    defer allocator.free(dp);
    const total = try allocator.alloc(u64, cells[0]);
    defer allocator.free(total);
    const root = try allocator.alloc(usize, cells[0]);
    defer allocator.free(root);

    @memset(dp, 0);
    @memset(total, 0);
    @memset(root, 0);

    for (0..n) |i| {
        dp[index(n, i, i)] = nodes[i].freq;
        total[index(n, i, i)] = nodes[i].freq;
        root[index(n, i, i)] = i;
    }

    var interval_len: usize = 2;
    while (interval_len <= n) : (interval_len += 1) {
        var i: usize = 0;
        while (i + interval_len <= n) : (i += 1) {
            const j = i + interval_len - 1;

            dp[index(n, i, j)] = std.math.maxInt(u64);

            const total_sum = @addWithOverflow(total[index(n, i, j - 1)], nodes[j].freq);
            if (total_sum[1] != 0) return OptimalBstError.Overflow;
            total[index(n, i, j)] = total_sum[0];

            const start_r = root[index(n, i, j - 1)];
            const end_r = root[index(n, i + 1, j)];

            var r = start_r;
            while (r <= end_r) : (r += 1) {
                const left = if (r != i) dp[index(n, i, r - 1)] else 0;
                const right = if (r != j) dp[index(n, r + 1, j)] else 0;

                const left_plus_total = @addWithOverflow(left, total[index(n, i, j)]);
                if (left_plus_total[1] != 0) return OptimalBstError.Overflow;
                const cost = @addWithOverflow(left_plus_total[0], right);
                if (cost[1] != 0) return OptimalBstError.Overflow;

                if (cost[0] < dp[index(n, i, j)]) {
                    dp[index(n, i, j)] = cost[0];
                    root[index(n, i, j)] = r;
                }
            }
        }
    }

    return dp[index(n, 0, n - 1)];
}

test "optimal bst: python sample cost" {
    const keys = [_]i64{ 12, 10, 20, 42, 25, 37 };
    const freqs = [_]u64{ 8, 34, 50, 3, 40, 30 };
    try testing.expectEqual(@as(u64, 324), try optimalBinarySearchTreeCost(testing.allocator, &keys, &freqs));
}

test "optimal bst: boundary behavior" {
    try testing.expectEqual(@as(u64, 0), try optimalBinarySearchTreeCost(testing.allocator, &[_]i64{}, &[_]u64{}));

    const keys = [_]i64{7};
    const freqs = [_]u64{42};
    try testing.expectEqual(@as(u64, 42), try optimalBinarySearchTreeCost(testing.allocator, &keys, &freqs));
}

test "optimal bst: mismatched input length" {
    const keys = [_]i64{ 1, 2, 3 };
    const freqs = [_]u64{ 10, 20 };
    try testing.expectError(OptimalBstError.LengthMismatch, optimalBinarySearchTreeCost(testing.allocator, &keys, &freqs));
}

test "optimal bst: extreme overflow detection" {
    const keys = [_]i64{ 1, 2, 3 };
    const freqs = [_]u64{ std.math.maxInt(u64), std.math.maxInt(u64), std.math.maxInt(u64) };
    try testing.expectError(OptimalBstError.Overflow, optimalBinarySearchTreeCost(testing.allocator, &keys, &freqs));
}
