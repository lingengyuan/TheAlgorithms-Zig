//! Matrix Chain Order (Cost + Split Tables) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/matrix_chain_order.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const MatrixChainOrderError = error{
    Overflow,
};

pub const MatrixChainOrderResult = struct {
    costs: []u64,
    splits: []usize,
    n: usize,

    pub fn deinit(self: MatrixChainOrderResult, allocator: Allocator) void {
        allocator.free(self.costs);
        allocator.free(self.splits);
    }

    pub fn costAt(self: MatrixChainOrderResult, row: usize, col: usize) u64 {
        return self.costs[row * self.n + col];
    }

    pub fn splitAt(self: MatrixChainOrderResult, row: usize, col: usize) usize {
        return self.splits[row * self.n + col];
    }
};

/// Builds DP cost and split tables for matrix-chain parenthesization.
/// `dims` are matrix dimensions where matrix i has shape dims[i-1] x dims[i]
/// in the 1-based indexing used by the Python implementation.
/// Time complexity: O(n^3), Space complexity: O(n^2)
pub fn matrixChainOrder(
    allocator: Allocator,
    dims: []const u64,
) (MatrixChainOrderError || Allocator.Error)!MatrixChainOrderResult {
    const n = dims.len;
    const cells = @mulWithOverflow(n, n);
    if (cells[1] != 0) return MatrixChainOrderError.Overflow;

    const costs = try allocator.alloc(u64, cells[0]);
    errdefer allocator.free(costs);
    const splits = try allocator.alloc(usize, cells[0]);
    errdefer allocator.free(splits);

    @memset(costs, 0);
    @memset(splits, 0);

    const inf = std.math.maxInt(u64);

    var chain_len: usize = 2;
    while (chain_len < n) : (chain_len += 1) {
        var a: usize = 1;
        const a_end_exclusive = n - chain_len + 1;
        while (a < a_end_exclusive) : (a += 1) {
            const b = a + chain_len - 1;
            costs[a * n + b] = inf;

            var c = a;
            while (c < b) : (c += 1) {
                const left = costs[a * n + c];
                const right = costs[(c + 1) * n + b];

                const m1 = @mulWithOverflow(dims[a - 1], dims[c]);
                if (m1[1] != 0) return MatrixChainOrderError.Overflow;
                const m2 = @mulWithOverflow(m1[0], dims[b]);
                if (m2[1] != 0) return MatrixChainOrderError.Overflow;

                const s1 = @addWithOverflow(left, right);
                if (s1[1] != 0) return MatrixChainOrderError.Overflow;
                const total = @addWithOverflow(s1[0], m2[0]);
                if (total[1] != 0) return MatrixChainOrderError.Overflow;

                if (total[0] < costs[a * n + b]) {
                    costs[a * n + b] = total[0];
                    splits[a * n + b] = c;
                }
            }
        }
    }

    return .{ .costs = costs, .splits = splits, .n = n };
}

test "matrix chain order: python example" {
    const dims = [_]u64{ 10, 30, 5 };
    const result = try matrixChainOrder(testing.allocator, &dims);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), result.n);
    try testing.expectEqual(@as(u64, 1500), result.costAt(1, 2));
    try testing.expectEqual(@as(usize, 1), result.splitAt(1, 2));
}

test "matrix chain order: classic CLRS sample final cost" {
    const dims = [_]u64{ 30, 35, 15, 5, 10, 20, 25 };
    const result = try matrixChainOrder(testing.allocator, &dims);
    defer result.deinit(testing.allocator);
    try testing.expectEqual(@as(u64, 15125), result.costAt(1, 6));
}

test "matrix chain order: boundary inputs" {
    const empty = [_]u64{};
    const r1 = try matrixChainOrder(testing.allocator, &empty);
    defer r1.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), r1.n);
    try testing.expectEqual(@as(usize, 0), r1.costs.len);
    try testing.expectEqual(@as(usize, 0), r1.splits.len);

    const one = [_]u64{10};
    const r2 = try matrixChainOrder(testing.allocator, &one);
    defer r2.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), r2.n);
    try testing.expectEqual(@as(usize, 1), r2.costs.len);
    try testing.expectEqual(@as(usize, 1), r2.splits.len);
}

test "matrix chain order: extreme overflow detection" {
    const dims = [_]u64{ std.math.maxInt(u64), std.math.maxInt(u64), std.math.maxInt(u64) };
    try testing.expectError(MatrixChainOrderError.Overflow, matrixChainOrder(testing.allocator, &dims));
}
