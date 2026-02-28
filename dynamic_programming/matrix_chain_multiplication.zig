//! Matrix Chain Multiplication - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/matrix_chain_multiplication.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Given matrix dimensions `dims` where matrix i has size dims[i] x dims[i+1],
/// returns the minimum scalar multiplication count to multiply the whole chain.
/// Time complexity: O(n^3), Space complexity: O(n^2)
pub fn matrixChainMultiplication(allocator: Allocator, dims: []const usize) !usize {
    if (dims.len <= 2) return 0; // 0 or 1 matrix needs no multiplication

    const n = dims.len - 1;
    const dp = try allocator.alloc(usize, n * n);
    defer allocator.free(dp);
    @memset(dp, 0);

    const inf = std.math.maxInt(usize);

    var chain_len: usize = 2;
    while (chain_len <= n) : (chain_len += 1) {
        var i: usize = 0;
        while (i + chain_len <= n) : (i += 1) {
            const j = i + chain_len - 1;
            dp[i * n + j] = inf;

            var k = i;
            while (k < j) : (k += 1) {
                const left = dp[i * n + k];
                const right = dp[(k + 1) * n + j];

                const mul_ab = @mulWithOverflow(dims[i], dims[k + 1]);
                if (mul_ab[1] != 0) continue;
                const mul_abc = @mulWithOverflow(mul_ab[0], dims[j + 1]);
                if (mul_abc[1] != 0) continue;

                const sum_lr = @addWithOverflow(left, right);
                if (sum_lr[1] != 0) continue;
                const cost = @addWithOverflow(sum_lr[0], mul_abc[0]);
                if (cost[1] != 0) continue;

                if (cost[0] < dp[i * n + j]) {
                    dp[i * n + j] = cost[0];
                }
            }
        }
    }

    if (dp[n - 1] == inf) return error.Overflow;
    return dp[n - 1];
}

test "matrix chain multiplication: classic example 1" {
    const alloc = testing.allocator;
    const dims = [_]usize{ 40, 20, 30, 10, 30 };
    try testing.expectEqual(@as(usize, 26000), try matrixChainMultiplication(alloc, &dims));
}

test "matrix chain multiplication: classic example 2" {
    const alloc = testing.allocator;
    const dims = [_]usize{ 10, 20, 30, 40, 30 };
    try testing.expectEqual(@as(usize, 30000), try matrixChainMultiplication(alloc, &dims));
}

test "matrix chain multiplication: two matrices" {
    const alloc = testing.allocator;
    const dims = [_]usize{ 10, 20, 30 };
    try testing.expectEqual(@as(usize, 6000), try matrixChainMultiplication(alloc, &dims));
}

test "matrix chain multiplication: one matrix" {
    const alloc = testing.allocator;
    const dims = [_]usize{ 10, 20 };
    try testing.expectEqual(@as(usize, 0), try matrixChainMultiplication(alloc, &dims));
}

test "matrix chain multiplication: empty dims" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 0), try matrixChainMultiplication(alloc, &[_]usize{}));
}
