//! Catalan Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/catalan_numbers.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Returns the nth Catalan number.
/// Uses DP recurrence: C(n) = sum(C(i) * C(n-1-i)).
/// Time complexity: O(n^2), Space complexity: O(n)
pub fn catalanNumber(allocator: Allocator, n: usize) !u64 {
    if (n == 0) return 1;

    const n_plus = @addWithOverflow(n, @as(usize, 1));
    if (n_plus[1] != 0) return error.Overflow;
    const dp = try allocator.alloc(u128, n_plus[0]);
    defer allocator.free(dp);
    @memset(dp, 0);
    dp[0] = 1;
    if (n >= 1) dp[1] = 1;

    var i: usize = 2;
    while (i <= n) : (i += 1) {
        var sum: u128 = 0;
        var j: usize = 0;
        while (j < i) : (j += 1) {
            const mul = @mulWithOverflow(dp[j], dp[i - 1 - j]);
            if (mul[1] != 0) return error.Overflow;
            const add = @addWithOverflow(sum, mul[0]);
            if (add[1] != 0) return error.Overflow;
            sum = add[0];
        }
        dp[i] = sum;
    }

    if (dp[n] > std.math.maxInt(u64)) return error.Overflow;
    return @intCast(dp[n]);
}

test "catalan: n=0" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 1), try catalanNumber(alloc, 0));
}

test "catalan: n=1" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 1), try catalanNumber(alloc, 1));
}

test "catalan: n=5" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 42), try catalanNumber(alloc, 5));
}

test "catalan: n=10" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 16_796), try catalanNumber(alloc, 10));
}

test "catalan: overflow for large n" {
    const alloc = testing.allocator;
    try testing.expectError(error.Overflow, catalanNumber(alloc, 37));
}

test "catalan: oversize n returns overflow" {
    try testing.expectError(error.Overflow, catalanNumber(testing.allocator, std.math.maxInt(usize)));
}
