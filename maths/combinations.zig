//! Combinations - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/combinations.py

const std = @import("std");
const testing = std.testing;

pub const CombinationError = error{ InvalidInput, Overflow };

/// Returns nCk for `n >= k >= 0`.
/// Time complexity: O(k), Space complexity: O(1)
pub fn combinations(n: i64, k: i64) CombinationError!u128 {
    if (n < k or k < 0) return CombinationError.InvalidInput;

    var res: u128 = 1;
    var i: i64 = 0;
    while (i < k) : (i += 1) {
        const mul = @mulWithOverflow(res, @as(u128, @intCast(n - i)));
        if (mul[1] != 0) return CombinationError.Overflow;
        res = mul[0];
        res /= @as(u128, @intCast(i + 1));
    }
    return res;
}

test "combinations: python reference examples" {
    try testing.expectEqual(@as(u128, 252), try combinations(10, 5));
    try testing.expectEqual(@as(u128, 20), try combinations(6, 3));
    try testing.expectEqual(@as(u128, 15_504), try combinations(20, 5));
    try testing.expectEqual(@as(u128, 2_598_960), try combinations(52, 5));
    try testing.expectEqual(@as(u128, 1), try combinations(0, 0));
    try testing.expectError(CombinationError.InvalidInput, combinations(-4, -5));
}

test "combinations: edge and extreme cases" {
    try testing.expectEqual(@as(u128, 1), try combinations(100, 0));
    try testing.expectEqual(@as(u128, 100), try combinations(100, 1));
    try testing.expectEqual(@as(u128, 100), try combinations(100, 99));
    try testing.expectError(CombinationError.Overflow, combinations(200, 100));
}
