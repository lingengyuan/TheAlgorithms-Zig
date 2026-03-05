//! Power Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/power_sum.py

const std = @import("std");
const testing = std.testing;

pub const PowerSumError = error{InvalidInput};

fn powBounded(base: i32, exponent: i32, bound: i32) i32 {
    var result: i64 = 1;
    const base64: i64 = base;
    const bound64: i64 = bound;

    var i: i32 = 0;
    while (i < exponent) : (i += 1) {
        const mul = @mulWithOverflow(result, base64);
        if (mul[1] != 0) return bound + 1;
        result = mul[0];
        if (result > bound64) return bound + 1;
    }

    return @intCast(result);
}

fn countWays(needed_sum: i32, power: i32, current_number: i32, current_sum: i32) u32 {
    if (current_sum == needed_sum) return 1;

    const i_to_n = powBounded(current_number, power, needed_sum);
    var ways: u32 = 0;

    if (current_sum + i_to_n <= needed_sum) {
        ways += countWays(needed_sum, power, current_number + 1, current_sum + i_to_n);
    }

    if (i_to_n < needed_sum) {
        ways += countWays(needed_sum, power, current_number + 1, current_sum);
    }

    return ways;
}

/// Returns number of ways `needed_sum` can be written as sum of unique natural
/// numbers raised to `power`.
///
/// Time complexity: exponential (backtracking)
/// Space complexity: O(needed_sum) recursion depth in worst case
pub fn solve(needed_sum: i32, power: i32) PowerSumError!u32 {
    if (!(needed_sum >= 1 and needed_sum <= 1000 and power >= 2 and power <= 10)) {
        return PowerSumError.InvalidInput;
    }

    return countWays(needed_sum, power, 1, 0);
}

test "power sum: python examples" {
    try testing.expectEqual(@as(u32, 1), try solve(13, 2));
    try testing.expectEqual(@as(u32, 1), try solve(10, 2));
    try testing.expectEqual(@as(u32, 0), try solve(10, 3));
    try testing.expectEqual(@as(u32, 1), try solve(20, 2));
    try testing.expectEqual(@as(u32, 0), try solve(15, 10));
    try testing.expectEqual(@as(u32, 1), try solve(16, 2));
}

test "power sum: invalid inputs" {
    try testing.expectError(PowerSumError.InvalidInput, solve(20, 1));
    try testing.expectError(PowerSumError.InvalidInput, solve(-10, 5));
    try testing.expectError(PowerSumError.InvalidInput, solve(1001, 2));
    try testing.expectError(PowerSumError.InvalidInput, solve(1000, 11));
}

test "power sum: boundary extremes" {
    try testing.expectEqual(@as(u32, 1), try solve(1, 2));
    try testing.expectEqual(@as(u32, 1269), try solve(1000, 2));
    try testing.expectEqual(@as(u32, 0), try solve(1000, 10));
}
