//! Factorial (DP memoization variant) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/factorial.py

const std = @import("std");
const testing = std.testing;
pub const FactorialError = error{ NegativeInput, Overflow };

/// Returns `n!`.
/// Zig uses an explicit error for negative input instead of Python's ValueError.
/// Time complexity: O(n), Space complexity: O(1)
pub fn factorial(n: i32) FactorialError!u64 {
    if (n < 0) return error.NegativeInput;

    var result: u64 = 1;
    var i: u64 = 2;
    while (i <= @as(u64, @intCast(n))) : (i += 1) {
        const product = @mulWithOverflow(result, i);
        if (product[1] != 0) return error.Overflow;
        result = product[0];
    }
    return result;
}

test "dynamic programming factorial: python examples" {
    try testing.expectEqual(@as(u64, 5040), try factorial(7));
    try testing.expectError(error.NegativeInput, factorial(-1));
}

test "dynamic programming factorial: range values" {
    const expected = [_]u64{ 1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880 };
    for (expected, 0..) |value, index| {
        try testing.expectEqual(value, try factorial(@intCast(index)));
    }
}

test "dynamic programming factorial: extreme overflow" {
    try testing.expectError(error.Overflow, factorial(21));
}
