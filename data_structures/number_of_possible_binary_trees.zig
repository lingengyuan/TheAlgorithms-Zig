//! Number Of Possible Binary Trees - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/number_of_possible_binary_trees.py

const std = @import("std");
const testing = std.testing;

/// Computes C(n, k).
/// Time complexity: O(min(k, n-k)), Space complexity: O(1)
pub fn binomialCoefficient(n: u64, k_in: u64) !u128 {
    if (k_in > n) return error.InvalidK;

    var result: u128 = 1;
    const k = @min(k_in, n - k_in);

    var i: u64 = 0;
    while (i < k) : (i += 1) {
        const numerator: u128 = n - i;
        const mul = @mulWithOverflow(result, numerator);
        if (mul[1] != 0) return error.Overflow;
        result = mul[0] / @as(u128, i + 1);
    }

    return result;
}

/// Returns n-th Catalan number.
/// Time complexity: O(n), Space complexity: O(1)
pub fn catalanNumber(node_count: u64) !u128 {
    const c = try binomialCoefficient(node_count * 2, node_count);
    return c / @as(u128, node_count + 1);
}

/// Returns factorial(n).
/// Time complexity: O(n), Space complexity: O(1)
pub fn factorial(n: i64) !u128 {
    if (n < 0) return error.NegativeValue;

    var result: u128 = 1;
    var i: i64 = 1;
    while (i <= n) : (i += 1) {
        const mul = @mulWithOverflow(result, @as(u128, @intCast(i)));
        if (mul[1] != 0) return error.Overflow;
        result = mul[0];
    }

    return result;
}

/// Returns number of possible binary trees with `node_count` nodes.
/// Time complexity: O(n), Space complexity: O(1)
pub fn binaryTreeCount(node_count: u64) !u128 {
    const catalan = try catalanNumber(node_count);
    const fact = try factorial(@intCast(node_count));

    const mul = @mulWithOverflow(catalan, fact);
    if (mul[1] != 0) return error.Overflow;
    return mul[0];
}

test "number of possible binary trees: python doctest samples" {
    try testing.expectEqual(@as(u128, 6), try binomialCoefficient(4, 2));

    try testing.expectEqual(@as(u128, 42), try catalanNumber(5));
    try testing.expectEqual(@as(u128, 132), try catalanNumber(6));

    try testing.expectEqual(@as(u128, 5040), try binaryTreeCount(5));
    try testing.expectEqual(@as(u128, 95040), try binaryTreeCount(6));
}

test "number of possible binary trees: factorial behavior" {
    try testing.expectError(error.NegativeValue, factorial(-5));

    const known = [_]u128{ 1, 1, 2, 6, 24, 120, 720, 5040, 40_320, 362_880, 3_628_800 };
    for (known, 0..) |v, i| {
        try testing.expectEqual(v, try factorial(@intCast(i)));
    }
}

test "number of possible binary trees: extreme monotonic" {
    var prev_catalan: u128 = 0;
    var prev_count: u128 = 0;

    var n: u64 = 0;
    while (n <= 20) : (n += 1) {
        const c = try catalanNumber(n);
        const b = try binaryTreeCount(n);

        if (n > 1) {
            try testing.expect(c > prev_catalan);
            try testing.expect(b > prev_count);
        }

        prev_catalan = c;
        prev_count = b;
    }
}
