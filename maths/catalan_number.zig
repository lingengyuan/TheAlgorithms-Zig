//! Catalan Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/special_numbers/catalan_number.py

const std = @import("std");
const testing = std.testing;

pub const CatalanError = error{ InvalidInput, Overflow };

/// Returns the nth Catalan number for `number >= 1`.
/// Time complexity: O(n), Space complexity: O(1)
pub fn catalan(number: i64) CatalanError!u128 {
    if (number < 1) return CatalanError.InvalidInput;

    var current: u128 = 1;
    var i: u128 = 1;
    while (i < @as(u128, @intCast(number))) : (i += 1) {
        const term_mul = @subWithOverflow(@as(u128, 4) * i, @as(u128, 2));
        if (term_mul[1] != 0) return CatalanError.Overflow;

        const mul = @mulWithOverflow(current, term_mul[0]);
        if (mul[1] != 0) return CatalanError.Overflow;

        const divisor = i + 1;
        current = mul[0] / divisor;
    }

    return current;
}

test "catalan number: python reference examples" {
    try testing.expectEqual(@as(u128, 14), try catalan(5));
    try testing.expectError(CatalanError.InvalidInput, catalan(0));
    try testing.expectError(CatalanError.InvalidInput, catalan(-1));
}

test "catalan number: known sequence prefix" {
    const expected = [_]u128{ 1, 1, 2, 5, 14, 42, 132, 429, 1_430, 4_862 };
    var i: usize = 0;
    while (i < expected.len) : (i += 1) {
        try testing.expectEqual(expected[i], try catalan(@intCast(i + 1)));
    }
}

test "catalan number: overflow on very large n" {
    try testing.expectError(CatalanError.Overflow, catalan(100));
}
