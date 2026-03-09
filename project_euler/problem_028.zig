//! Project Euler Problem 28: Number Spiral Diagonals - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_028/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem028Error = error{
    Overflow,
};

fn checkedMul(a: i128, b: i128) Problem028Error!i128 {
    const result = @mulWithOverflow(a, b);
    if (result[1] != 0) return Problem028Error.Overflow;
    return result[0];
}

fn checkedAdd(a: i128, b: i128) Problem028Error!i128 {
    const result = @addWithOverflow(a, b);
    if (result[1] != 0) return Problem028Error.Overflow;
    return result[0];
}

/// Returns diagonal sum for n x n spiral, mirroring Python reference behavior.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn solution(n: i64) Problem028Error!i128 {
    var total: i128 = 1;

    const ceil_half: i64 = if (n > 0) @divFloor(n + 1, 2) else 0;

    var i: i64 = 1;
    while (i < ceil_half) : (i += 1) {
        const odd: i128 = @as(i128, 2 * i + 1);
        const even: i128 = @as(i128, 2 * i);

        const odd_sq = try checkedMul(odd, odd);
        const term_left = try checkedMul(4, odd_sq);
        const term_right = try checkedMul(6, even);
        const term = try checkedAdd(term_left, -term_right);

        total = try checkedAdd(total, term);
    }

    return total;
}

test "problem 028: python reference" {
    try testing.expectEqual(@as(i128, 669_171_001), try solution(1001));
    try testing.expectEqual(@as(i128, 82_959_497), try solution(500));
    try testing.expectEqual(@as(i128, 651_897), try solution(100));
    try testing.expectEqual(@as(i128, 79_697), try solution(50));
    try testing.expectEqual(@as(i128, 537), try solution(10));
}

test "problem 028: boundaries and odd/even behavior" {
    try testing.expectEqual(@as(i128, 101), try solution(5));
    try testing.expectEqual(@as(i128, 25), try solution(4));
    try testing.expectEqual(@as(i128, 25), try solution(3));
    try testing.expectEqual(@as(i128, 1), try solution(2));
    try testing.expectEqual(@as(i128, 1), try solution(1));
    try testing.expectEqual(@as(i128, 1), try solution(0));
    try testing.expectEqual(@as(i128, 1), try solution(-1));
    try testing.expectEqual(@as(i128, 1), try solution(-10));
}
