//! Project Euler Problem 206: Concealed Square - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_206/sol1.py

const std = @import("std");
const testing = std.testing;

pub fn isSquareForm(num: u64) bool {
    var value = num;
    var digit: u64 = 9;
    while (value > 0) : (digit -= 1) {
        if (value % 10 != digit) return false;
        value /= 100;
    }
    return digit == 0;
}

/// Returns the unique positive integer whose square has the form 1_2_3_4_5_6_7_8_9_0.
/// Time complexity: narrow brute-force search over candidates ending in 3 or 7
/// Space complexity: O(1)
pub fn solution() u64 {
    var num: u64 = 138_902_663;
    while (!isSquareForm(num * num)) {
        if (num % 10 == 3) {
            num -= 6;
        } else {
            num -= 4;
        }
    }
    return num * 10;
}

test "problem 206: square form helper" {
    try testing.expect(!isSquareForm(1));
    try testing.expect(isSquareForm(112233445566778899));
    try testing.expect(!isSquareForm(123456789012345678));
}

test "problem 206: python reference" {
    try testing.expectEqual(@as(u64, 1389019170), solution());
}
