//! Project Euler Problem 17: Number Letter Counts - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_017/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem017Error = error{
    NumberOutOfRange,
};

const ones_counts = [20]u16{ 0, 3, 3, 5, 4, 4, 3, 5, 5, 4, 3, 6, 6, 8, 8, 7, 7, 9, 8, 8 };
const tens_counts = [10]u16{ 0, 0, 6, 6, 5, 5, 5, 7, 6, 6 };

/// Returns the number of letters used to write an integer in [1, 1000]
/// using British usage (includes "and", excludes spaces/hyphens).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn lettersInNumber(value: u16) Problem017Error!u16 {
    if (value == 0 or value > 1000) return Problem017Error.NumberOutOfRange;

    if (value == 1000) {
        // "one thousand" => 3 + 8
        return 11;
    }

    var count: u16 = 0;

    if (value >= 100) {
        const hundreds: usize = value / 100;
        count += ones_counts[hundreds] + 7; // "x hundred"

        if (value % 100 != 0) {
            count += 3; // "and"
        }
    }

    const suffix = value % 100;
    if (suffix > 0 and suffix < 20) {
        count += ones_counts[suffix];
    } else {
        const tens: usize = suffix / 10;
        const ones: usize = suffix % 10;
        count += tens_counts[tens] + ones_counts[ones];
    }

    return count;
}

/// Returns the total letters used to write all numbers in [1, n], n <= 1000.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn solution(n: u16) Problem017Error!u64 {
    if (n > 1000) return Problem017Error.NumberOutOfRange;

    var total: u64 = 0;
    for (1..@as(usize, n) + 1) |i| {
        total += try lettersInNumber(@intCast(i));
    }

    return total;
}

test "problem 017: python reference" {
    try testing.expectEqual(@as(u64, 21_124), try solution(1000));
    try testing.expectEqual(@as(u64, 19), try solution(5));
}

test "problem 017: known examples and boundaries" {
    try testing.expectEqual(@as(u16, 23), try lettersInNumber(342));
    try testing.expectEqual(@as(u16, 20), try lettersInNumber(115));

    try testing.expectEqual(@as(u64, 0), try solution(0));
    try testing.expectEqual(@as(u64, 3), try solution(1));
    try testing.expectEqual(@as(u64, 21_113), try solution(999));

    try testing.expectError(Problem017Error.NumberOutOfRange, solution(1001));
    try testing.expectError(Problem017Error.NumberOutOfRange, lettersInNumber(0));
}
