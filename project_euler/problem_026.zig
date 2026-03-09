//! Project Euler Problem 26: Reciprocal Cycles - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_026/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem026Error = error{
    InvalidInput,
    OutOfMemory,
};

fn contains(list: []const u64, value: u64) bool {
    for (list) |item| {
        if (item == value) return true;
    }
    return false;
}

/// Mirrors the Python reference behavior for the `(numerator, digit)` search.
///
/// Time complexity: O((digit - numerator + 1) * digit^2)
/// Space complexity: O(digit)
pub fn solution(numerator: u64, digit: u64, allocator: std.mem.Allocator) Problem026Error!u64 {
    if (numerator == 0 or digit == 0) return Problem026Error.InvalidInput;

    var the_digit: u64 = 1;
    var longest_list_length: usize = 0;

    if (numerator > digit) return the_digit;

    var divide_by_number = numerator;
    while (divide_by_number <= digit) : (divide_by_number += 1) {
        var has_been_divided = std.ArrayListUnmanaged(u64){};
        defer has_been_divided.deinit(allocator);

        var now_divide: u64 = numerator;

        var iteration: u64 = 1;
        while (iteration <= digit) : (iteration += 1) {
            if (contains(has_been_divided.items, now_divide)) {
                if (longest_list_length < has_been_divided.items.len) {
                    longest_list_length = has_been_divided.items.len;
                    the_digit = divide_by_number;
                }
            } else {
                try has_been_divided.append(allocator, now_divide);
                now_divide = (now_divide * 10) % divide_by_number;
            }
        }
    }

    return the_digit;
}

test "problem 026: python reference" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 7), try solution(1, 10, allocator));
    try testing.expectEqual(@as(u64, 97), try solution(10, 100, allocator));
    try testing.expectEqual(@as(u64, 983), try solution(10, 1000, allocator));
}

test "problem 026: boundaries and python-compatible edge cases" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 1), try solution(1, 1, allocator));
    try testing.expectEqual(@as(u64, 1), try solution(1, 2, allocator));
    try testing.expectEqual(@as(u64, 1), try solution(1, 3, allocator));
    try testing.expectEqual(@as(u64, 100), try solution(100, 100, allocator));

    // Empty loop range in Python keeps default `the_digit = 1`.
    try testing.expectEqual(@as(u64, 1), try solution(200, 100, allocator));

    try testing.expectError(Problem026Error.InvalidInput, solution(0, 100, allocator));
    try testing.expectError(Problem026Error.InvalidInput, solution(1, 0, allocator));
}
