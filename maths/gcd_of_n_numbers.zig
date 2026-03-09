//! GCD Of N Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/gcd_of_n_numbers.py

const std = @import("std");
const testing = std.testing;
const gcd_mod = @import("gcd.zig");

pub const GcdOfNError = error{InvalidInput};

/// Returns the GCD of all positive integers in the slice.
/// Time complexity: O(n log m), Space complexity: O(1)
pub fn getGreatestCommonDivisor(numbers: []const i64) GcdOfNError!u64 {
    if (numbers.len == 0) return error.InvalidInput;
    for (numbers) |number| {
        if (number <= 0) return error.InvalidInput;
    }

    var result = gcd_mod.gcd(numbers[0], numbers[0]);
    for (numbers[1..]) |number| {
        result = gcd_mod.gcd(@intCast(result), number);
    }
    return result;
}

test "gcd of n numbers: python reference examples" {
    try testing.expectEqual(@as(u64, 9), try getGreatestCommonDivisor(&[_]i64{ 18, 45 }));
    try testing.expectEqual(@as(u64, 1), try getGreatestCommonDivisor(&[_]i64{ 23, 37 }));
    try testing.expectEqual(@as(u64, 10), try getGreatestCommonDivisor(&[_]i64{ 2520, 8350 }));
    try testing.expectEqual(@as(u64, 1), try getGreatestCommonDivisor(&[_]i64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }));
}

test "gcd of n numbers: edge cases" {
    try testing.expectError(error.InvalidInput, getGreatestCommonDivisor(&[_]i64{}));
    try testing.expectError(error.InvalidInput, getGreatestCommonDivisor(&[_]i64{ -10, 20 }));
    try testing.expectError(error.InvalidInput, getGreatestCommonDivisor(&[_]i64{ 1, 0, 3 }));
}
