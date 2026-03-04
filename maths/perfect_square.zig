//! Perfect Square - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/perfect_square.py

const std = @import("std");
const testing = std.testing;

/// Checks whether `num` is a perfect square.
/// Time complexity: O(log n), Space complexity: O(1)
pub fn perfectSquare(num: i64) bool {
    if (num < 0) return false;
    const value: u128 = @intCast(num);
    const root = integerSqrt(value);
    return root * root == value;
}

/// Checks whether `n` is a perfect square using binary search.
/// Time complexity: O(log n), Space complexity: O(1)
pub fn perfectSquareBinarySearch(n: i64) bool {
    if (n < 0) return false;
    const value: u128 = @intCast(n);

    var left: u128 = 0;
    var right: u128 = value;

    while (left <= right) {
        const mid = left + (right - left) / 2;
        const square = mid * mid;
        if (square == value) return true;
        if (square > value) {
            if (mid == 0) break;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return false;
}

fn integerSqrt(value: u128) u128 {
    if (value < 2) return value;

    var x = value;
    var y = (x + value / x) / 2;
    while (y < x) {
        x = y;
        y = (x + value / x) / 2;
    }
    return x;
}

test "perfect square: python reference examples" {
    try testing.expect(perfectSquare(9));
    try testing.expect(perfectSquare(16));
    try testing.expect(perfectSquare(1));
    try testing.expect(perfectSquare(0));
    try testing.expect(!perfectSquare(10));

    try testing.expect(perfectSquareBinarySearch(9));
    try testing.expect(perfectSquareBinarySearch(16));
    try testing.expect(perfectSquareBinarySearch(1));
    try testing.expect(perfectSquareBinarySearch(0));
    try testing.expect(!perfectSquareBinarySearch(10));
}

test "perfect square: edge and extreme cases" {
    const boundary_square: i64 = 9_223_372_030_926_249_001; // 3_037_000_499^2
    try testing.expect(perfectSquare(boundary_square));
    try testing.expect(perfectSquareBinarySearch(boundary_square));

    try testing.expect(!perfectSquareBinarySearch(-1));
    try testing.expect(!perfectSquare(std.math.maxInt(i64)));
    try testing.expect(!perfectSquareBinarySearch(std.math.maxInt(i64)));
}
