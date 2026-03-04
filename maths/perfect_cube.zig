//! Perfect Cube - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/perfect_cube.py

const std = @import("std");
const testing = std.testing;

/// Checks whether `n` is a perfect cube.
/// Time complexity: O(log n), Space complexity: O(1)
pub fn perfectCube(n: i64) bool {
    return isPerfectCubeAbs(absAsU128(n));
}

/// Checks whether `n` is a perfect cube using binary search.
/// Time complexity: O(log n), Space complexity: O(1)
pub fn perfectCubeBinarySearch(n: i64) bool {
    return isPerfectCubeAbs(absAsU128(n));
}

fn isPerfectCubeAbs(value: u128) bool {
    var left: u128 = 0;
    var right: u128 = value;

    while (left <= right) {
        const mid = left + (right - left) / 2;
        if (mid == 0) {
            if (value == 0) return true;
            left = 1;
            continue;
        }

        const square = @mulWithOverflow(mid, mid);
        if (square[1] != 0 or square[0] > value / mid) {
            if (mid == 0) break;
            right = mid - 1;
            continue;
        }

        const cube = square[0] * mid;
        if (cube == value) return true;
        if (cube > value) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return false;
}

fn absAsU128(n: i64) u128 {
    const extended: i128 = n;
    if (extended >= 0) return @intCast(extended);
    return @intCast(-extended);
}

test "perfect cube: python reference examples" {
    try testing.expect(perfectCube(27));
    try testing.expect(!perfectCube(4));

    try testing.expect(perfectCubeBinarySearch(27));
    try testing.expect(perfectCubeBinarySearch(64));
    try testing.expect(!perfectCubeBinarySearch(4));
}

test "perfect cube: edge and extreme cases" {
    const boundary_cube: i64 = 9_223_358_842_721_533_951; // 2_097_151^3
    try testing.expect(perfectCube(boundary_cube));
    try testing.expect(perfectCubeBinarySearch(boundary_cube));

    try testing.expect(perfectCube(-27));
    try testing.expect(perfectCubeBinarySearch(-27));
    try testing.expect(!perfectCube(-4));
    try testing.expect(perfectCube(0));
    try testing.expect(!perfectCube(std.math.maxInt(i64)));
}
