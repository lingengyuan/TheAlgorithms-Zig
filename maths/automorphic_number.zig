//! Automorphic Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/special_numbers/automorphic_number.py

const std = @import("std");
const testing = std.testing;

/// Returns true if `number` is automorphic (`number^2` ends with digits of `number`).
/// Time complexity: O(d), Space complexity: O(1), where d is number of digits.
pub fn isAutomorphicNumber(number: i64) bool {
    if (number < 0) return false;

    var n: u128 = @intCast(number);
    var square = n * n;

    while (n > 0) {
        if ((n % 10) != (square % 10)) return false;
        n /= 10;
        square /= 10;
    }
    return true;
}

test "automorphic number: python reference examples" {
    try testing.expect(!isAutomorphicNumber(-1));
    try testing.expect(isAutomorphicNumber(0));
    try testing.expect(isAutomorphicNumber(5));
    try testing.expect(isAutomorphicNumber(6));
    try testing.expect(!isAutomorphicNumber(7));
    try testing.expect(isAutomorphicNumber(25));
    try testing.expect(isAutomorphicNumber(259_918_212_890_625));
    try testing.expect(!isAutomorphicNumber(259_918_212_890_636));
    try testing.expect(isAutomorphicNumber(740_081_787_109_376));
}

test "automorphic number: edge and extreme cases" {
    try testing.expect(isAutomorphicNumber(1));
    try testing.expect(!isAutomorphicNumber(2));
    try testing.expect(!isAutomorphicNumber(std.math.maxInt(i64)));
}
