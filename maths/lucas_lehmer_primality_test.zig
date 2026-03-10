//! Lucas-Lehmer Primality Test - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/lucas_lehmer_primality_test.py

const std = @import("std");
const testing = std.testing;

pub const LucasLehmerError = error{InvalidInput};

/// Returns true when `2^p - 1` is prime according to the Lucas-Lehmer test.
/// Time complexity: O(p), Space complexity: O(1)
pub fn lucasLehmerTest(p: i64) LucasLehmerError!bool {
    if (p < 2 or p >= 65) return error.InvalidInput;
    if (p == 2) return true;

    var s: u128 = 4;
    const shift: u7 = @intCast(p);
    const m: u128 = (@as(u128, 1) << shift) - 1;

    var i: i64 = 0;
    while (i < p - 2) : (i += 1) {
        s = (s * s - 2) % m;
    }
    return s == 0;
}

test "lucas lehmer: python reference examples" {
    try testing.expect(try lucasLehmerTest(7));
    try testing.expect(!(try lucasLehmerTest(11)));
}

test "lucas lehmer: edge cases" {
    try testing.expect(try lucasLehmerTest(2));
    try testing.expectError(error.InvalidInput, lucasLehmerTest(1));
    try testing.expectError(error.InvalidInput, lucasLehmerTest(65));
}
