//! Modular Exponential - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/modular_exponential.py

const std = @import("std");
const testing = std.testing;

pub const ModularExponentialError = error{InvalidModulus};

/// Computes `(base ^ power) mod mod`.
/// Returns -1 for negative powers to match the Python reference.
/// Time complexity: O(log power), Space complexity: O(1)
pub fn modularExponential(base: i64, power: i64, mod: i64) ModularExponentialError!i64 {
    if (mod <= 0) return error.InvalidModulus;
    if (power < 0) return -1;

    var result: i64 = 1;
    var current_base = @mod(base, mod);
    var current_power: u64 = @intCast(power);

    while (current_power > 0) {
        if (current_power & 1 == 1) {
            result = @intCast(@mod(@as(i128, result) * @as(i128, current_base), @as(i128, mod)));
        }
        current_power >>= 1;
        current_base = @intCast(@mod(@as(i128, current_base) * @as(i128, current_base), @as(i128, mod)));
    }
    return result;
}

test "modular exponential: python reference examples" {
    try testing.expectEqual(@as(i64, 1), try modularExponential(5, 0, 10));
    try testing.expectEqual(@as(i64, 4), try modularExponential(2, 8, 7));
    try testing.expectEqual(@as(i64, -1), try modularExponential(3, -2, 9));
}

test "modular exponential: edge and extreme cases" {
    try testing.expectEqual(@as(i64, 0), try modularExponential(12, 5, 1));
    try testing.expectEqual(@as(i64, 1), try modularExponential(0, 0, 13));
    try testing.expectEqual(@as(i64, 11), try modularExponential(-3, 3, 19));
    try testing.expectError(error.InvalidModulus, modularExponential(2, 3, 0));
}
