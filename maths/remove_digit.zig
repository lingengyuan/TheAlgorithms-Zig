//! Remove Digit - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/remove_digit.py

const std = @import("std");
const testing = std.testing;

pub const RemoveDigitError = error{ InvalidInput, Overflow };

/// Returns maximum number obtainable by removing one digit from absolute value of `num`.
/// Time complexity: O(d^2), Space complexity: O(d), where d is number of digits.
pub fn removeDigit(num: i64) RemoveDigitError!u64 {
    const abs_num = absAsU128(num);

    var buf: [64]u8 = undefined;
    const repr = std.fmt.bufPrint(&buf, "{d}", .{abs_num}) catch return RemoveDigitError.Overflow;
    if (repr.len <= 1) return RemoveDigitError.InvalidInput;

    var max_value: u64 = 0;
    var has_candidate = false;

    var i: usize = 0;
    while (i < repr.len) : (i += 1) {
        var candidate_buf: [64]u8 = undefined;
        var idx: usize = 0;

        for (repr, 0..) |ch, j| {
            if (j == i) continue;
            candidate_buf[idx] = ch;
            idx += 1;
        }

        const candidate = std.fmt.parseInt(u64, candidate_buf[0..idx], 10) catch return RemoveDigitError.Overflow;
        if (!has_candidate or candidate > max_value) {
            max_value = candidate;
            has_candidate = true;
        }
    }

    return max_value;
}

fn absAsU128(value: i64) u128 {
    const widened: i128 = value;
    return if (widened < 0) @intCast(-widened) else @intCast(widened);
}

test "remove digit: python reference examples" {
    try testing.expectEqual(@as(u64, 52), try removeDigit(152));
    try testing.expectEqual(@as(u64, 685), try removeDigit(6385));
    try testing.expectEqual(@as(u64, 1), try removeDigit(-11));
    try testing.expectEqual(@as(u64, 222_222), try removeDigit(2_222_222));
}

test "remove digit: edge and extreme cases" {
    try testing.expectError(RemoveDigitError.InvalidInput, removeDigit(0));
    try testing.expectError(RemoveDigitError.InvalidInput, removeDigit(5));
    try testing.expectEqual(@as(u64, 1), try removeDigit(10));
    try testing.expectEqual(@as(u64, 11), try removeDigit(101));
    try testing.expectEqual(@as(u64, 923_372_036_854_775_807), try removeDigit(std.math.maxInt(i64)));
}
