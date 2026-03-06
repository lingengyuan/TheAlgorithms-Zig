//! Luhn Algorithm - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/hashes/luhn.py

const std = @import("std");
const testing = std.testing;

pub const LuhnError = error{
    EmptyInput,
    InvalidCharacter,
};

/// Performs Luhn checksum validation.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn isLuhn(string: []const u8) LuhnError!bool {
    if (string.len == 0) {
        return LuhnError.EmptyInput;
    }

    var check_digit: u32 = 0;

    const last = string[string.len - 1];
    if (last < '0' or last > '9') {
        return LuhnError.InvalidCharacter;
    }
    check_digit = last - '0';

    var i: usize = 0;
    var pos = string.len - 1;
    while (pos > 0) {
        pos -= 1;
        const ch = string[pos];
        if (ch < '0' or ch > '9') {
            return LuhnError.InvalidCharacter;
        }
        const digit: u32 = ch - '0';
        if ((i & 1) == 0) {
            var doubled = digit * 2;
            if (doubled > 9) doubled -= 9;
            check_digit += doubled;
        } else {
            check_digit += digit;
        }
        i += 1;
    }

    return (check_digit % 10) == 0;
}

test "luhn: python examples" {
    const test_cases = [_][]const u8{ "79927398710", "79927398711", "79927398712", "79927398713", "79927398714", "79927398715", "79927398716", "79927398717", "79927398718", "79927398719" };
    const expected = [_]bool{ false, false, false, true, false, false, false, false, false, false };

    for (test_cases, 0..) |test_case, idx| {
        try testing.expectEqual(expected[idx], try isLuhn(test_case));
    }
}

test "luhn: validation and extreme values" {
    try testing.expectError(LuhnError.EmptyInput, isLuhn(""));
    try testing.expectError(LuhnError.InvalidCharacter, isLuhn("79927A98713"));

    const long_valid = "4" ** 50_000;
    _ = try isLuhn(long_valid);
}
