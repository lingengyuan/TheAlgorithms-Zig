//! Polish National ID Validator - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/is_polish_national_id.py

const std = @import("std");
const testing = std.testing;

pub const PolishNationalIdError = error{ExpectedNumber};

/// Returns true if `input_str` is a valid PESEL number under the Python reference rules.
/// Time complexity: O(n), Space complexity: O(1)
pub fn isPolishNationalId(input_str: []const u8) PolishNationalIdError!bool {
    if (input_str.len == 0) return PolishNationalIdError.ExpectedNumber;

    var value: u64 = 0;
    for (input_str) |char| {
        if (!std.ascii.isDigit(char)) return PolishNationalIdError.ExpectedNumber;
        value = value * 10 + (char - '0');
    }

    if (value < 10_100_000 or value > 99_923_199_999) return false;
    if (input_str.len != 11) return false;

    const month = try parseTwoDigits(input_str[2], input_str[3]);
    const valid_month = (month >= 1 and month <= 12) or
        (month >= 21 and month <= 32) or
        (month >= 41 and month <= 52) or
        (month >= 61 and month <= 72) or
        (month >= 81 and month <= 92);
    if (!valid_month) return false;

    const day = try parseTwoDigits(input_str[4], input_str[5]);
    if (day < 1 or day > 31) return false;

    const multipliers = [_]u8{ 1, 3, 7, 9, 1, 3, 7, 9, 1, 3 };
    var subtotal: u32 = 0;
    for (multipliers, 0..) |multiplier, index| {
        subtotal += ((input_str[index] - '0') * multiplier) % 10;
    }
    const checksum: u8 = @intCast((10 - (subtotal % 10)) % 10);
    return checksum == input_str[10] - '0';
}

fn parseTwoDigits(a: u8, b: u8) PolishNationalIdError!u8 {
    if (!std.ascii.isDigit(a) or !std.ascii.isDigit(b)) return PolishNationalIdError.ExpectedNumber;
    return (a - '0') * 10 + (b - '0');
}

test "polish national id: python samples" {
    try testing.expectError(PolishNationalIdError.ExpectedNumber, isPolishNationalId("abc"));
    try testing.expect(try isPolishNationalId("02070803628"));
    try testing.expect(!(try isPolishNationalId("02150803629")));
    try testing.expect(!(try isPolishNationalId("02075503622")));
    try testing.expect(!(try isPolishNationalId("990122123499999")));
    try testing.expect(!(try isPolishNationalId("02070803621")));
}

test "polish national id: edge and extreme" {
    try testing.expect(!(try isPolishNationalId("-99012212349"[1..])));
    try testing.expect(!(try isPolishNationalId("00000000000")));
    try testing.expectError(PolishNationalIdError.ExpectedNumber, isPolishNationalId("02A70803628"));
    try testing.expect(try isPolishNationalId("80010100000"));
}
