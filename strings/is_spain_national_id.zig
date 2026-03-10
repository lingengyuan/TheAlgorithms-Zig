//! Spain National ID Validator - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/is_spain_national_id.py

const std = @import("std");
const testing = std.testing;

pub const SpainNationalIdError = error{InvalidFormat};
const lookup_letters = "TRWAGMYFPDXBNJZSQVHLCKE";

/// Returns true if `spanish_id` is a valid DNI string under the Python reference rules.
/// Time complexity: O(n), Space complexity: O(1)
pub fn isSpainNationalId(spanish_id: []const u8) SpainNationalIdError!bool {
    var cleaned: [9]u8 = undefined;
    var len: usize = 0;
    for (spanish_id) |char| {
        if (char == '-') continue;
        if (len >= cleaned.len) return SpainNationalIdError.InvalidFormat;
        cleaned[len] = std.ascii.toUpper(char);
        len += 1;
    }
    if (len != 9) return SpainNationalIdError.InvalidFormat;

    var number: u32 = 0;
    for (cleaned[0..8]) |char| {
        if (!std.ascii.isDigit(char)) return SpainNationalIdError.InvalidFormat;
        number = number * 10 + (char - '0');
    }

    const letter = cleaned[8];
    if (std.ascii.isDigit(letter)) return SpainNationalIdError.InvalidFormat;
    return letter == lookup_letters[number % 23];
}

test "spain national id: python samples" {
    try testing.expect(try isSpainNationalId("12345678Z"));
    try testing.expect(try isSpainNationalId("12345678z"));
    try testing.expect(!(try isSpainNationalId("12345678x")));
    try testing.expect(!(try isSpainNationalId("12345678I")));
    try testing.expect(try isSpainNationalId("12345678-Z"));
}

test "spain national id: invalid format" {
    try testing.expectError(SpainNationalIdError.InvalidFormat, isSpainNationalId("12345678"));
    try testing.expectError(SpainNationalIdError.InvalidFormat, isSpainNationalId("123456709"));
    try testing.expectError(SpainNationalIdError.InvalidFormat, isSpainNationalId("1234567--Z"));
    try testing.expectError(SpainNationalIdError.InvalidFormat, isSpainNationalId("1234Z"));
    try testing.expectError(SpainNationalIdError.InvalidFormat, isSpainNationalId("1234ZzZZ"));
}
