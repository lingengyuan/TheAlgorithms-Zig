//! Indian Phone Validator - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/indian_phone_validator.py

const std = @import("std");
const testing = std.testing;

/// Returns true if `phone` matches the Indian mobile formats accepted by the Python reference.
/// Time complexity: O(n), Space complexity: O(1)
pub fn indianPhoneValidator(phone: []const u8) bool {
    var rest = phone;

    if (std.mem.startsWith(u8, rest, "+91")) {
        rest = rest[3..];
        if (rest.len > 0 and (rest[0] == '-' or rest[0] == ' ')) rest = rest[1..];
    }

    if (rest.len > 0 and rest[0] == '0') rest = rest[1..];
    if (std.mem.startsWith(u8, rest, "91")) rest = rest[2..];

    if (rest.len != 10) return false;
    if (rest[0] != '7' and rest[0] != '8' and rest[0] != '9') return false;
    for (rest) |char| {
        if (!std.ascii.isDigit(char)) return false;
    }
    return true;
}

test "indian phone validator: python samples" {
    try testing.expect(!indianPhoneValidator("+91123456789"));
    try testing.expect(indianPhoneValidator("+919876543210"));
    try testing.expect(!indianPhoneValidator("01234567896"));
    try testing.expect(indianPhoneValidator("919876543218"));
    try testing.expect(!indianPhoneValidator("+91-1234567899"));
    try testing.expect(indianPhoneValidator("+91-9876543218"));
}

test "indian phone validator: edge and extreme" {
    try testing.expect(indianPhoneValidator("09876543210"));
    try testing.expect(indianPhoneValidator("9876543210"));
    try testing.expect(!indianPhoneValidator("+91 6876543210"));
    try testing.expect(!indianPhoneValidator("+91-987654321"));
}
