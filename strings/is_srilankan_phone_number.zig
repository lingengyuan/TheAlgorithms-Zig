//! Sri Lankan Phone Number Validator - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/is_srilankan_phone_number.py

const std = @import("std");
const testing = std.testing;

/// Returns true if `phone` matches the Sri Lankan mobile formats accepted by the Python reference.
/// Time complexity: O(n), Space complexity: O(1)
pub fn isSriLankanPhoneNumber(phone: []const u8) bool {
    var rest = phone;
    if (std.mem.startsWith(u8, rest, "0094")) {
        rest = rest[4..];
    } else if (std.mem.startsWith(u8, rest, "+94")) {
        rest = rest[3..];
    } else if (std.mem.startsWith(u8, rest, "94")) {
        rest = rest[2..];
    } else if (std.mem.startsWith(u8, rest, "0")) {
        rest = rest[1..];
    } else {
        return false;
    }

    if (rest.len != 9 and rest.len != 10) return false;
    if (rest.len < 2 or rest[0] != '7') return false;

    switch (rest[1]) {
        '0', '1', '2', '4', '5', '6', '7', '8' => {},
        else => return false,
    }

    var index: usize = 2;
    if (rest.len == 10) {
        if (rest[2] != '-' and rest[2] != ' ') return false;
        index = 3;
    }
    if (rest.len - index != 7) return false;

    for (rest[index..]) |char| {
        if (!std.ascii.isDigit(char)) return false;
    }
    return true;
}

test "sri lankan phone validator: python samples" {
    try testing.expect(isSriLankanPhoneNumber("+94773283048"));
    try testing.expect(isSriLankanPhoneNumber("+9477-3283048"));
    try testing.expect(isSriLankanPhoneNumber("0718382399"));
    try testing.expect(isSriLankanPhoneNumber("0094702343221"));
    try testing.expect(isSriLankanPhoneNumber("075 3201568"));
    try testing.expect(!isSriLankanPhoneNumber("07779209245"));
    try testing.expect(!isSriLankanPhoneNumber("0957651234"));
}

test "sri lankan phone validator: edge cases" {
    try testing.expect(isSriLankanPhoneNumber("94773283048"));
    try testing.expect(!isSriLankanPhoneNumber("+9479 3283048"));
    try testing.expect(!isSriLankanPhoneNumber("07183"));
}
