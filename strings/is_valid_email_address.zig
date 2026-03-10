//! Email Address Validator - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/is_valid_email_address.py

const std = @import("std");
const testing = std.testing;

pub const MAX_LOCAL_PART_OCTETS: usize = 64;
pub const MAX_DOMAIN_OCTETS: usize = 255;

/// Returns true if `email` satisfies the same validation rules as the Python reference.
/// Time complexity: O(n), Space complexity: O(1)
pub fn isValidEmailAddress(email: []const u8) bool {
    var at_index: ?usize = null;
    for (email, 0..) |char, index| {
        if (char == '@') {
            if (at_index != null) return false;
            at_index = index;
        }
    }
    const split_index = at_index orelse return false;

    const local_part = email[0..split_index];
    const domain = email[split_index + 1 ..];
    if (local_part.len == 0 or domain.len == 0) return false;
    if (local_part.len > MAX_LOCAL_PART_OCTETS or domain.len > MAX_DOMAIN_OCTETS) return false;

    const local_allowed = ".(!#$%&'*+-/=?^_`{|}~)";
    for (local_part) |char| {
        if (!std.ascii.isAlphabetic(char) and !std.ascii.isDigit(char) and std.mem.indexOfScalar(u8, local_allowed, char) == null) {
            return false;
        }
    }
    if (std.mem.startsWith(u8, local_part, ".") or std.mem.endsWith(u8, local_part, ".") or std.mem.indexOf(u8, local_part, "..") != null) {
        return false;
    }

    for (domain) |char| {
        if (!std.ascii.isAlphabetic(char) and !std.ascii.isDigit(char) and char != '.' and char != '-') {
            return false;
        }
    }
    if (std.mem.startsWith(u8, domain, "-") or std.mem.endsWith(u8, domain, ".")) return false;
    if (std.mem.startsWith(u8, domain, ".") or std.mem.indexOf(u8, domain, "..") != null) return false;

    return true;
}

test "email validator: python samples valid" {
    const valid = [_][]const u8{
        "simple@example.com",
        "very.common@example.com",
        "disposable.style.email.with+symbol@example.com",
        "other-email-with-hyphen@and.subdomains.example.com",
        "fully-qualified-domain@example.com",
        "user.name+tag+sorting@example.com",
        "x@example.com",
        "example-indeed@strange-example.com",
        "test/test@test.com",
        "123456789012345678901234567890123456789012345678901234567890123@example.com",
        "admin@mailserver1",
        "example@s.example",
    };
    for (valid) |email| try testing.expect(isValidEmailAddress(email));
}

test "email validator: python samples invalid" {
    const invalid = [_][]const u8{
        "Abc.example.com",
        "A@b@c@example.com",
        "abc@example..com",
        "a(c)d,e:f;g<h>i[j\\k]l@example.com",
        "12345678901234567890123456789012345678901234567890123456789012345@example.com",
        "i.like.underscores@but_its_not_allowed_in_this_part",
        "",
    };
    for (invalid) |email| try testing.expect(!isValidEmailAddress(email));
}

test "email validator: edge and extreme" {
    try testing.expect(!isValidEmailAddress(".startdot@example.com"));
    try testing.expect(!isValidEmailAddress("enddot.@example.com"));
    try testing.expect(!isValidEmailAddress("user@.example.com"));
    try testing.expect(!isValidEmailAddress("user@example."));
    try testing.expect(!isValidEmailAddress("@domain.com"));
    try testing.expect(!isValidEmailAddress("user@"));
    try testing.expect(!isValidEmailAddress("@"));
}
