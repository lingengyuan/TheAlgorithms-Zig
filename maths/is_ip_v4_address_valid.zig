//! IPv4 Address Validation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/is_ip_v4_address_valid.py

const std = @import("std");
const testing = std.testing;

/// Returns true if the input is a valid dotted IPv4 address.
/// Time complexity: O(n), Space complexity: O(1)
pub fn isIpV4AddressValid(ip: []const u8) bool {
    var octet_count: usize = 0;
    var it = std.mem.splitScalar(u8, ip, '.');
    while (it.next()) |octet| {
        octet_count += 1;
        if (octet.len == 0) return false;
        for (octet) |ch| {
            if (!std.ascii.isDigit(ch)) return false;
        }
        if (octet.len > 1 and octet[0] == '0') return false;
        const number = std.fmt.parseInt(u16, octet, 10) catch return false;
        if (number > 255) return false;
    }
    return octet_count == 4;
}

test "ipv4 validation: python reference examples" {
    try testing.expect(isIpV4AddressValid("192.168.0.23"));
    try testing.expect(!isIpV4AddressValid("192.256.15.8"));
    try testing.expect(isIpV4AddressValid("172.100.0.8"));
    try testing.expect(!isIpV4AddressValid("255.256.0.256"));
    try testing.expect(!isIpV4AddressValid("1.2.33333333.4"));
    try testing.expect(!isIpV4AddressValid("1.2.-3.4"));
}

test "ipv4 validation: edge and extreme cases" {
    try testing.expect(!isIpV4AddressValid("1.2.3"));
    try testing.expect(!isIpV4AddressValid("1.2.3.4.5"));
    try testing.expect(!isIpV4AddressValid("1.2.A.4"));
    try testing.expect(isIpV4AddressValid("0.0.0.0"));
    try testing.expect(!isIpV4AddressValid("1.2.3."));
    try testing.expect(!isIpV4AddressValid("1.2.3.05"));
    try testing.expect(!isIpV4AddressValid(""));
}
