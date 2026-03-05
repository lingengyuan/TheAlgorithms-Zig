//! IPv4 Conversion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/conversions/ipv4_conversion.py

const std = @import("std");
const testing = std.testing;

pub const Ipv4ConversionError = error{ InvalidFormat, InvalidOctet, InvalidDecimal };

/// Converts dotted IPv4 string to decimal representation.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn ipv4ToDecimal(ipv4_address: []const u8) Ipv4ConversionError!u32 {
    var split = std.mem.splitScalar(u8, ipv4_address, '.');

    var decimal_ipv4: u32 = 0;
    var count: usize = 0;

    while (split.next()) |raw_part| {
        if (count >= 4) return Ipv4ConversionError.InvalidFormat;

        const part = std.mem.trim(u8, raw_part, " \t\r\n");
        if (part.len == 0) return Ipv4ConversionError.InvalidFormat;

        const octet = std.fmt.parseInt(u16, part, 10) catch return Ipv4ConversionError.InvalidFormat;
        if (octet > 255) return Ipv4ConversionError.InvalidOctet;

        decimal_ipv4 = (decimal_ipv4 << 8) + @as(u32, octet);
        count += 1;
    }

    if (count != 4) return Ipv4ConversionError.InvalidFormat;
    return decimal_ipv4;
}

/// Alternate form kept for parity with Python reference.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn altIpv4ToDecimal(ipv4_address: []const u8) Ipv4ConversionError!u32 {
    return ipv4ToDecimal(ipv4_address);
}

/// Converts decimal IPv4 representation to dotted string.
///
/// Time complexity: O(1)
/// Space complexity: O(1), excluding returned string allocation.
pub fn decimalToIpv4(
    allocator: std.mem.Allocator,
    decimal_ipv4: i64,
) (Ipv4ConversionError || std.mem.Allocator.Error)![]u8 {
    if (decimal_ipv4 < 0 or decimal_ipv4 > 4_294_967_295) {
        return Ipv4ConversionError.InvalidDecimal;
    }

    const value: u32 = @intCast(decimal_ipv4);
    const a = (value >> 24) & 0xff;
    const b = (value >> 16) & 0xff;
    const c = (value >> 8) & 0xff;
    const d = value & 0xff;

    return std.fmt.allocPrint(allocator, "{d}.{d}.{d}.{d}", .{ a, b, c, d });
}

test "ipv4 conversion: python examples" {
    try testing.expectEqual(@as(u32, 3_232_235_521), try ipv4ToDecimal("192.168.0.1"));
    try testing.expectEqual(@as(u32, 167_772_415), try ipv4ToDecimal("10.0.0.255"));
    try testing.expectEqual(@as(u32, 3_232_235_521), try altIpv4ToDecimal("192.168.0.1"));

    const alloc = testing.allocator;
    const s1 = try decimalToIpv4(alloc, 3_232_235_521);
    defer alloc.free(s1);
    try testing.expectEqualStrings("192.168.0.1", s1);

    const s2 = try decimalToIpv4(alloc, 167_772_415);
    defer alloc.free(s2);
    try testing.expectEqualStrings("10.0.0.255", s2);
}

test "ipv4 conversion: validation" {
    const alloc = testing.allocator;

    try testing.expectError(Ipv4ConversionError.InvalidFormat, ipv4ToDecimal("10.0.255"));
    try testing.expectError(Ipv4ConversionError.InvalidOctet, ipv4ToDecimal("10.0.0.256"));
    try testing.expectError(Ipv4ConversionError.InvalidFormat, ipv4ToDecimal("10.0.0.a"));
    try testing.expectError(Ipv4ConversionError.InvalidDecimal, decimalToIpv4(alloc, -1));
}

test "ipv4 conversion: extreme boundaries" {
    const alloc = testing.allocator;

    try testing.expectEqual(@as(u32, 0), try ipv4ToDecimal("0.0.0.0"));
    try testing.expectEqual(@as(u32, 4_294_967_295), try ipv4ToDecimal("255.255.255.255"));

    const min_ip = try decimalToIpv4(alloc, 0);
    defer alloc.free(min_ip);
    try testing.expectEqualStrings("0.0.0.0", min_ip);

    const max_ip = try decimalToIpv4(alloc, 4_294_967_295);
    defer alloc.free(max_ip);
    try testing.expectEqualStrings("255.255.255.255", max_ip);
}
