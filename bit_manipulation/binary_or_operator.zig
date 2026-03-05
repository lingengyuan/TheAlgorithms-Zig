//! Binary OR Operator - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/binary_or_operator.py

const std = @import("std");
const testing = std.testing;

pub const BinaryOrError = error{NegativeValue};

fn bitLengthU64(value: u64) usize {
    return if (value == 0) 1 else 64 - @clz(value);
}

/// Returns binary string (`0b...`) of bitwise OR with left zero-padding to max input width.
///
/// Time complexity: O(w)
/// Space complexity: O(w)
pub fn binaryOr(
    allocator: std.mem.Allocator,
    a: i64,
    b: i64,
) (BinaryOrError || std.mem.Allocator.Error)![]u8 {
    if (a < 0 or b < 0) return BinaryOrError.NegativeValue;

    const ua: u64 = @intCast(a);
    const ub: u64 = @intCast(b);
    const or_value = ua | ub;

    const width = @max(bitLengthU64(ua), bitLengthU64(ub));
    const out = try allocator.alloc(u8, 2 + width);
    out[0] = '0';
    out[1] = 'b';

    for (0..width) |i| {
        const bit_index = width - 1 - i;
        const bit = (or_value >> @intCast(bit_index)) & 1;
        out[2 + i] = if (bit == 1) '1' else '0';
    }
    return out;
}

test "binary or: python examples" {
    const alloc = testing.allocator;

    const c1 = try binaryOr(alloc, 25, 32);
    defer alloc.free(c1);
    try testing.expectEqualStrings("0b111001", c1);

    const c2 = try binaryOr(alloc, 37, 50);
    defer alloc.free(c2);
    try testing.expectEqualStrings("0b110111", c2);

    const c3 = try binaryOr(alloc, 21, 30);
    defer alloc.free(c3);
    try testing.expectEqualStrings("0b11111", c3);

    const c4 = try binaryOr(alloc, 58, 73);
    defer alloc.free(c4);
    try testing.expectEqualStrings("0b1111011", c4);

    const c5 = try binaryOr(alloc, 0, 255);
    defer alloc.free(c5);
    try testing.expectEqualStrings("0b11111111", c5);

    const c6 = try binaryOr(alloc, 0, 256);
    defer alloc.free(c6);
    try testing.expectEqualStrings("0b100000000", c6);
}

test "binary or: validation and boundary" {
    const alloc = testing.allocator;
    try testing.expectError(BinaryOrError.NegativeValue, binaryOr(alloc, 0, -1));

    const c0 = try binaryOr(alloc, 0, 0);
    defer alloc.free(c0);
    try testing.expectEqualStrings("0b0", c0);
}

test "binary or: extreme width" {
    const alloc = testing.allocator;
    const c = try binaryOr(alloc, std.math.maxInt(i64), 0);
    defer alloc.free(c);
    try testing.expectEqual(@as(usize, 65), c.len); // 0b + 63 bits
    try testing.expectEqual(@as(u8, '1'), c[2]);
}
