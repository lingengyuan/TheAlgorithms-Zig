//! Binary XOR Operator - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/binary_xor_operator.py

const std = @import("std");
const testing = std.testing;

pub const BinaryXorError = error{NegativeValue};

fn bitLengthU64(value: u64) usize {
    return if (value == 0) 1 else 64 - @clz(value);
}

/// Returns binary string (`0b...`) of bitwise XOR with left zero-padding to max input width.
///
/// Time complexity: O(w)
/// Space complexity: O(w)
pub fn binaryXor(
    allocator: std.mem.Allocator,
    a: i64,
    b: i64,
) (BinaryXorError || std.mem.Allocator.Error)![]u8 {
    if (a < 0 or b < 0) return BinaryXorError.NegativeValue;

    const ua: u64 = @intCast(a);
    const ub: u64 = @intCast(b);
    const xor_value = ua ^ ub;

    const width = @max(bitLengthU64(ua), bitLengthU64(ub));
    const out = try allocator.alloc(u8, 2 + width);
    out[0] = '0';
    out[1] = 'b';

    for (0..width) |i| {
        const bit_index = width - 1 - i;
        const bit = (xor_value >> @intCast(bit_index)) & 1;
        out[2 + i] = if (bit == 1) '1' else '0';
    }
    return out;
}

test "binary xor: python examples" {
    const alloc = testing.allocator;

    const c1 = try binaryXor(alloc, 25, 32);
    defer alloc.free(c1);
    try testing.expectEqualStrings("0b111001", c1);

    const c2 = try binaryXor(alloc, 37, 50);
    defer alloc.free(c2);
    try testing.expectEqualStrings("0b010111", c2);

    const c3 = try binaryXor(alloc, 21, 30);
    defer alloc.free(c3);
    try testing.expectEqualStrings("0b01011", c3);

    const c4 = try binaryXor(alloc, 58, 73);
    defer alloc.free(c4);
    try testing.expectEqualStrings("0b1110011", c4);

    const c5 = try binaryXor(alloc, 0, 255);
    defer alloc.free(c5);
    try testing.expectEqualStrings("0b11111111", c5);

    const c6 = try binaryXor(alloc, 256, 256);
    defer alloc.free(c6);
    try testing.expectEqualStrings("0b000000000", c6);
}

test "binary xor: validation and boundary" {
    const alloc = testing.allocator;
    try testing.expectError(BinaryXorError.NegativeValue, binaryXor(alloc, 0, -1));

    const c0 = try binaryXor(alloc, 0, 0);
    defer alloc.free(c0);
    try testing.expectEqualStrings("0b0", c0);
}

test "binary xor: extreme width" {
    const alloc = testing.allocator;
    const c = try binaryXor(alloc, std.math.maxInt(i64), 0);
    defer alloc.free(c);
    try testing.expectEqual(@as(usize, 65), c.len); // 0b + 63 bits
    try testing.expectEqual(@as(u8, '1'), c[2]);
}
