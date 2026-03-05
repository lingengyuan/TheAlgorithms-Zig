//! Gray Code Sequence - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/gray_code_sequence.py

const std = @import("std");
const testing = std.testing;

pub const GrayCodeError = error{ NegativeBitCount, BitCountTooLarge };

/// Generates an n-bit Gray code sequence of length 2^n.
///
/// Time complexity: O(2^n)
/// Space complexity: O(2^n)
pub fn grayCode(
    allocator: std.mem.Allocator,
    bit_count: i32,
) (GrayCodeError || std.mem.Allocator.Error)![]u64 {
    if (bit_count < 0) return GrayCodeError.NegativeBitCount;
    if (bit_count >= @bitSizeOf(usize)) return GrayCodeError.BitCountTooLarge;

    const shift: usize = @intCast(bit_count);
    const len = @as(usize, 1) << @intCast(shift);

    const sequence = try allocator.alloc(u64, len);
    for (0..len) |i| {
        const value: u64 = @intCast(i);
        sequence[i] = value ^ (value >> 1);
    }
    return sequence;
}

test "gray code: known sequences" {
    const alloc = testing.allocator;

    const seq1 = try grayCode(alloc, 1);
    defer alloc.free(seq1);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 1 }, seq1);

    const seq2 = try grayCode(alloc, 2);
    defer alloc.free(seq2);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 1, 3, 2 }, seq2);

    const seq3 = try grayCode(alloc, 3);
    defer alloc.free(seq3);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 1, 3, 2, 6, 7, 5, 4 }, seq3);
}

test "gray code: zero bit count" {
    const alloc = testing.allocator;
    const seq0 = try grayCode(alloc, 0);
    defer alloc.free(seq0);

    try testing.expectEqual(@as(usize, 1), seq0.len);
    try testing.expectEqual(@as(u64, 0), seq0[0]);
}

test "gray code: adjacent numbers differ by one bit" {
    const alloc = testing.allocator;
    const seq = try grayCode(alloc, 5);
    defer alloc.free(seq);

    for (0..seq.len - 1) |i| {
        const diff = seq[i] ^ seq[i + 1];
        try testing.expect(@popCount(diff) == 1);
    }

    const circular_diff = seq[0] ^ seq[seq.len - 1];
    try testing.expect(@popCount(circular_diff) == 1);
}

test "gray code: invalid bit count" {
    const alloc = testing.allocator;
    try testing.expectError(GrayCodeError.NegativeBitCount, grayCode(alloc, -1));

    const too_large: i32 = @intCast(@bitSizeOf(usize));
    try testing.expectError(GrayCodeError.BitCountTooLarge, grayCode(alloc, too_large));
}
