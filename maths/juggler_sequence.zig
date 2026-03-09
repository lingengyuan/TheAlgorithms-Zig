//! Juggler Sequence - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/juggler_sequence.py

const std = @import("std");
const testing = std.testing;

pub const JugglerError = error{InvalidInput};

/// Returns the juggler sequence for a positive integer.
/// Caller owns the returned slice.
/// Time complexity: O(sequence length), Space complexity: O(sequence length)
pub fn jugglerSequence(allocator: std.mem.Allocator, number: i64) (JugglerError || std.mem.Allocator.Error)![]u64 {
    if (number < 1) return error.InvalidInput;

    var current: u64 = @intCast(number);
    var sequence = std.ArrayListUnmanaged(u64){};
    defer sequence.deinit(allocator);

    try sequence.append(allocator, current);
    while (current != 1) {
        if (current % 2 == 0) {
            current = std.math.sqrt(current);
        } else {
            const root = std.math.sqrt(@as(f128, @floatFromInt(current)));
            current = @intFromFloat(@floor(root * root * root));
        }
        try sequence.append(allocator, current);
    }
    return sequence.toOwnedSlice(allocator);
}

test "juggler sequence: python reference examples" {
    const alloc = testing.allocator;

    const s1 = try jugglerSequence(alloc, 1);
    defer alloc.free(s1);
    try testing.expectEqualSlices(u64, &[_]u64{1}, s1);

    const s2 = try jugglerSequence(alloc, 3);
    defer alloc.free(s2);
    try testing.expectEqualSlices(u64, &[_]u64{ 3, 5, 11, 36, 6, 2, 1 }, s2);

    const s3 = try jugglerSequence(alloc, 25);
    defer alloc.free(s3);
    try testing.expectEqualSlices(u64, &[_]u64{ 25, 125, 1397, 52214, 228, 15, 58, 7, 18, 4, 2, 1 }, s3);
}

test "juggler sequence: edge and extreme cases" {
    const alloc = testing.allocator;
    const s = try jugglerSequence(alloc, 2);
    defer alloc.free(s);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 1 }, s);
    try testing.expectError(error.InvalidInput, jugglerSequence(alloc, 0));
    try testing.expectError(error.InvalidInput, jugglerSequence(alloc, -1));
}
