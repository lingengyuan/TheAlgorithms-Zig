//! Contains Unique Characters - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/is_contains_unique_chars.py

const std = @import("std");
const testing = std.testing;

/// Returns true if all bytes in input are unique.
/// Time complexity: O(n), Space complexity: O(1) for byte alphabet.
pub fn isContainsUniqueChars(input_str: []const u8) bool {
    var seen = [_]bool{false} ** 256;
    for (input_str) |ch| {
        if (seen[ch]) return false;
        seen[ch] = true;
    }
    return true;
}

test "contains unique chars: python reference examples" {
    try testing.expect(isContainsUniqueChars("I_love.py"));
    try testing.expect(!isContainsUniqueChars("I don't love Python"));
}

test "contains unique chars: edge and extreme cases" {
    try testing.expect(isContainsUniqueChars(""));
    try testing.expect(!isContainsUniqueChars("aa"));

    var all_bytes: [256]u8 = undefined;
    for (&all_bytes, 0..) |*slot, idx| slot.* = @intCast(idx);
    try testing.expect(isContainsUniqueChars(&all_bytes));

    var repeated: [257]u8 = undefined;
    for (&repeated, 0..) |*slot, idx| slot.* = @intCast(idx % 256);
    try testing.expect(!isContainsUniqueChars(&repeated));
}
