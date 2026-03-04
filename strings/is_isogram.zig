//! Is Isogram - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/is_isogram.py

const std = @import("std");
const testing = std.testing;

pub const IsogramError = error{NonAlphabeticCharacter};

/// Returns true if string is an isogram (no repeated letters).
/// Requires alphabetic-only input.
/// Time complexity: O(n), Space complexity: O(1)
pub fn isIsogram(string: []const u8) IsogramError!bool {
    var seen = [_]bool{false} ** 26;

    for (string) |ch| {
        if (!std.ascii.isAlphabetic(ch)) return IsogramError.NonAlphabeticCharacter;
        const lower = std.ascii.toLower(ch);
        const idx = lower - 'a';
        if (seen[idx]) return false;
        seen[idx] = true;
    }

    return true;
}

test "is isogram: python reference examples" {
    try testing.expect(try isIsogram("Uncopyrightable"));
    try testing.expect(!(try isIsogram("allowance")));
    try testing.expectError(IsogramError.NonAlphabeticCharacter, isIsogram("copy1"));
}

test "is isogram: edge and extreme cases" {
    try testing.expect(try isIsogram(""));
    try testing.expect(try isIsogram("a"));
    try testing.expect(!(try isIsogram("Aa"))); // case-insensitive duplicate

    var long_unique = [_]u8{0} ** 26;
    for (&long_unique, 0..) |*slot, i| slot.* = @as(u8, @intCast(i)) + 'A';
    try testing.expect(try isIsogram(&long_unique));
}
