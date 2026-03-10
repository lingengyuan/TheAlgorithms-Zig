//! Wave String - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/wave_string.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const WaveResult = []const []u8;

/// Returns the Mexican-wave variants of the input string.
/// Time complexity: O(n²), Space complexity: O(k · n)
pub fn wave(allocator: Allocator, txt: []const u8) !WaveResult {
    var results = std.ArrayListUnmanaged([]u8){};
    errdefer freeWaveResults(allocator, results.items);

    for (txt, 0..) |char, index| {
        if (!std.ascii.isAlphabetic(char)) continue;
        const item = try allocator.dupe(u8, txt);
        item[index] = std.ascii.toUpper(item[index]);
        try results.append(allocator, item);
    }

    return results.toOwnedSlice(allocator);
}

pub fn freeWaveResults(allocator: Allocator, results: WaveResult) void {
    for (results) |item| allocator.free(item);
    allocator.free(results);
}

test "wave string: python samples" {
    const cat = try wave(testing.allocator, "cat");
    defer freeWaveResults(testing.allocator, cat);
    try testing.expectEqual(@as(usize, 3), cat.len);
    try testing.expectEqualStrings("Cat", cat[0]);
    try testing.expectEqualStrings("cAt", cat[1]);
    try testing.expectEqualStrings("caT", cat[2]);

    const one = try wave(testing.allocator, "one");
    defer freeWaveResults(testing.allocator, one);
    try testing.expectEqualStrings("One", one[0]);
    try testing.expectEqualStrings("oNe", one[1]);
    try testing.expectEqualStrings("onE", one[2]);
}

test "wave string: edge and extreme" {
    const spaced = try wave(testing.allocator, "a b");
    defer freeWaveResults(testing.allocator, spaced);
    try testing.expectEqual(@as(usize, 2), spaced.len);
    try testing.expectEqualStrings("A b", spaced[0]);
    try testing.expectEqualStrings("a B", spaced[1]);

    const empty = try wave(testing.allocator, "");
    defer freeWaveResults(testing.allocator, empty);
    try testing.expectEqual(@as(usize, 0), empty.len);
}
