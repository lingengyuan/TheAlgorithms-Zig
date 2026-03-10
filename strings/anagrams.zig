//! Anagrams - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/anagrams.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const StringList = []const []const u8;

/// Returns the frequency signature of a word.
/// Caller owns returned slice.
/// Time complexity: O(n log k), Space complexity: O(k)
pub fn signature(allocator: Allocator, word: []const u8) ![]u8 {
    var counts = std.AutoHashMap(u8, usize).init(allocator);
    defer counts.deinit();
    for (word) |char| {
        const entry = try counts.getOrPut(char);
        if (!entry.found_existing) entry.value_ptr.* = 0;
        entry.value_ptr.* += 1;
    }

    var keys = std.ArrayListUnmanaged(u8){};
    defer keys.deinit(allocator);
    var it = counts.iterator();
    while (it.next()) |entry| try keys.append(allocator, entry.key_ptr.*);
    std.sort.heap(u8, keys.items, {}, std.sort.asc(u8));

    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(allocator);
    for (keys.items) |char| {
        try out.append(allocator, char);
        try out.writer(allocator).print("{d}", .{counts.get(char).?});
    }
    return out.toOwnedSlice(allocator);
}

/// Returns every anagram of `my_word` from the provided dictionary list.
/// Caller owns the outer slice.
/// Time complexity: O(d * n log k), Space complexity: O(k)
pub fn anagram(allocator: Allocator, my_word: []const u8, words: []const []const u8) !StringList {
    const target_signature = try signature(allocator, my_word);
    defer allocator.free(target_signature);

    var result = std.ArrayListUnmanaged([]const u8){};
    defer result.deinit(allocator);
    for (words) |word| {
        const word_signature = try signature(allocator, word);
        defer allocator.free(word_signature);
        if (std.mem.eql(u8, target_signature, word_signature)) {
            try result.append(allocator, word);
        }
    }
    return result.toOwnedSlice(allocator);
}

pub fn freeStringRefs(allocator: Allocator, items: StringList) void {
    allocator.free(items);
}

test "anagrams: signature samples" {
    const one = try signature(testing.allocator, "test");
    defer testing.allocator.free(one);
    try testing.expectEqualStrings("e1s1t2", one);

    const two = try signature(testing.allocator, "this is a test");
    defer testing.allocator.free(two);
    try testing.expectEqualStrings(" 3a1e1h1i2s3t3", two);
}

test "anagrams: sample dictionary matches" {
    const words = [_][]const u8{ "final", "sett", "stet", "test" };
    const one = try anagram(testing.allocator, "test", &words);
    defer freeStringRefs(testing.allocator, one);
    try testing.expectEqual(@as(usize, 3), one.len);
    try testing.expectEqualStrings("sett", one[0]);
    try testing.expectEqualStrings("stet", one[1]);
    try testing.expectEqualStrings("test", one[2]);

    const two = try anagram(testing.allocator, "this is a test", &words);
    defer freeStringRefs(testing.allocator, two);
    try testing.expectEqual(@as(usize, 0), two.len);
}
