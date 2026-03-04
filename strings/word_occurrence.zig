//! Word Occurrence - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/word_occurrence.py

const std = @import("std");
const testing = std.testing;

const whitespace = " \t\n\r\x0b\x0c";

/// Counts occurrences of words split by whitespace.
/// Returned map keys borrow slices from `sentence`.
/// Caller owns returned map and must call `deinit()`.
/// Time complexity: O(n) average, Space complexity: O(k)
pub fn wordOccurrence(allocator: std.mem.Allocator, sentence: []const u8) !std.StringHashMap(usize) {
    var occurrence = std.StringHashMap(usize).init(allocator);

    var tokenizer = std.mem.tokenizeAny(u8, sentence, whitespace);
    while (tokenizer.next()) |word| {
        const entry = try occurrence.getOrPut(word);
        if (!entry.found_existing) entry.value_ptr.* = 0;
        entry.value_ptr.* += 1;
    }

    return occurrence;
}

test "word occurrence: python reference examples" {
    const sentence = "a b A b c b d b d e f e g e h e i e j e 0";
    var counts = try wordOccurrence(testing.allocator, sentence);
    defer counts.deinit();

    try testing.expectEqual(@as(usize, 1), counts.get("a").?);
    try testing.expectEqual(@as(usize, 4), counts.get("b").?);
    try testing.expectEqual(@as(usize, 1), counts.get("A").?);
    try testing.expectEqual(@as(usize, 6), counts.get("e").?);
    try testing.expectEqual(@as(usize, 1), counts.get("0").?);
}

test "word occurrence: double spaces and boundaries" {
    var counts = try wordOccurrence(testing.allocator, "Two  spaces");
    defer counts.deinit();
    try testing.expectEqual(@as(usize, 2), counts.count());
    try testing.expectEqual(@as(usize, 1), counts.get("Two").?);
    try testing.expectEqual(@as(usize, 1), counts.get("spaces").?);

    var empty_counts = try wordOccurrence(testing.allocator, "");
    defer empty_counts.deinit();
    try testing.expectEqual(@as(usize, 0), empty_counts.count());
}

test "word occurrence: extreme repeated input" {
    const alloc = testing.allocator;
    var builder = std.ArrayListUnmanaged(u8){};
    defer builder.deinit(alloc);

    for (0..80_000) |_| try builder.appendSlice(alloc, "zig ");
    for (0..20_000) |_| try builder.appendSlice(alloc, "zag ");

    const sentence = builder.items;
    var counts = try wordOccurrence(alloc, sentence);
    defer counts.deinit();

    try testing.expectEqual(@as(usize, 2), counts.count());
    try testing.expectEqual(@as(usize, 80_000), counts.get("zig").?);
    try testing.expectEqual(@as(usize, 20_000), counts.get("zag").?);
}
