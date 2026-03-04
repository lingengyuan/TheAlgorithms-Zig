//! Top K Frequent Words - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/top_k_frequent_words.py

const std = @import("std");
const testing = std.testing;

const CountInfo = struct {
    count: usize,
    first_index: usize,
};

const WordStat = struct {
    word: []const u8,
    count: usize,
    first_index: usize,
};

fn statLessThan(_: void, a: WordStat, b: WordStat) bool {
    if (a.count != b.count) return a.count > b.count;
    return a.first_index < b.first_index;
}

/// Returns top-k words by descending frequency.
/// Ties are resolved by first appearance order (matching Counter insertion order behavior).
/// Returned outer slice is owned by caller; inner slices reference `words`.
/// Time complexity: O(n + u log u), Space complexity: O(u)
/// where u is number of unique words.
pub fn topKFrequentWords(
    allocator: std.mem.Allocator,
    words: []const []const u8,
    kValue: usize,
) ![][]const u8 {
    if (words.len == 0 or kValue == 0) return try allocator.alloc([]const u8, 0);

    var counts = std.StringHashMap(CountInfo).init(allocator);
    defer counts.deinit();

    for (words, 0..) |word, index| {
        const entry = try counts.getOrPut(word);
        if (!entry.found_existing) {
            entry.value_ptr.* = .{ .count = 0, .first_index = index };
        }
        entry.value_ptr.count += 1;
    }

    const stats = try allocator.alloc(WordStat, counts.count());
    defer allocator.free(stats);

    var i: usize = 0;
    var iterator = counts.iterator();
    while (iterator.next()) |entry| {
        stats[i] = .{
            .word = entry.key_ptr.*,
            .count = entry.value_ptr.count,
            .first_index = entry.value_ptr.first_index,
        };
        i += 1;
    }

    std.mem.sort(WordStat, stats, {}, statLessThan);

    const out_len = @min(kValue, stats.len);
    const out = try allocator.alloc([]const u8, out_len);
    for (0..out_len) |idx| out[idx] = stats[idx].word;
    return out;
}

test "top k frequent words: python reference examples" {
    const alloc = testing.allocator;
    const words = [_][]const u8{ "a", "b", "c", "a", "c", "c" };

    const r1 = try topKFrequentWords(alloc, &words, 3);
    defer alloc.free(r1);
    try testing.expectEqual(@as(usize, 3), r1.len);
    try testing.expectEqualStrings("c", r1[0]);
    try testing.expectEqualStrings("a", r1[1]);
    try testing.expectEqualStrings("b", r1[2]);

    const r2 = try topKFrequentWords(alloc, &words, 2);
    defer alloc.free(r2);
    try testing.expectEqual(@as(usize, 2), r2.len);
    try testing.expectEqualStrings("c", r2[0]);
    try testing.expectEqualStrings("a", r2[1]);

    const r3 = try topKFrequentWords(alloc, &words, 1);
    defer alloc.free(r3);
    try testing.expectEqual(@as(usize, 1), r3.len);
    try testing.expectEqualStrings("c", r3[0]);

    const r4 = try topKFrequentWords(alloc, &words, 0);
    defer alloc.free(r4);
    try testing.expectEqual(@as(usize, 0), r4.len);

    const empty_words = [_][]const u8{};
    const r5 = try topKFrequentWords(alloc, &empty_words, 1);
    defer alloc.free(r5);
    try testing.expectEqual(@as(usize, 0), r5.len);

    const single = [_][]const u8{ "a", "a" };
    const r6 = try topKFrequentWords(alloc, &single, 2);
    defer alloc.free(r6);
    try testing.expectEqual(@as(usize, 1), r6.len);
    try testing.expectEqualStrings("a", r6[0]);
}

test "top k frequent words: tie by first appearance" {
    const alloc = testing.allocator;
    const words = [_][]const u8{ "b", "a", "c", "a", "b", "d" };
    const result = try topKFrequentWords(alloc, &words, 4);
    defer alloc.free(result);

    try testing.expectEqual(@as(usize, 4), result.len);
    try testing.expectEqualStrings("b", result[0]); // b and a both frequency 2, b appears first
    try testing.expectEqualStrings("a", result[1]);
}

test "top k frequent words: extreme large input" {
    const alloc = testing.allocator;
    const words = try alloc.alloc([]const u8, 160_000);
    defer alloc.free(words);

    var idx: usize = 0;
    while (idx < 90_000) : (idx += 1) words[idx] = "zig";
    while (idx < 140_000) : (idx += 1) words[idx] = "zag";
    while (idx < 160_000) : (idx += 1) words[idx] = "zip";

    const top2 = try topKFrequentWords(alloc, words, 2);
    defer alloc.free(top2);
    try testing.expectEqual(@as(usize, 2), top2.len);
    try testing.expectEqualStrings("zig", top2[0]);
    try testing.expectEqualStrings("zag", top2[1]);
}
