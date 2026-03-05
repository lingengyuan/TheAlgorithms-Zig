//! Word Ladder (Backtracking) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/word_ladder.py

const std = @import("std");
const testing = std.testing;

pub fn freeWordPath(allocator: std.mem.Allocator, path: [][]u8) void {
    for (path) |word| allocator.free(word);
    allocator.free(path);
}

fn freeOwnedWords(allocator: std.mem.Allocator, words: []const []u8) void {
    for (words) |word| allocator.free(word);
}

fn backtrack(
    allocator: std.mem.Allocator,
    current_word: []const u8,
    path: *std.ArrayListUnmanaged([]u8),
    end_word: []const u8,
    word_set: *std.StringHashMap(void),
) std.mem.Allocator.Error!bool {
    if (std.mem.eql(u8, current_word, end_word)) return true;

    for (0..current_word.len) |index| {
        var ch: u8 = 'a';
        while (ch <= 'z') : (ch += 1) {
            const transformed = try allocator.dupe(u8, current_word);
            defer allocator.free(transformed);
            transformed[index] = ch;

            const removed = word_set.fetchRemove(transformed) orelse continue;
            errdefer _ = word_set.put(removed.key, removed.value) catch {};

            const path_word = try allocator.dupe(u8, removed.key);
            try path.append(allocator, path_word);

            if (try backtrack(allocator, path_word, path, end_word, word_set)) return true;

            allocator.free(path.pop().?);
            try word_set.put(removed.key, removed.value);
        }
    }

    return false;
}

/// Solves the word ladder backtracking variant from the Python reference.
/// Returns one transformation path or an empty path if no solution exists.
///
/// Time complexity: exponential in word count and word length (pure backtracking).
/// Space complexity: O(n * m) for recursion/path, n=path length, m=word length.
pub fn wordLadder(
    allocator: std.mem.Allocator,
    begin_word: []const u8,
    end_word: []const u8,
    words: []const []const u8,
) std.mem.Allocator.Error![][]u8 {
    var word_set = std.StringHashMap(void).init(allocator);
    defer word_set.deinit();

    var owned_words = std.ArrayListUnmanaged([]u8){};
    defer {
        freeOwnedWords(allocator, owned_words.items);
        owned_words.deinit(allocator);
    }

    for (words) |word| {
        const dup = try allocator.dupe(u8, word);
        try owned_words.append(allocator, dup);
        try word_set.put(dup, {});
    }

    if (!word_set.contains(end_word)) {
        return allocator.alloc([]u8, 0);
    }

    var path = std.ArrayListUnmanaged([]u8){};
    errdefer {
        for (path.items) |word| allocator.free(word);
        path.deinit(allocator);
    }
    try path.append(allocator, try allocator.dupe(u8, begin_word));

    if (try backtrack(allocator, begin_word, &path, end_word, &word_set)) {
        return path.toOwnedSlice(allocator);
    }

    for (path.items) |word| allocator.free(word);
    path.deinit(allocator);
    return allocator.alloc([]u8, 0);
}

test "word ladder: python examples" {
    const alloc = testing.allocator;

    const p1 = try wordLadder(alloc, "hit", "cog", &[_][]const u8{ "hot", "dot", "dog", "lot", "log", "cog" });
    defer freeWordPath(alloc, p1);
    const expected1 = [_][]const u8{ "hit", "hot", "dot", "lot", "log", "cog" };
    try testing.expectEqual(expected1.len, p1.len);
    for (expected1, 0..) |item, i| try testing.expectEqualStrings(item, p1[i]);

    const p2 = try wordLadder(alloc, "hit", "cog", &[_][]const u8{ "hot", "dot", "dog", "lot", "log" });
    defer freeWordPath(alloc, p2);
    try testing.expectEqual(@as(usize, 0), p2.len);

    const p3 = try wordLadder(alloc, "lead", "gold", &[_][]const u8{ "load", "goad", "gold", "lead", "lord" });
    defer freeWordPath(alloc, p3);
    const expected3 = [_][]const u8{ "lead", "lead", "load", "goad", "gold" };
    try testing.expectEqual(expected3.len, p3.len);
    for (expected3, 0..) |item, i| try testing.expectEqualStrings(item, p3[i]);

    const p4 = try wordLadder(alloc, "game", "code", &[_][]const u8{ "came", "cage", "code", "cade", "gave" });
    defer freeWordPath(alloc, p4);
    const expected4 = [_][]const u8{ "game", "came", "cade", "code" };
    try testing.expectEqual(expected4.len, p4.len);
    for (expected4, 0..) |item, i| try testing.expectEqualStrings(item, p4[i]);
}

test "word ladder: boundary and edge cases" {
    const alloc = testing.allocator;

    const same = try wordLadder(alloc, "same", "same", &[_][]const u8{"same"});
    defer freeWordPath(alloc, same);
    try testing.expectEqual(@as(usize, 1), same.len);
    try testing.expectEqualStrings("same", same[0]);

    const no_end = try wordLadder(alloc, "abc", "xyz", &[_][]const u8{ "abz", "ayz" });
    defer freeWordPath(alloc, no_end);
    try testing.expectEqual(@as(usize, 0), no_end.len);
}

test "word ladder: extreme no-path search" {
    const alloc = testing.allocator;
    const words = [_][]const u8{
        "aaaa", "aaab", "aabb", "abbb", "bbbb",
        "zzzz", "zzzy", "zzyy", "zyyy", "yyyy",
    };
    const path = try wordLadder(alloc, "aaaa", "zzzz", &words);
    defer freeWordPath(alloc, path);
    try testing.expectEqual(@as(usize, 0), path.len);
}
