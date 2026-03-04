//! Boyer-Moore Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/boyer_moore_search.py

const std = @import("std");
const testing = std.testing;

fn buildLastOccurrence(pattern: []const u8) [256]isize {
    var last = [_]isize{-1} ** 256;
    for (pattern, 0..) |ch, index| {
        last[ch] = @intCast(index);
    }
    return last;
}

/// Finds all pattern occurrences in text using bad-character heuristic.
/// Caller owns returned slice.
/// Time complexity: O(n * m) worst case, sublinear average
/// Space complexity: O(1) for byte alphabet
pub fn boyerMooreSearch(allocator: std.mem.Allocator, text: []const u8, pattern: []const u8) ![]usize {
    if (pattern.len == 0 or pattern.len > text.len) {
        return try allocator.alloc(usize, 0);
    }

    var matches = std.ArrayListUnmanaged(usize){};
    errdefer matches.deinit(allocator);

    const n = text.len;
    const m = pattern.len;
    const last = buildLastOccurrence(pattern);

    var shift: usize = 0;
    while (shift + m <= n) {
        var j: isize = @as(isize, @intCast(m)) - 1;
        while (j >= 0) : (j -= 1) {
            const idx: usize = @intCast(j);
            if (pattern[idx] != text[shift + idx]) break;
        }

        if (j < 0) {
            try matches.append(allocator, shift);
            shift += 1; // keep overlap matches
            continue;
        }

        const mismatch_idx: usize = @intCast(j);
        const bad_char = text[shift + mismatch_idx];
        const bad_pos = last[bad_char];
        const jump = j - bad_pos;
        if (jump > 0) {
            shift += @as(usize, @intCast(jump));
        } else {
            shift += 1;
        }
    }

    return try matches.toOwnedSlice(allocator);
}

test "boyer moore search: python reference example" {
    const matches = try boyerMooreSearch(testing.allocator, "ABAABA", "AB");
    defer testing.allocator.free(matches);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 3 }, matches);
}

test "boyer moore search: overlap and no-match scenarios" {
    const overlap = try boyerMooreSearch(testing.allocator, "AAAAA", "AAA");
    defer testing.allocator.free(overlap);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2 }, overlap);

    const none = try boyerMooreSearch(testing.allocator, "hello", "world");
    defer testing.allocator.free(none);
    try testing.expectEqual(@as(usize, 0), none.len);
}

test "boyer moore search: boundary and extreme cases" {
    const empty_pattern = try boyerMooreSearch(testing.allocator, "abc", "");
    defer testing.allocator.free(empty_pattern);
    try testing.expectEqual(@as(usize, 0), empty_pattern.len);

    const longer_pattern = try boyerMooreSearch(testing.allocator, "abc", "abcdef");
    defer testing.allocator.free(longer_pattern);
    try testing.expectEqual(@as(usize, 0), longer_pattern.len);

    const alloc = testing.allocator;
    const text_len = 150_001;
    const long_text = try alloc.alloc(u8, text_len);
    defer alloc.free(long_text);
    @memset(long_text, 'a');
    long_text[text_len - 1] = 'b';

    const tail_match = try boyerMooreSearch(alloc, long_text, "ab");
    defer alloc.free(tail_match);
    try testing.expectEqualSlices(usize, &[_]usize{text_len - 2}, tail_match);
}
