//! Naive String Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/naive_string_search.py

const std = @import("std");
const testing = std.testing;

/// Finds all occurrences of pattern in text. Returns a slice of starting indices.
/// Caller owns the returned slice.
/// Time complexity: O(n × m)
pub fn naiveSearch(allocator: std.mem.Allocator, text: []const u8, pattern: []const u8) ![]usize {
    var result = std.ArrayListUnmanaged(usize){};
    defer result.deinit(allocator);

    if (pattern.len == 0 or pattern.len > text.len) {
        return try allocator.alloc(usize, 0);
    }

    const limit = text.len - pattern.len + 1;
    for (0..limit) |i| {
        if (std.mem.eql(u8, text[i..][0..pattern.len], pattern)) {
            try result.append(allocator, i);
        }
    }

    const out = try allocator.alloc(usize, result.items.len);
    @memcpy(out, result.items);
    return out;
}

test "naive search: multiple matches" {
    const alloc = testing.allocator;
    const r = try naiveSearch(alloc, "ABAAABCDBBABCDDEBCABC", "ABC");
    defer alloc.free(r);
    try testing.expectEqualSlices(usize, &[_]usize{ 4, 10, 18 }, r);
}

test "naive search: no match" {
    const alloc = testing.allocator;
    const r = try naiveSearch(alloc, "ABC", "ABAAABCDBBABCDDEBCABC");
    defer alloc.free(r);
    try testing.expectEqual(@as(usize, 0), r.len);
}

test "naive search: empty text" {
    const alloc = testing.allocator;
    const r = try naiveSearch(alloc, "", "ABC");
    defer alloc.free(r);
    try testing.expectEqual(@as(usize, 0), r.len);
}

test "naive search: exact match" {
    const alloc = testing.allocator;
    const r = try naiveSearch(alloc, "TEST", "TEST");
    defer alloc.free(r);
    try testing.expectEqualSlices(usize, &[_]usize{0}, r);
}

test "naive search: end of string" {
    const alloc = testing.allocator;
    const r = try naiveSearch(alloc, "ABCDEGFTEST", "TEST");
    defer alloc.free(r);
    try testing.expectEqualSlices(usize, &[_]usize{7}, r);
}
