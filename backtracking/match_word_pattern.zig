//! Match Word Pattern (Backtracking) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/match_word_pattern.py

const std = @import("std");
const testing = std.testing;

fn startsWithAt(haystack: []const u8, needle: []const u8, start_index: usize) bool {
    if (start_index + needle.len > haystack.len) return false;
    return std.mem.eql(u8, haystack[start_index .. start_index + needle.len], needle);
}

fn backtrack(
    pattern: []const u8,
    input_string: []const u8,
    pattern_index: usize,
    str_index: usize,
    pattern_map: *std.AutoHashMap(u8, []const u8),
    str_map: *std.StringHashMap(u8),
) std.mem.Allocator.Error!bool {
    if (pattern_index == pattern.len and str_index == input_string.len) return true;
    if (pattern_index == pattern.len or str_index == input_string.len) return false;

    const ch = pattern[pattern_index];
    if (pattern_map.get(ch)) |mapped_str| {
        if (!startsWithAt(input_string, mapped_str, str_index)) return false;
        return backtrack(pattern, input_string, pattern_index + 1, str_index + mapped_str.len, pattern_map, str_map);
    }

    var end = str_index + 1;
    while (end <= input_string.len) : (end += 1) {
        const substr = input_string[str_index..end];
        if (str_map.contains(substr)) continue;

        try pattern_map.put(ch, substr);
        try str_map.put(substr, ch);

        if (try backtrack(pattern, input_string, pattern_index + 1, end, pattern_map, str_map)) {
            return true;
        }

        _ = pattern_map.remove(ch);
        _ = str_map.remove(substr);
    }

    return false;
}

/// Determines whether `pattern` bijectively maps to substrings of `input_string`.
///
/// Time complexity: exponential in pattern/string lengths in worst case.
/// Space complexity: O(p + s) for recursion and maps.
pub fn matchWordPattern(
    allocator: std.mem.Allocator,
    pattern: []const u8,
    input_string: []const u8,
) std.mem.Allocator.Error!bool {
    var pattern_map = std.AutoHashMap(u8, []const u8).init(allocator);
    defer pattern_map.deinit();

    var str_map = std.StringHashMap(u8).init(allocator);
    defer str_map.deinit();

    return backtrack(pattern, input_string, 0, 0, &pattern_map, &str_map);
}

test "match word pattern: python examples" {
    const alloc = testing.allocator;

    try testing.expect(try matchWordPattern(alloc, "aba", "GraphTreesGraph"));
    try testing.expect(try matchWordPattern(alloc, "xyx", "PythonRubyPython"));
    try testing.expect(!(try matchWordPattern(alloc, "GG", "PythonJavaPython")));
}

test "match word pattern: boundary empty inputs" {
    const alloc = testing.allocator;

    try testing.expect(try matchWordPattern(alloc, "", ""));
    try testing.expect(!(try matchWordPattern(alloc, "a", "")));
    try testing.expect(!(try matchWordPattern(alloc, "", "nonempty")));
}

test "match word pattern: extreme branching and repeated mapping" {
    const alloc = testing.allocator;

    try testing.expect(try matchWordPattern(alloc, "aaaa", "xyzxyzxyzxyz"));
    try testing.expect(!(try matchWordPattern(alloc, "abca", "redbluegreenredx")));
}
