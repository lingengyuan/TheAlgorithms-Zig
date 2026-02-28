//! Knuth-Morris-Pratt (KMP) String Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/knuth_morris_pratt.py

const std = @import("std");
const testing = std.testing;

/// Builds the KMP failure function (prefix table) for pattern.
/// Caller owns the returned slice.
fn buildFailure(allocator: std.mem.Allocator, pattern: []const u8) ![]usize {
    const m = pattern.len;
    const f = try allocator.alloc(usize, m);
    f[0] = 0;
    var k: usize = 0;
    for (1..m) |i| {
        while (k > 0 and pattern[k] != pattern[i]) {
            k = f[k - 1];
        }
        if (pattern[k] == pattern[i]) k += 1;
        f[i] = k;
    }
    return f;
}

/// Returns the index of the first occurrence of pattern in text, or null.
/// Time complexity: O(n + m)
pub fn kmpSearch(allocator: std.mem.Allocator, text: []const u8, pattern: []const u8) !?usize {
    if (pattern.len == 0) return 0;
    if (pattern.len > text.len) return null;

    const failure = try buildFailure(allocator, pattern);
    defer allocator.free(failure);

    var j: usize = 0;
    for (text, 0..) |c, i| {
        while (j > 0 and pattern[j] != c) {
            j = failure[j - 1];
        }
        if (pattern[j] == c) j += 1;
        if (j == pattern.len) {
            return i + 1 - pattern.len;
        }
    }
    return null;
}

test "kmp: found" {
    const alloc = testing.allocator;
    const text = "knuth_morris_pratt";
    try testing.expectEqual(@as(?usize, 0), try kmpSearch(alloc, text, "kn"));
    try testing.expectEqual(@as(?usize, 4), try kmpSearch(alloc, text, "h_m"));
    try testing.expectEqual(@as(?usize, 16), try kmpSearch(alloc, text, "tt"));
    try testing.expectEqual(@as(?usize, 8), try kmpSearch(alloc, text, "rr"));
}

test "kmp: not found" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(?usize, null), try kmpSearch(alloc, "hello", "world"));
    try testing.expectEqual(@as(?usize, null), try kmpSearch(alloc, "abcde", "xyz"));
}

test "kmp: empty pattern returns 0" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(?usize, 0), try kmpSearch(alloc, "hello", ""));
}

test "kmp: pattern longer than text" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(?usize, null), try kmpSearch(alloc, "hi", "hello"));
}
