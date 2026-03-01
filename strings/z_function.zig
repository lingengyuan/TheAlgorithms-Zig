//! Z-Function (Z-Algorithm) for string matching - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/z_function.py

const std = @import("std");
const testing = std.testing;

pub const ZError = error{Overflow};

/// Computes the Z-array for s. z[i] = length of the longest substring starting
/// at s[i] that is also a prefix of s. z[0] is undefined (set to 0).
/// Caller owns the returned slice. Time complexity: O(n)
pub fn zFunction(allocator: std.mem.Allocator, s: []const u8) (ZError || std.mem.Allocator.Error)![]usize {
    const n = s.len;
    const z = try allocator.alloc(usize, n);
    @memset(z, 0);
    var l: usize = 0;
    var r: usize = 0;
    for (1..n) |i| {
        if (i < r) {
            z[i] = @min(r - i, z[i - l]);
        }
        while (i + z[i] < n and s[z[i]] == s[i + z[i]]) {
            z[i] += 1;
        }
        if (i + z[i] > r) {
            l = i;
            r = i + z[i];
        }
    }
    return z;
}

/// Returns all positions where pattern appears in text using the Z-algorithm.
/// Caller owns the returned slice.
pub fn zSearch(allocator: std.mem.Allocator, text: []const u8, pattern: []const u8) ![]usize {
    if (pattern.len == 0 or pattern.len > text.len) {
        return try allocator.alloc(usize, 0);
    }
    // Concatenate pattern + sentinel + text
    const with_sentinel = @addWithOverflow(pattern.len, @as(usize, 1));
    if (with_sentinel[1] != 0) return ZError.Overflow;
    const combined_len = @addWithOverflow(with_sentinel[0], text.len);
    if (combined_len[1] != 0) return ZError.Overflow;
    const combined = try allocator.alloc(u8, combined_len[0]);
    defer allocator.free(combined);
    @memcpy(combined[0..pattern.len], pattern);
    combined[pattern.len] = 0; // sentinel (must not appear in alphabet)
    @memcpy(combined[pattern.len + 1 ..], text);

    const z = try zFunction(allocator, combined);
    defer allocator.free(z);

    var result = std.ArrayListUnmanaged(usize){};
    defer result.deinit(allocator);

    const offset = with_sentinel[0];
    for (offset..combined.len) |i| {
        if (z[i] == pattern.len) {
            try result.append(allocator, i - offset);
        }
    }

    const out = try allocator.alloc(usize, result.items.len);
    @memcpy(out, result.items);
    return out;
}

test "z function: known z-array" {
    const alloc = testing.allocator;
    // "aabxaa": z = [0, 1, 0, 0, 2, 1]
    const z = try zFunction(alloc, "aabxaa");
    defer alloc.free(z);
    try testing.expectEqual(@as(usize, 0), z[0]);
    try testing.expectEqual(@as(usize, 1), z[1]);
    try testing.expectEqual(@as(usize, 0), z[2]);
    try testing.expectEqual(@as(usize, 2), z[4]);
}

test "z search: multiple matches" {
    const alloc = testing.allocator;
    const r = try zSearch(alloc, "ABAAABCDBBABCDDEBCABC", "ABC");
    defer alloc.free(r);
    try testing.expectEqualSlices(usize, &[_]usize{ 4, 10, 18 }, r);
}

test "z search: no match" {
    const alloc = testing.allocator;
    const r = try zSearch(alloc, "hello", "world");
    defer alloc.free(r);
    try testing.expectEqual(@as(usize, 0), r.len);
}

test "z search: empty pattern" {
    const alloc = testing.allocator;
    const r = try zSearch(alloc, "hello", "");
    defer alloc.free(r);
    try testing.expectEqual(@as(usize, 0), r.len);
}

test "z search: oversize combined length returns overflow" {
    const fake_ptr: [*]const u8 = @ptrFromInt(@alignOf(u8));
    const fake_text = fake_ptr[0..std.math.maxInt(usize)];
    try testing.expectError(ZError.Overflow, zSearch(testing.allocator, fake_text, "a"));
}
