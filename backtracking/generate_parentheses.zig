//! Generate Valid Parentheses - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/generate_parentheses.py

const std = @import("std");
const testing = std.testing;

fn backtrack(
    allocator: std.mem.Allocator,
    buf: []u8,
    pos: usize,
    open: usize,
    close: usize,
    total_len: usize,
    n: usize,
    result: *std.ArrayListUnmanaged([]u8),
) !void {
    if (pos == total_len) {
        const copy = try allocator.dupe(u8, buf[0..pos]);
        try result.append(allocator, copy);
        return;
    }
    if (open < n) {
        buf[pos] = '(';
        try backtrack(allocator, buf, pos + 1, open + 1, close, total_len, n, result);
    }
    if (close < open) {
        buf[pos] = ')';
        try backtrack(allocator, buf, pos + 1, open, close + 1, total_len, n, result);
    }
}

/// Generates all combinations of n pairs of valid parentheses.
/// Caller must free each string and call result.deinit().
pub fn generateParentheses(
    allocator: std.mem.Allocator,
    n: usize,
    result: *std.ArrayListUnmanaged([]u8),
) !void {
    if (n == 0) return;
    const total_len = @mulWithOverflow(n, @as(usize, 2));
    if (total_len[1] != 0) return error.Overflow;
    const buf = try allocator.alloc(u8, total_len[0]);
    defer allocator.free(buf);
    try backtrack(allocator, buf, 0, 0, 0, total_len[0], n, result);
}

test "parentheses: n=2" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]u8){};
    defer {
        for (result.items) |s| alloc.free(s);
        result.deinit(alloc);
    }
    try generateParentheses(alloc, 2, &result);
    try testing.expectEqual(@as(usize, 2), result.items.len);
    try testing.expectEqualStrings("(())", result.items[0]);
    try testing.expectEqualStrings("()()", result.items[1]);
}

test "parentheses: n=3 has 5 combinations" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]u8){};
    defer {
        for (result.items) |s| alloc.free(s);
        result.deinit(alloc);
    }
    try generateParentheses(alloc, 3, &result);
    try testing.expectEqual(@as(usize, 5), result.items.len); // Catalan(3) = 5
}

test "parentheses: n=1" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]u8){};
    defer {
        for (result.items) |s| alloc.free(s);
        result.deinit(alloc);
    }
    try generateParentheses(alloc, 1, &result);
    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqualStrings("()", result.items[0]);
}

test "parentheses: n=0 produces nothing" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]u8){};
    defer result.deinit(alloc);
    try generateParentheses(alloc, 0, &result);
    try testing.expectEqual(@as(usize, 0), result.items.len);
}

test "parentheses: oversize n reports overflow" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]u8){};
    defer result.deinit(alloc);
    try testing.expectError(error.Overflow, generateParentheses(alloc, std.math.maxInt(usize), &result));
}
