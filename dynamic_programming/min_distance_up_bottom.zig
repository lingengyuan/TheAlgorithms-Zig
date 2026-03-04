//! Minimum Distance (Top-Down Edit Distance) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/min_distance_up_bottom.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const MinDistanceUpBottomError = error{
    Overflow,
};

fn index(cols: usize, row: usize, col: usize) usize {
    return row * cols + col;
}

/// Computes Levenshtein distance using top-down recursion + memoization.
/// Time complexity: O(m * n), Space complexity: O(m * n)
pub fn minDistanceUpBottom(
    allocator: Allocator,
    word1: []const u8,
    word2: []const u8,
) (MinDistanceUpBottomError || Allocator.Error)!usize {
    const rows = @addWithOverflow(word1.len, @as(usize, 1));
    if (rows[1] != 0) return MinDistanceUpBottomError.Overflow;
    const cols = @addWithOverflow(word2.len, @as(usize, 1));
    if (cols[1] != 0) return MinDistanceUpBottomError.Overflow;
    const cells = @mulWithOverflow(rows[0], cols[0]);
    if (cells[1] != 0) return MinDistanceUpBottomError.Overflow;

    const unseen = std.math.maxInt(usize);
    const memo = try allocator.alloc(usize, cells[0]);
    defer allocator.free(memo);
    @memset(memo, unseen);

    return recurse(word1, word2, memo, cols[0], 0, 0);
}

fn recurse(
    word1: []const u8,
    word2: []const u8,
    memo: []usize,
    cols: usize,
    i: usize,
    j: usize,
) MinDistanceUpBottomError!usize {
    if (i >= word1.len) return word2.len - j;
    if (j >= word2.len) return word1.len - i;

    const idx = index(cols, i, j);
    if (memo[idx] != std.math.maxInt(usize)) return memo[idx];

    const delete_step = try recurse(word1, word2, memo, cols, i + 1, j);
    const insert_step = try recurse(word1, word2, memo, cols, i, j + 1);
    const replace_step = try recurse(word1, word2, memo, cols, i + 1, j + 1);
    const diff: usize = if (word1[i] == word2[j]) 0 else 1;

    const c1 = @addWithOverflow(@as(usize, 1), delete_step);
    if (c1[1] != 0) return MinDistanceUpBottomError.Overflow;
    const c2 = @addWithOverflow(@as(usize, 1), insert_step);
    if (c2[1] != 0) return MinDistanceUpBottomError.Overflow;
    const c3 = @addWithOverflow(diff, replace_step);
    if (c3[1] != 0) return MinDistanceUpBottomError.Overflow;

    const best = @min(c1[0], @min(c2[0], c3[0]));
    memo[idx] = best;
    return best;
}

test "min distance up bottom: python examples" {
    try testing.expectEqual(@as(usize, 5), try minDistanceUpBottom(testing.allocator, "intention", "execution"));
    try testing.expectEqual(@as(usize, 9), try minDistanceUpBottom(testing.allocator, "intention", ""));
    try testing.expectEqual(@as(usize, 0), try minDistanceUpBottom(testing.allocator, "", ""));
    try testing.expectEqual(@as(usize, 10), try minDistanceUpBottom(testing.allocator, "zooicoarchaeologist", "zoologist"));
}

test "min distance up bottom: boundary and identity" {
    try testing.expectEqual(@as(usize, 0), try minDistanceUpBottom(testing.allocator, "abc", "abc"));
    try testing.expectEqual(@as(usize, 1), try minDistanceUpBottom(testing.allocator, "a", "b"));
    try testing.expectEqual(@as(usize, 3), try minDistanceUpBottom(testing.allocator, "", "abc"));
}

test "min distance up bottom: extreme long identical string" {
    var a: [2048]u8 = undefined;
    @memset(&a, 'x');
    try testing.expectEqual(@as(usize, 0), try minDistanceUpBottom(testing.allocator, &a, &a));
}
