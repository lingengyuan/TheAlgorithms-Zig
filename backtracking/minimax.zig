//! Minimax (Game Tree) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/minimax.py

const std = @import("std");
const testing = std.testing;

pub const MinimaxError = error{
    DepthNegative,
    ScoresEmpty,
    InvalidHeight,
    NodeIndexOutOfRange,
    Overflow,
};

/// Minimax recursion over a complete binary game tree represented by leaf scores.
///
/// Time complexity: O(2^h)
/// Space complexity: O(h)
pub fn minimax(
    depth: i64,
    node_index: usize,
    is_max: bool,
    scores: []const i64,
    height: i64,
) MinimaxError!i64 {
    if (depth < 0) return MinimaxError.DepthNegative;
    if (scores.len == 0) return MinimaxError.ScoresEmpty;
    if (height < 0) return MinimaxError.InvalidHeight;
    if (depth > height) return MinimaxError.InvalidHeight;

    if (depth == height) {
        if (node_index >= scores.len) return MinimaxError.NodeIndexOutOfRange;
        return scores[node_index];
    }

    const left_mul = @mulWithOverflow(node_index, 2);
    if (left_mul[1] != 0) return MinimaxError.Overflow;
    const left_index = left_mul[0];
    const right_add = @addWithOverflow(left_index, 1);
    if (right_add[1] != 0) return MinimaxError.Overflow;
    const right_index = right_add[0];

    const left_value = try minimax(depth + 1, left_index, !is_max, scores, height);
    const right_value = try minimax(depth + 1, right_index, !is_max, scores, height);

    return if (is_max) @max(left_value, right_value) else @min(left_value, right_value);
}

test "minimax: python examples" {
    const scores1 = [_]i64{ 90, 23, 6, 33, 21, 65, 123, 34423 };
    try testing.expectEqual(@as(i64, 65), try minimax(0, 0, true, &scores1, 3));

    const scores2 = [_]i64{ 3, 5, 2, 9, 12, 5, 23, 23 };
    try testing.expectEqual(@as(i64, 12), try minimax(0, 0, true, &scores2, 3));
}

test "minimax: validation" {
    const scores = [_]i64{ 1, 2 };

    try testing.expectError(MinimaxError.DepthNegative, minimax(-1, 0, true, &scores, 1));
    try testing.expectError(MinimaxError.ScoresEmpty, minimax(0, 0, true, &[_]i64{}, 0));
    try testing.expectError(MinimaxError.InvalidHeight, minimax(2, 0, true, &scores, 1));
    try testing.expectError(MinimaxError.NodeIndexOutOfRange, minimax(0, 0, true, &scores, 2));
}

test "minimax: extreme full tree" {
    var leaves: [1024]i64 = undefined;
    for (0..leaves.len) |i| {
        leaves[i] = @as(i64, @intCast((i * 97 + 13) % 10_000));
    }

    const result = try minimax(0, 0, true, leaves[0..], 10);
    try testing.expect(result >= 0);
    try testing.expect(result <= 9_999);
}
