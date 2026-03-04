//! Distribute Coins In Binary Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/distribute_coins.py

const std = @import("std");
const testing = std.testing;

pub const TreeNode = struct {
    data: i64,
    left: ?*TreeNode = null,
    right: ?*TreeNode = null,
};

fn countNodes(root: ?*const TreeNode) usize {
    const node = root orelse return 0;
    return 1 + countNodes(node.left) + countNodes(node.right);
}

fn countCoins(root: ?*const TreeNode) i64 {
    const node = root orelse return 0;
    return node.data + countCoins(node.left) + countCoins(node.right);
}

const DfsResult = struct {
    moves: i64,
    excess: i64,
};

fn distributeDfs(root: ?*const TreeNode) DfsResult {
    const node = root orelse return .{ .moves = 0, .excess = 0 };

    const left = distributeDfs(node.left);
    const right = distributeDfs(node.right);

    const moves = left.moves + right.moves + @as(i64, @intCast(@abs(left.excess))) + @as(i64, @intCast(@abs(right.excess)));
    const excess = node.data + left.excess + right.excess - 1;

    return .{ .moves = moves, .excess = excess };
}

/// Returns minimum move count to make every node contain exactly one coin.
/// Time complexity: O(n), Space complexity: O(h)
pub fn distributeCoins(root: ?*const TreeNode) !usize {
    const node = root orelse return 0;

    const nodes = countNodes(node);
    const coins = countCoins(node);
    if (coins != @as(i64, @intCast(nodes))) return error.InvalidCoinCount;

    const result = distributeDfs(node);
    if (result.excess != 0 or result.moves < 0) return error.InvalidCoinCount;

    return @intCast(result.moves);
}

test "distribute coins: python examples" {
    {
        var left = TreeNode{ .data = 0 };
        var right = TreeNode{ .data = 0 };
        var root = TreeNode{ .data = 3, .left = &left, .right = &right };
        try testing.expectEqual(@as(usize, 2), try distributeCoins(&root));
    }

    {
        var left = TreeNode{ .data = 3 };
        var right = TreeNode{ .data = 0 };
        var root = TreeNode{ .data = 0, .left = &left, .right = &right };
        try testing.expectEqual(@as(usize, 3), try distributeCoins(&root));
    }

    {
        var left = TreeNode{ .data = 0 };
        var right = TreeNode{ .data = 3 };
        var root = TreeNode{ .data = 0, .left = &left, .right = &right };
        try testing.expectEqual(@as(usize, 3), try distributeCoins(&root));
    }

    try testing.expectEqual(@as(usize, 0), try distributeCoins(null));
}

test "distribute coins: invalid total coin count" {
    {
        var left = TreeNode{ .data = 0 };
        var right = TreeNode{ .data = 0 };
        var root = TreeNode{ .data = 0, .left = &left, .right = &right };
        try testing.expectError(error.InvalidCoinCount, distributeCoins(&root));
    }

    {
        var left = TreeNode{ .data = 1 };
        var right = TreeNode{ .data = 1 };
        var root = TreeNode{ .data = 0, .left = &left, .right = &right };
        try testing.expectError(error.InvalidCoinCount, distributeCoins(&root));
    }
}

test "distribute coins: extreme complete tree root-heavy" {
    const depth: usize = 11;
    const n: usize = (@as(usize, 1) << depth) - 1;

    const nodes = try testing.allocator.alloc(TreeNode, n);
    defer testing.allocator.free(nodes);

    for (0..n) |i| {
        nodes[i] = .{ .data = 0 };
    }
    for (0..n) |i| {
        const left = i * 2 + 1;
        const right = i * 2 + 2;
        nodes[i].left = if (left < n) &nodes[left] else null;
        nodes[i].right = if (right < n) &nodes[right] else null;
    }

    nodes[0].data = @intCast(n);

    var expected_moves: usize = 0;
    var consumed: usize = 0;
    var width: usize = 1;
    var level: usize = 0;
    while (consumed < n) : (level += 1) {
        const take = @min(width, n - consumed);
        expected_moves += level * take;
        consumed += take;
        width *= 2;
    }

    const moves = try distributeCoins(&nodes[0]);
    try testing.expectEqual(expected_moves, moves);
}
