//! Maximum Sum BST In Binary Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/maximum_sum_bst.py

const std = @import("std");
const testing = std.testing;

pub const TreeNode = struct {
    val: i64,
    left: ?*TreeNode = null,
    right: ?*TreeNode = null,
};

pub fn createNode(allocator: std.mem.Allocator, value: i64) !*TreeNode {
    const node = try allocator.create(TreeNode);
    node.* = .{ .val = value };
    return node;
}

pub fn freeTree(allocator: std.mem.Allocator, root: ?*TreeNode) void {
    const start = root orelse return;

    var stack = std.ArrayListUnmanaged(*TreeNode){};
    defer stack.deinit(allocator);
    stack.append(allocator, start) catch return;

    while (stack.items.len > 0) {
        const node = stack.pop().?;
        if (node.left) |left| stack.append(allocator, left) catch {};
        if (node.right) |right| stack.append(allocator, right) catch {};
        allocator.destroy(node);
    }
}

const Info = struct {
    is_bst: bool,
    min_val: i64,
    max_val: i64,
    sum: i64,
};

fn solve(node: ?*const TreeNode, best: *i64) Info {
    if (node == null) {
        return .{
            .is_bst = true,
            .min_val = std.math.maxInt(i64),
            .max_val = std.math.minInt(i64),
            .sum = 0,
        };
    }

    const n = node.?;
    const left = solve(n.left, best);
    const right = solve(n.right, best);

    if (left.is_bst and right.is_bst and left.max_val < n.val and n.val < right.min_val) {
        const total = left.sum + right.sum + n.val;
        best.* = @max(best.*, total);
        return .{
            .is_bst = true,
            .min_val = @min(left.min_val, n.val),
            .max_val = @max(right.max_val, n.val),
            .sum = total,
        };
    }

    return .{
        .is_bst = false,
        .min_val = 0,
        .max_val = 0,
        .sum = 0,
    };
}

/// Returns maximum sum among all subtrees that satisfy BST property.
/// Time complexity: O(n), Space complexity: O(h)
pub fn maxSumBst(root: ?*const TreeNode) i64 {
    var best: i64 = 0;
    _ = solve(root, &best);
    return best;
}

fn buildBalanced(allocator: std.mem.Allocator, lo: i64, hi: i64) !?*TreeNode {
    if (lo > hi) return null;

    const mid = lo + @divFloor(hi - lo, 2);
    const node = try createNode(allocator, mid);
    errdefer freeTree(allocator, node);

    node.left = try buildBalanced(allocator, lo, mid - 1);
    node.right = try buildBalanced(allocator, mid + 1, hi);
    return node;
}

test "maximum sum bst: python examples" {
    {
        var t1_ll = TreeNode{ .val = 1 };
        var t1_lr = TreeNode{ .val = 2 };
        var t1_l = TreeNode{ .val = 3, .left = &t1_ll, .right = &t1_lr };
        var t1 = TreeNode{ .val = 4, .left = &t1_l };
        try testing.expectEqual(@as(i64, 2), maxSumBst(&t1));
    }

    {
        var t2_l = TreeNode{ .val = -2 };
        var t2_r = TreeNode{ .val = -5 };
        var t2 = TreeNode{ .val = -4, .left = &t2_l, .right = &t2_r };
        try testing.expectEqual(@as(i64, 0), maxSumBst(&t2));
    }

    {
        var t3_ll = TreeNode{ .val = 2 };
        var t3_lr = TreeNode{ .val = 4 };
        var t3_l = TreeNode{ .val = 4, .left = &t3_ll, .right = &t3_lr };

        var t3_rl = TreeNode{ .val = 2 };
        var t3_rrl = TreeNode{ .val = 4 };
        var t3_rrr = TreeNode{ .val = 6 };
        var t3_rr = TreeNode{ .val = 5, .left = &t3_rrl, .right = &t3_rrr };
        var t3_r = TreeNode{ .val = 3, .left = &t3_rl, .right = &t3_rr };

        var t3 = TreeNode{ .val = 1, .left = &t3_l, .right = &t3_r };
        try testing.expectEqual(@as(i64, 20), maxSumBst(&t3));
    }
}

test "maximum sum bst: boundary" {
    try testing.expectEqual(@as(i64, 0), maxSumBst(null));

    var single = TreeNode{ .val = 9 };
    try testing.expectEqual(@as(i64, 9), maxSumBst(&single));
}

test "maximum sum bst: extreme large balanced bst" {
    const n: i64 = 4_095;
    const root = (try buildBalanced(testing.allocator, 1, n)).?;
    defer freeTree(testing.allocator, root);

    const expected = (n * (n + 1)) / 2;
    try testing.expectEqual(expected, maxSumBst(root));

    // Corrupt one node to break global BST property and ensure result decreases.
    root.left.?.right.?.val = n + 5;
    try testing.expect(maxSumBst(root) < expected);
}
