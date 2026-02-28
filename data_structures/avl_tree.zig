//! AVL Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/avl_tree.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Self-balancing BST (AVL) for `i64` keys.
/// Search/insert/remove time complexity: O(log n), space: O(n)
pub const AvlTree = struct {
    const Self = @This();

    pub const Node = struct {
        key: i64,
        height: i32 = 1,
        left: ?*Node = null,
        right: ?*Node = null,
    };

    allocator: Allocator,
    root: ?*Node = null,
    len: usize = 0,

    pub fn init(allocator: Allocator) Self {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        freeNode(self.allocator, self.root);
        self.* = undefined;
    }

    pub fn size(self: *const Self) usize {
        return self.len;
    }

    pub fn height(self: *const Self) i32 {
        return nodeHeight(self.root);
    }

    pub fn contains(self: *const Self, key: i64) bool {
        var cur = self.root;
        while (cur) |n| {
            if (key == n.key) return true;
            cur = if (key < n.key) n.left else n.right;
        }
        return false;
    }

    pub fn insert(self: *Self, key: i64) !bool {
        var inserted = false;
        self.root = try insertNode(self.allocator, self.root, key, &inserted);
        if (inserted) self.len += 1;
        return inserted;
    }

    pub fn remove(self: *Self, key: i64) bool {
        var removed = false;
        self.root = removeNode(self.allocator, self.root, key, &removed);
        if (removed) self.len -= 1;
        return removed;
    }

    pub fn inorder(self: *const Self, allocator: Allocator) ![]i64 {
        const out = try allocator.alloc(i64, self.len);
        var idx: usize = 0;
        inorderWalk(self.root, out, &idx);
        return out;
    }

    fn freeNode(allocator: Allocator, node: ?*Node) void {
        const n = node orelse return;
        freeNode(allocator, n.left);
        freeNode(allocator, n.right);
        allocator.destroy(n);
    }

    fn nodeHeight(node: ?*Node) i32 {
        return if (node) |n| n.height else 0;
    }

    fn updateHeight(node: *Node) void {
        const hl = nodeHeight(node.left);
        const hr = nodeHeight(node.right);
        node.height = @max(hl, hr) + 1;
    }

    fn balanceFactor(node: *Node) i32 {
        return nodeHeight(node.left) - nodeHeight(node.right);
    }

    fn rotateRight(y: *Node) *Node {
        const x = y.left.?;
        const t2 = x.right;

        x.right = y;
        y.left = t2;

        updateHeight(y);
        updateHeight(x);
        return x;
    }

    fn rotateLeft(x: *Node) *Node {
        const y = x.right.?;
        const t2 = y.left;

        y.left = x;
        x.right = t2;

        updateHeight(x);
        updateHeight(y);
        return y;
    }

    fn rebalance(node: *Node) *Node {
        const bf = balanceFactor(node);

        if (bf > 1) {
            if (balanceFactor(node.left.?) < 0) {
                node.left = rotateLeft(node.left.?);
            }
            return rotateRight(node);
        }

        if (bf < -1) {
            if (balanceFactor(node.right.?) > 0) {
                node.right = rotateRight(node.right.?);
            }
            return rotateLeft(node);
        }

        return node;
    }

    fn insertNode(allocator: Allocator, node: ?*Node, key: i64, inserted: *bool) !*Node {
        const n = node orelse {
            const new = try allocator.create(Node);
            new.* = .{ .key = key };
            inserted.* = true;
            return new;
        };

        if (key < n.key) {
            n.left = try insertNode(allocator, n.left, key, inserted);
        } else if (key > n.key) {
            n.right = try insertNode(allocator, n.right, key, inserted);
        } else {
            return n;
        }

        updateHeight(n);
        return rebalance(n);
    }

    fn minNode(node: *Node) *Node {
        var cur = node;
        while (cur.left) |left| cur = left;
        return cur;
    }

    fn removeNode(allocator: Allocator, node: ?*Node, key: i64, removed: *bool) ?*Node {
        const n = node orelse return null;

        if (key < n.key) {
            n.left = removeNode(allocator, n.left, key, removed);
        } else if (key > n.key) {
            n.right = removeNode(allocator, n.right, key, removed);
        } else {
            removed.* = true;

            if (n.left == null or n.right == null) {
                const replacement = if (n.left) |left| left else n.right;
                allocator.destroy(n);
                return replacement;
            }

            const successor = minNode(n.right.?);
            n.key = successor.key;
            var dummy = false;
            n.right = removeNode(allocator, n.right, successor.key, &dummy);
        }

        updateHeight(n);
        return rebalance(n);
    }

    fn inorderWalk(node: ?*Node, out: []i64, idx: *usize) void {
        const n = node orelse return;
        inorderWalk(n.left, out, idx);
        out[idx.*] = n.key;
        idx.* += 1;
        inorderWalk(n.right, out, idx);
    }
};

test "avl: insertion keeps inorder sorted" {
    var tree = AvlTree.init(testing.allocator);
    defer tree.deinit();

    for ([_]i64{ 10, 20, 30, 40, 50, 25 }) |v| {
        _ = try tree.insert(v);
    }

    const out = try tree.inorder(testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(i64, &[_]i64{ 10, 20, 25, 30, 40, 50 }, out);
    try testing.expect(tree.height() <= 3);
}

test "avl: duplicate insert is ignored" {
    var tree = AvlTree.init(testing.allocator);
    defer tree.deinit();

    try testing.expect(try tree.insert(5));
    try testing.expect(!try tree.insert(5));
    try testing.expectEqual(@as(usize, 1), tree.size());
}

test "avl: contains and remove" {
    var tree = AvlTree.init(testing.allocator);
    defer tree.deinit();

    for ([_]i64{ 9, 5, 10, 0, 6, 11, -1, 1, 2 }) |v| {
        _ = try tree.insert(v);
    }

    try testing.expect(tree.contains(6));
    try testing.expect(!tree.contains(99));

    try testing.expect(tree.remove(10));
    try testing.expect(!tree.contains(10));
    try testing.expect(!tree.remove(10));

    const out = try tree.inorder(testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(i64, &[_]i64{ -1, 0, 1, 2, 5, 6, 9, 11 }, out);
}

test "avl: ll/lr/rr/rl rotation scenarios" {
    // LL
    {
        var t = AvlTree.init(testing.allocator);
        defer t.deinit();
        _ = try t.insert(30);
        _ = try t.insert(20);
        _ = try t.insert(10);
        try testing.expectEqual(@as(i64, 20), t.root.?.key);
    }

    // RR
    {
        var t = AvlTree.init(testing.allocator);
        defer t.deinit();
        _ = try t.insert(10);
        _ = try t.insert(20);
        _ = try t.insert(30);
        try testing.expectEqual(@as(i64, 20), t.root.?.key);
    }

    // LR
    {
        var t = AvlTree.init(testing.allocator);
        defer t.deinit();
        _ = try t.insert(30);
        _ = try t.insert(10);
        _ = try t.insert(20);
        try testing.expectEqual(@as(i64, 20), t.root.?.key);
    }

    // RL
    {
        var t = AvlTree.init(testing.allocator);
        defer t.deinit();
        _ = try t.insert(10);
        _ = try t.insert(30);
        _ = try t.insert(20);
        try testing.expectEqual(@as(i64, 20), t.root.?.key);
    }
}
