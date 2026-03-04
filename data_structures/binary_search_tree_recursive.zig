//! Binary Search Tree (Recursive) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/binary_search_tree_recursive.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    label: i64,
    parent: ?*Node,
    left: ?*Node = null,
    right: ?*Node = null,
};

pub const BinarySearchTree = struct {
    root: ?*Node = null,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) BinarySearchTree {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *BinarySearchTree) void {
        freeTree(self.allocator, self.root);
        self.root = null;
    }

    /// Empties the tree.
    /// Time complexity: O(n), Space complexity: O(h)
    pub fn empty(self: *BinarySearchTree) void {
        freeTree(self.allocator, self.root);
        self.root = null;
    }

    pub fn isEmpty(self: *const BinarySearchTree) bool {
        return self.root == null;
    }

    fn putRec(self: *BinarySearchTree, node: ?*Node, label: i64, parent: ?*Node) !*Node {
        if (node == null) {
            const created = try self.allocator.create(Node);
            created.* = .{ .label = label, .parent = parent };
            return created;
        }

        if (label < node.?.label) {
            node.?.left = try self.putRec(node.?.left, label, node);
        } else if (label > node.?.label) {
            node.?.right = try self.putRec(node.?.right, label, node);
        } else {
            return error.DuplicateLabel;
        }

        return node.?;
    }

    /// Inserts a label.
    /// Time complexity: O(h), Space complexity: O(h)
    pub fn put(self: *BinarySearchTree, label: i64) !void {
        self.root = try self.putRec(self.root, label, null);
    }

    fn searchRec(node: ?*Node, label: i64) !*Node {
        if (node == null) return error.NodeNotFound;

        if (label < node.?.label) return searchRec(node.?.left, label);
        if (label > node.?.label) return searchRec(node.?.right, label);
        return node.?;
    }

    /// Searches for a label.
    /// Time complexity: O(h), Space complexity: O(h)
    pub fn search(self: *const BinarySearchTree, label: i64) !*Node {
        return searchRec(self.root, label);
    }

    pub fn exists(self: *const BinarySearchTree, label: i64) bool {
        _ = self.search(label) catch return false;
        return true;
    }

    fn reassignNodes(self: *BinarySearchTree, node: *Node, new_children: ?*Node) void {
        if (new_children) |child| {
            child.parent = node.parent;
        }

        if (node.parent) |p| {
            if (p.right == node) {
                p.right = new_children;
            } else {
                p.left = new_children;
            }
        } else {
            self.root = new_children;
        }
    }

    fn getLowestNode(self: *BinarySearchTree, node: *Node) *Node {
        if (node.left) |left| {
            return self.getLowestNode(left);
        }

        const lowest = node;
        self.reassignNodes(node, node.right);
        return lowest;
    }

    /// Removes a node by label.
    /// Time complexity: O(h), Space complexity: O(h)
    pub fn remove(self: *BinarySearchTree, label: i64) !void {
        const node = try self.search(label);

        if (node.right != null and node.left != null) {
            const lowest = self.getLowestNode(node.right.?);
            lowest.left = node.left;
            lowest.right = node.right;

            if (node.left) |left| left.parent = lowest;
            if (node.right) |right| right.parent = lowest;

            self.reassignNodes(node, lowest);
        } else if (node.left != null) {
            self.reassignNodes(node, node.left);
        } else if (node.right != null) {
            self.reassignNodes(node, node.right);
        } else {
            self.reassignNodes(node, null);
        }

        self.allocator.destroy(node);
    }

    /// Returns maximum label.
    /// Time complexity: O(h), Space complexity: O(1)
    pub fn getMaxLabel(self: *const BinarySearchTree) !i64 {
        var node = self.root orelse return error.EmptyTree;
        while (node.right) |right| node = right;
        return node.label;
    }

    /// Returns minimum label.
    /// Time complexity: O(h), Space complexity: O(1)
    pub fn getMinLabel(self: *const BinarySearchTree) !i64 {
        var node = self.root orelse return error.EmptyTree;
        while (node.left) |left| node = left;
        return node.label;
    }

    pub fn inorderTraversal(self: *const BinarySearchTree, allocator: std.mem.Allocator) ![]i64 {
        var out = std.ArrayListUnmanaged(i64){};
        errdefer out.deinit(allocator);

        var stack = std.ArrayListUnmanaged(*const Node){};
        defer stack.deinit(allocator);

        var current = self.root;
        while (current != null or stack.items.len > 0) {
            while (current) |n| {
                try stack.append(allocator, n);
                current = n.left;
            }

            const n = stack.pop().?;
            try out.append(allocator, n.label);
            current = n.right;
        }

        return out.toOwnedSlice(allocator);
    }

    pub fn preorderTraversal(self: *const BinarySearchTree, allocator: std.mem.Allocator) ![]i64 {
        var out = std.ArrayListUnmanaged(i64){};
        errdefer out.deinit(allocator);

        const start = self.root orelse return out.toOwnedSlice(allocator);

        var stack = std.ArrayListUnmanaged(*const Node){};
        defer stack.deinit(allocator);
        try stack.append(allocator, start);

        while (stack.items.len > 0) {
            const node = stack.pop().?;
            try out.append(allocator, node.label);
            if (node.right) |right| try stack.append(allocator, right);
            if (node.left) |left| try stack.append(allocator, left);
        }

        return out.toOwnedSlice(allocator);
    }

    pub fn postorderTraversal(self: *const BinarySearchTree, allocator: std.mem.Allocator) ![]i64 {
        var out = std.ArrayListUnmanaged(i64){};
        errdefer out.deinit(allocator);

        const start = self.root orelse return out.toOwnedSlice(allocator);

        var s1 = std.ArrayListUnmanaged(*const Node){};
        defer s1.deinit(allocator);
        var s2 = std.ArrayListUnmanaged(*const Node){};
        defer s2.deinit(allocator);

        try s1.append(allocator, start);
        while (s1.items.len > 0) {
            const node = s1.pop().?;
            try s2.append(allocator, node);
            if (node.left) |left| try s1.append(allocator, left);
            if (node.right) |right| try s1.append(allocator, right);
        }

        while (s2.items.len > 0) {
            try out.append(allocator, s2.pop().?.label);
        }

        return out.toOwnedSlice(allocator);
    }
};

pub fn freeTree(allocator: std.mem.Allocator, root: ?*Node) void {
    const start = root orelse return;

    var stack = std.ArrayListUnmanaged(*Node){};
    defer stack.deinit(allocator);
    stack.append(allocator, start) catch return;

    while (stack.items.len > 0) {
        const node = stack.pop().?;
        if (node.left) |left| stack.append(allocator, left) catch {};
        if (node.right) |right| stack.append(allocator, right) catch {};
        allocator.destroy(node);
    }
}

fn insertBalanced(tree: *BinarySearchTree, lo: i64, hi: i64) !void {
    if (lo > hi) return;

    const mid = lo + @divFloor(hi - lo, 2);
    try tree.put(mid);
    try insertBalanced(tree, lo, mid - 1);
    try insertBalanced(tree, mid + 1, hi);
}

test "binary search tree recursive: put/search/exists" {
    var t = BinarySearchTree.init(testing.allocator);
    defer t.deinit();

    try testing.expect(t.isEmpty());

    try t.put(8);
    try testing.expect(!t.isEmpty());
    try testing.expect(t.root != null);
    try testing.expect(t.root.?.parent == null);
    try testing.expectEqual(@as(i64, 8), t.root.?.label);

    try t.put(10);
    try testing.expect(t.root.?.right != null);
    try testing.expect(t.root.?.right.?.parent == t.root);
    try testing.expectEqual(@as(i64, 10), t.root.?.right.?.label);

    try t.put(3);
    try testing.expect(t.root.?.left != null);
    try testing.expect(t.root.?.left.?.parent == t.root);
    try testing.expectEqual(@as(i64, 3), t.root.?.left.?.label);

    const n = try t.search(8);
    try testing.expectEqual(@as(i64, 8), n.label);
    try testing.expectError(error.NodeNotFound, t.search(42));
    try testing.expectError(error.DuplicateLabel, t.put(8));

    try testing.expect(t.exists(10));
    try testing.expect(!t.exists(99));
}

test "binary search tree recursive: remove and min/max" {
    var t = BinarySearchTree.init(testing.allocator);
    defer t.deinit();

    try testing.expectError(error.EmptyTree, t.getMaxLabel());
    try testing.expectError(error.EmptyTree, t.getMinLabel());

    try t.put(8);
    try t.put(10);
    try t.put(9);

    try testing.expectEqual(@as(i64, 10), try t.getMaxLabel());
    try testing.expectEqual(@as(i64, 8), try t.getMinLabel());

    const inorder_before = try t.inorderTraversal(testing.allocator);
    defer testing.allocator.free(inorder_before);
    try testing.expectEqualSlices(i64, &[_]i64{ 8, 9, 10 }, inorder_before);

    try t.remove(8);
    try testing.expect(t.root != null);
    try testing.expectEqual(@as(i64, 10), t.root.?.label);
    try testing.expectError(error.NodeNotFound, t.remove(8));

    const inorder_after = try t.inorderTraversal(testing.allocator);
    defer testing.allocator.free(inorder_after);
    try testing.expectEqualSlices(i64, &[_]i64{ 9, 10 }, inorder_after);

    t.empty();
    try testing.expect(t.isEmpty());
}

test "binary search tree recursive: traversal outputs" {
    var t = BinarySearchTree.init(testing.allocator);
    defer t.deinit();

    try t.put(5);
    try t.put(3);
    try t.put(7);
    try t.put(2);
    try t.put(4);
    try t.put(6);
    try t.put(8);

    const inorder = try t.inorderTraversal(testing.allocator);
    defer testing.allocator.free(inorder);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 3, 4, 5, 6, 7, 8 }, inorder);

    const preorder = try t.preorderTraversal(testing.allocator);
    defer testing.allocator.free(preorder);
    try testing.expectEqualSlices(i64, &[_]i64{ 5, 3, 2, 4, 7, 6, 8 }, preorder);

    const postorder = try t.postorderTraversal(testing.allocator);
    defer testing.allocator.free(postorder);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 4, 3, 6, 8, 7, 5 }, postorder);
}

test "binary search tree recursive: extreme balanced operations" {
    var t = BinarySearchTree.init(testing.allocator);
    defer t.deinit();

    const n: i64 = 4_095;
    try insertBalanced(&t, 1, n);

    try testing.expectEqual(@as(i64, 1), try t.getMinLabel());
    try testing.expectEqual(n, try t.getMaxLabel());

    var i: i64 = 1;
    while (i <= n) : (i += 2) {
        try t.remove(i);
    }

    i = 1;
    while (i <= n) : (i += 1) {
        if (@mod(i, 2) == 1) {
            try testing.expect(!t.exists(i));
        } else {
            try testing.expect(t.exists(i));
        }
    }

    const inorder = try t.inorderTraversal(testing.allocator);
    defer testing.allocator.free(inorder);
    try testing.expectEqual(@as(usize, @intCast(n / 2)), inorder.len);
    try testing.expectEqual(@as(i64, 2), inorder[0]);
    try testing.expectEqual(n - 1, inorder[inorder.len - 1]);
}
