//! Red-Black Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/red_black_tree.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const RedBlackTree = struct {
    const Self = @This();

    const Color = enum {
        red,
        black,
    };

    const Node = struct {
        key: i64,
        color: Color,
        parent: ?*Node,
        left: ?*Node,
        right: ?*Node,
    };

    allocator: Allocator,
    root: ?*Node = null,
    size: usize = 0,

    pub fn init(allocator: Allocator) Self {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        if (self.root) |root| {
            freeNode(self.allocator, root);
        }
        self.* = undefined;
    }

    pub fn len(self: *const Self) usize {
        return self.size;
    }

    pub fn contains(self: *const Self, key: i64) bool {
        var cur = self.root;
        while (cur) |node| {
            if (key == node.key) return true;
            cur = if (key < node.key) node.left else node.right;
        }
        return false;
    }

    pub fn insert(self: *Self, key: i64) !bool {
        if (self.root == null) {
            const node = try self.createNode(key, .black, null);
            self.root = node;
            self.size = 1;
            return true;
        }

        var parent: ?*Node = null;
        var cur = self.root;
        while (cur) |node| {
            parent = node;
            if (key == node.key) return false;
            cur = if (key < node.key) node.left else node.right;
        }

        const p = parent.?;
        const inserted = try self.createNode(key, .red, p);
        if (key < p.key) {
            p.left = inserted;
        } else {
            p.right = inserted;
        }
        self.size += 1;
        self.fixInsert(inserted);
        return true;
    }

    pub fn inorder(self: *const Self, allocator: Allocator) ![]i64 {
        var list = std.ArrayListUnmanaged(i64){};
        defer list.deinit(allocator);

        try inorderWalk(allocator, self.root, &list);
        return try list.toOwnedSlice(allocator);
    }

    pub fn checkColorProperties(self: *const Self) bool {
        if (self.root == null) return true;
        if (self.root.?.color != .black) return false;
        return checkBlackHeight(self.root) != null;
    }

    fn createNode(self: *Self, key: i64, color: Color, parent: ?*Node) !*Node {
        const node = try self.allocator.create(Node);
        node.* = .{
            .key = key,
            .color = color,
            .parent = parent,
            .left = null,
            .right = null,
        };
        return node;
    }

    fn freeNode(allocator: Allocator, node: *Node) void {
        if (node.left) |l| freeNode(allocator, l);
        if (node.right) |r| freeNode(allocator, r);
        allocator.destroy(node);
    }

    fn colorOf(node: ?*Node) Color {
        return if (node) |n| n.color else .black;
    }

    fn fixInsert(self: *Self, start: *Node) void {
        var z = start;

        while (z.parent != null and colorOf(z.parent) == .red) {
            const p = z.parent.?;
            const gp = p.parent orelse break;

            if (gp.left == p) {
                const uncle = gp.right;
                if (colorOf(uncle) == .red) {
                    p.color = .black;
                    uncle.?.color = .black;
                    gp.color = .red;
                    z = gp;
                } else {
                    if (p.right == z) {
                        z = p;
                        self.rotateLeft(z);
                    }
                    const p2 = z.parent.?;
                    const gp2 = p2.parent orelse break;
                    p2.color = .black;
                    gp2.color = .red;
                    self.rotateRight(gp2);
                }
            } else {
                const uncle = gp.left;
                if (colorOf(uncle) == .red) {
                    p.color = .black;
                    uncle.?.color = .black;
                    gp.color = .red;
                    z = gp;
                } else {
                    if (p.left == z) {
                        z = p;
                        self.rotateRight(z);
                    }
                    const p2 = z.parent.?;
                    const gp2 = p2.parent orelse break;
                    p2.color = .black;
                    gp2.color = .red;
                    self.rotateLeft(gp2);
                }
            }
        }

        if (self.root) |r| r.color = .black;
    }

    fn rotateLeft(self: *Self, x: *Node) void {
        const y = x.right orelse return;
        x.right = y.left;
        if (y.left) |yl| yl.parent = x;

        y.parent = x.parent;
        if (x.parent == null) {
            self.root = y;
        } else if (x.parent.?.left == x) {
            x.parent.?.left = y;
        } else {
            x.parent.?.right = y;
        }

        y.left = x;
        x.parent = y;
    }

    fn rotateRight(self: *Self, y: *Node) void {
        const x = y.left orelse return;
        y.left = x.right;
        if (x.right) |xr| xr.parent = y;

        x.parent = y.parent;
        if (y.parent == null) {
            self.root = x;
        } else if (y.parent.?.left == y) {
            y.parent.?.left = x;
        } else {
            y.parent.?.right = x;
        }

        x.right = y;
        y.parent = x;
    }

    fn inorderWalk(allocator: Allocator, node: ?*Node, out: *std.ArrayListUnmanaged(i64)) !void {
        if (node) |n| {
            try inorderWalk(allocator, n.left, out);
            try out.append(allocator, n.key);
            try inorderWalk(allocator, n.right, out);
        }
    }

    fn checkBlackHeight(node: ?*Node) ?usize {
        if (node == null) return 1;
        const n = node.?;

        if (n.color == .red) {
            if (colorOf(n.left) == .red or colorOf(n.right) == .red) return null;
        }

        const left_h = checkBlackHeight(n.left) orelse return null;
        const right_h = checkBlackHeight(n.right) orelse return null;
        if (left_h != right_h) return null;

        return left_h + if (n.color == .black) @as(usize, 1) else @as(usize, 0);
    }
};

test "red black tree: insert and inorder sorted" {
    var tree = RedBlackTree.init(testing.allocator);
    defer tree.deinit();

    for ([_]i64{ 8, -8, 4, 12, 10, 11, 0, -16, 24, 20, 22 }) |v| {
        _ = try tree.insert(v);
    }

    const ordered = try tree.inorder(testing.allocator);
    defer testing.allocator.free(ordered);
    try testing.expectEqualSlices(i64, &[_]i64{ -16, -8, 0, 4, 8, 10, 11, 12, 20, 22, 24 }, ordered);
}

test "red black tree: duplicate insert ignored" {
    var tree = RedBlackTree.init(testing.allocator);
    defer tree.deinit();

    try testing.expect(try tree.insert(5));
    try testing.expect(!(try tree.insert(5)));
    try testing.expectEqual(@as(usize, 1), tree.len());
}

test "red black tree: contains and extremes" {
    var tree = RedBlackTree.init(testing.allocator);
    defer tree.deinit();

    const lo = std.math.minInt(i64);
    const hi = std.math.maxInt(i64);
    _ = try tree.insert(lo);
    _ = try tree.insert(0);
    _ = try tree.insert(hi);

    try testing.expect(tree.contains(lo));
    try testing.expect(tree.contains(hi));
    try testing.expect(!tree.contains(7));
}

test "red black tree: color properties hold after many inserts" {
    var tree = RedBlackTree.init(testing.allocator);
    defer tree.deinit();

    for (0..1_500) |i| {
        const v: i64 = @intCast((i * 97) % 2_000);
        _ = try tree.insert(v - 1_000);
    }

    try testing.expect(tree.checkColorProperties());
}

test "red black tree: root is black when non-empty" {
    var tree = RedBlackTree.init(testing.allocator);
    defer tree.deinit();
    _ = try tree.insert(10);
    _ = try tree.insert(5);
    _ = try tree.insert(15);
    try testing.expect(tree.root != null);
    try testing.expect(tree.checkColorProperties());
}

test "red black tree: empty tree basics" {
    var tree = RedBlackTree.init(testing.allocator);
    defer tree.deinit();

    try testing.expectEqual(@as(usize, 0), tree.len());
    try testing.expect(!tree.contains(1));
    try testing.expect(tree.checkColorProperties());

    const ordered = try tree.inorder(testing.allocator);
    defer testing.allocator.free(ordered);
    try testing.expectEqual(@as(usize, 0), ordered.len);
}
