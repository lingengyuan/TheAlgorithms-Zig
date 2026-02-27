//! Binary Search Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/binary_search_tree.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// A generic binary search tree with dynamic node allocation.
pub fn BinarySearchTree(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Node = struct {
            value: T,
            left: ?*Node = null,
            right: ?*Node = null,
        };

        root: ?*Node = null,
        len: usize = 0,
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Free all nodes recursively.
        pub fn deinit(self: *Self) void {
            freeSubtree(self.allocator, self.root);
            self.root = null;
            self.len = 0;
        }

        fn freeSubtree(allocator: Allocator, node: ?*Node) void {
            const n = node orelse return;
            freeSubtree(allocator, n.left);
            freeSubtree(allocator, n.right);
            allocator.destroy(n);
        }

        /// Insert a value. Duplicates go to the right subtree.
        pub fn insert(self: *Self, value: T) !void {
            self.root = try insertNode(self.allocator, self.root, value);
            self.len += 1;
        }

        fn insertNode(allocator: Allocator, node: ?*Node, value: T) !*Node {
            const n = node orelse {
                const new = try allocator.create(Node);
                new.* = .{ .value = value };
                return new;
            };
            if (value < n.value) {
                n.left = try insertNode(allocator, n.left, value);
            } else {
                n.right = try insertNode(allocator, n.right, value);
            }
            return n;
        }

        /// Search for a value. Returns true if found.
        pub fn search(self: *const Self, value: T) bool {
            return searchNode(self.root, value);
        }

        fn searchNode(node: ?*Node, value: T) bool {
            const n = node orelse return false;
            if (value == n.value) return true;
            if (value < n.value) return searchNode(n.left, value);
            return searchNode(n.right, value);
        }

        /// Get the minimum value, or null if empty.
        pub fn getMin(self: *const Self) ?T {
            var current = self.root orelse return null;
            while (current.left) |left| {
                current = left;
            }
            return current.value;
        }

        /// Get the maximum value, or null if empty.
        pub fn getMax(self: *const Self) ?T {
            var current = self.root orelse return null;
            while (current.right) |right| {
                current = right;
            }
            return current.value;
        }

        /// In-order traversal: collect elements into a sorted slice.
        pub fn inorder(self: *const Self, allocator: Allocator) ![]T {
            const result = try allocator.alloc(T, self.len);
            var idx: usize = 0;
            inorderWalk(self.root, result, &idx);
            return result;
        }

        fn inorderWalk(node: ?*Node, buf: []T, idx: *usize) void {
            const n = node orelse return;
            inorderWalk(n.left, buf, idx);
            buf[idx.*] = n.value;
            idx.* += 1;
            inorderWalk(n.right, buf, idx);
        }

        /// Remove a value. Returns true if removed, false if not found.
        pub fn remove(self: *Self, value: T) bool {
            var found = false;
            self.root = removeNode(self.allocator, self.root, value, &found);
            if (found) self.len -= 1;
            return found;
        }

        fn removeNode(allocator: Allocator, node: ?*Node, value: T, found: *bool) ?*Node {
            const n = node orelse return null;
            if (value < n.value) {
                n.left = removeNode(allocator, n.left, value, found);
                return n;
            } else if (value > n.value) {
                n.right = removeNode(allocator, n.right, value, found);
                return n;
            } else {
                found.* = true;
                // Node to delete found
                if (n.left == null) {
                    const right = n.right;
                    allocator.destroy(n);
                    return right;
                } else if (n.right == null) {
                    const left = n.left;
                    allocator.destroy(n);
                    return left;
                } else {
                    // Two children: replace with in-order successor (min of right subtree)
                    var successor = n.right.?;
                    while (successor.left) |left| {
                        successor = left;
                    }
                    n.value = successor.value;
                    var dummy = false;
                    n.right = removeNode(allocator, n.right, successor.value, &dummy);
                    return n;
                }
            }
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.root == null;
        }
    };
}

// ===== Tests =====

test "bst: insert and inorder" {
    const alloc = testing.allocator;
    var bst = BinarySearchTree(i32).init(alloc);
    defer bst.deinit();

    const values = [_]i32{ 8, 3, 10, 1, 6, 14, 4, 7, 13 };
    for (values) |v| try bst.insert(v);

    const sorted = try bst.inorder(alloc);
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 3, 4, 6, 7, 8, 10, 13, 14 }, sorted);
}

test "bst: search" {
    const alloc = testing.allocator;
    var bst = BinarySearchTree(i32).init(alloc);
    defer bst.deinit();

    for ([_]i32{ 8, 3, 10, 1, 6, 14 }) |v| try bst.insert(v);

    try testing.expect(bst.search(6));
    try testing.expect(bst.search(8));
    try testing.expect(!bst.search(99));
    try testing.expect(!bst.search(-1));
}

test "bst: min and max" {
    const alloc = testing.allocator;
    var bst = BinarySearchTree(i32).init(alloc);
    defer bst.deinit();

    try testing.expectEqual(@as(?i32, null), bst.getMin());
    try testing.expectEqual(@as(?i32, null), bst.getMax());

    for ([_]i32{ 8, 3, 10, 1, 6, 14 }) |v| try bst.insert(v);

    try testing.expectEqual(@as(?i32, 1), bst.getMin());
    try testing.expectEqual(@as(?i32, 14), bst.getMax());
}

test "bst: remove leaf, one child, two children" {
    const alloc = testing.allocator;
    var bst = BinarySearchTree(i32).init(alloc);
    defer bst.deinit();

    for ([_]i32{ 8, 3, 10, 1, 6, 14, 4, 7, 13 }) |v| try bst.insert(v);

    // Remove leaf
    try testing.expect(bst.remove(4));
    try testing.expect(!bst.search(4));

    // Remove node with one child
    try testing.expect(bst.remove(14));
    try testing.expect(!bst.search(14));
    try testing.expect(bst.search(13));

    // Remove node with two children
    try testing.expect(bst.remove(3));
    try testing.expect(!bst.search(3));

    // Not found
    try testing.expect(!bst.remove(999));

    const sorted = try bst.inorder(alloc);
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 6, 7, 8, 10, 13 }, sorted);
}

test "bst: remove all" {
    const alloc = testing.allocator;
    var bst = BinarySearchTree(i32).init(alloc);
    defer bst.deinit();

    for ([_]i32{ 8, 3, 10, 1, 6, 14, 13, 4, 7 }) |v| try bst.insert(v);
    for ([_]i32{ 8, 3, 10, 1, 6, 14, 13, 4, 7 }) |v| {
        try testing.expect(bst.remove(v));
    }
    try testing.expect(bst.isEmpty());
}

test "bst: empty tree" {
    const alloc = testing.allocator;
    var bst = BinarySearchTree(i32).init(alloc);
    defer bst.deinit();

    try testing.expect(bst.isEmpty());
    try testing.expect(!bst.search(1));
    try testing.expect(!bst.remove(1));
}
