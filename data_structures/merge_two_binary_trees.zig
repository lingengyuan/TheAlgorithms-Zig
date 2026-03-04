//! Merge Two Binary Trees - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/merge_two_binary_trees.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    value: i64,
    left: ?*Node = null,
    right: ?*Node = null,
};

/// Merges `tree2` into `tree1` in-place.
/// Time complexity: O(n), Space complexity: O(h)
pub fn mergeTwoBinaryTrees(tree1: ?*Node, tree2: ?*Node, allocator: std.mem.Allocator) !?*Node {
    if (tree1 == null) return tree2;
    if (tree2 == null) return tree1;

    var stack = std.ArrayListUnmanaged(struct { a: *Node, b: *Node }){};
    defer stack.deinit(allocator);
    try stack.append(allocator, .{ .a = tree1.?, .b = tree2.? });

    while (stack.items.len > 0) {
        const pair = stack.pop().?;

        pair.a.value += pair.b.value;

        if (pair.a.left == null) {
            pair.a.left = pair.b.left;
        } else if (pair.b.left) |b_left| {
            try stack.append(allocator, .{ .a = pair.a.left.?, .b = b_left });
        }

        if (pair.a.right == null) {
            pair.a.right = pair.b.right;
        } else if (pair.b.right) |b_right| {
            try stack.append(allocator, .{ .a = pair.a.right.?, .b = b_right });
        }
    }

    return tree1;
}

pub fn preorderTraversal(allocator: std.mem.Allocator, root: ?*const Node) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);

    const start = root orelse return out.toOwnedSlice(allocator);

    var stack = std.ArrayListUnmanaged(*const Node){};
    defer stack.deinit(allocator);
    try stack.append(allocator, start);

    while (stack.items.len > 0) {
        const node = stack.pop().?;
        try out.append(allocator, node.value);
        if (node.right) |right| try stack.append(allocator, right);
        if (node.left) |left| try stack.append(allocator, left);
    }

    return out.toOwnedSlice(allocator);
}

test "merge two binary trees: python doctest sample" {
    var tree1 = Node{ .value = 5 };
    var t1_6 = Node{ .value = 6 };
    var t1_7 = Node{ .value = 7 };
    var t1_2 = Node{ .value = 2 };

    tree1.left = &t1_6;
    tree1.right = &t1_7;
    t1_6.left = &t1_2;

    var tree2 = Node{ .value = 4 };
    var t2_5 = Node{ .value = 5 };
    var t2_8 = Node{ .value = 8 };
    var t2_1 = Node{ .value = 1 };
    var t2_4 = Node{ .value = 4 };

    tree2.left = &t2_5;
    tree2.right = &t2_8;
    t2_5.right = &t2_1;
    t2_8.right = &t2_4;

    const merged = (try mergeTwoBinaryTrees(&tree1, &tree2, testing.allocator)).?;
    const preorder = try preorderTraversal(testing.allocator, merged);
    defer testing.allocator.free(preorder);

    try testing.expectEqualSlices(i64, &[_]i64{ 9, 11, 2, 1, 15, 4 }, preorder);
}

test "merge two binary trees: boundary" {
    var a = Node{ .value = 1 };
    const r1 = try mergeTwoBinaryTrees(&a, null, testing.allocator);
    try testing.expect(r1 == &a);

    var b = Node{ .value = 2 };
    const r2 = try mergeTwoBinaryTrees(null, &b, testing.allocator);
    try testing.expect(r2 == &b);

    const r3 = try mergeTwoBinaryTrees(null, null, testing.allocator);
    try testing.expect(r3 == null);
}

test "merge two binary trees: extreme long chain" {
    const n: usize = 50_000;

    const nodes1 = try testing.allocator.alloc(Node, n);
    defer testing.allocator.free(nodes1);
    const nodes2 = try testing.allocator.alloc(Node, n);
    defer testing.allocator.free(nodes2);

    for (0..n) |i| {
        nodes1[i] = .{ .value = 1 };
        nodes2[i] = .{ .value = 2 };
    }

    for (0..n) |i| {
        if (i + 1 < n) {
            nodes1[i].right = &nodes1[i + 1];
            nodes2[i].right = &nodes2[i + 1];
        }
    }

    const merged = (try mergeTwoBinaryTrees(&nodes1[0], &nodes2[0], testing.allocator)).?;

    try testing.expectEqual(@as(i64, 3), merged.value);
    try testing.expectEqual(@as(i64, 3), merged.right.?.value);

    var cur = merged;
    var steps: usize = 1;
    while (cur.right) |next| : (steps += 1) {
        cur = next;
    }

    try testing.expectEqual(n, steps);
    try testing.expectEqual(@as(i64, 3), cur.value);
}
