//! Floor And Ceiling In BST - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/floor_and_ceiling.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    key: i64,
    left: ?*Node = null,
    right: ?*Node = null,
};

pub const FloorCeiling = struct {
    floor: ?i64,
    ceiling: ?i64,
};

/// Finds floor and ceiling for `key` in a BST.
/// Time complexity: O(h), Space complexity: O(1)
pub fn floorCeiling(root: ?*const Node, key: i64) FloorCeiling {
    var floor_val: ?i64 = null;
    var ceiling_val: ?i64 = null;
    var current = root;

    while (current) |node| {
        if (node.key == key) {
            floor_val = node.key;
            ceiling_val = node.key;
            break;
        }

        if (key < node.key) {
            ceiling_val = node.key;
            current = node.left;
        } else {
            floor_val = node.key;
            current = node.right;
        }
    }

    return .{ .floor = floor_val, .ceiling = ceiling_val };
}

fn inorderTraversal(allocator: std.mem.Allocator, root: ?*const Node) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);

    var stack = std.ArrayListUnmanaged(*const Node){};
    defer stack.deinit(allocator);

    var current = root;
    while (current != null or stack.items.len > 0) {
        while (current) |node| {
            try stack.append(allocator, node);
            current = node.left;
        }

        const node = stack.pop().?;
        try out.append(allocator, node.key);
        current = node.right;
    }

    return out.toOwnedSlice(allocator);
}

test "floor and ceiling: python examples" {
    var n10 = Node{ .key = 10 };
    var n5 = Node{ .key = 5 };
    var n20 = Node{ .key = 20 };
    var n3 = Node{ .key = 3 };
    var n7 = Node{ .key = 7 };
    var n15 = Node{ .key = 15 };
    var n25 = Node{ .key = 25 };

    n10.left = &n5;
    n10.right = &n20;
    n5.left = &n3;
    n5.right = &n7;
    n20.left = &n15;
    n20.right = &n25;

    const values = try inorderTraversal(testing.allocator, &n10);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(i64, &[_]i64{ 3, 5, 7, 10, 15, 20, 25 }, values);

    const a = floorCeiling(&n10, 8);
    try testing.expectEqual(@as(?i64, 7), a.floor);
    try testing.expectEqual(@as(?i64, 10), a.ceiling);

    const b = floorCeiling(&n10, 14);
    try testing.expectEqual(@as(?i64, 10), b.floor);
    try testing.expectEqual(@as(?i64, 15), b.ceiling);

    const c = floorCeiling(&n10, -1);
    try testing.expectEqual(@as(?i64, null), c.floor);
    try testing.expectEqual(@as(?i64, 3), c.ceiling);

    const d = floorCeiling(&n10, 30);
    try testing.expectEqual(@as(?i64, 25), d.floor);
    try testing.expectEqual(@as(?i64, null), d.ceiling);
}

test "floor and ceiling: boundary and extreme" {
    const empty = floorCeiling(null, 42);
    try testing.expectEqual(@as(?i64, null), empty.floor);
    try testing.expectEqual(@as(?i64, null), empty.ceiling);

    const n: usize = 100_000;
    const nodes = try testing.allocator.alloc(Node, n);
    defer testing.allocator.free(nodes);

    for (0..n) |i| {
        nodes[i] = .{ .key = @intCast(i * 2) };
    }
    for (0..n) |i| {
        nodes[i].right = if (i + 1 < n) &nodes[i + 1] else null;
    }

    const mid = floorCeiling(&nodes[0], 155_555);
    try testing.expectEqual(@as(?i64, 155_554), mid.floor);
    try testing.expectEqual(@as(?i64, 155_556), mid.ceiling);

    const low = floorCeiling(&nodes[0], -100);
    try testing.expectEqual(@as(?i64, null), low.floor);
    try testing.expectEqual(@as(?i64, 0), low.ceiling);

    const high = floorCeiling(&nodes[0], 500_000);
    try testing.expectEqual(@as(?i64, @intCast((n - 1) * 2)), high.floor);
    try testing.expectEqual(@as(?i64, null), high.ceiling);
}
