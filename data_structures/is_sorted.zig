//! Is Sorted (BST Validity by Local Rule) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/is_sorted.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    data: f64,
    left: ?*Node = null,
    right: ?*Node = null,
};

/// Returns whether tree satisfies Python local BST ordering checks.
/// Time complexity: O(n), Space complexity: O(h)
pub fn isSorted(root: ?*const Node) bool {
    const node = root orelse return true;

    if (node.left != null and (node.data < node.left.?.data or !isSorted(node.left))) {
        return false;
    }

    return !(node.right != null and (node.data > node.right.?.data or !isSorted(node.right)));
}

fn inorderValues(allocator: std.mem.Allocator, root: ?*const Node) ![]f64 {
    var out = std.ArrayListUnmanaged(f64){};
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
        try out.append(allocator, node.data);
        current = node.right;
    }

    return out.toOwnedSlice(allocator);
}

test "is sorted: python doctest samples" {
    {
        var root = Node{ .data = 2.1 };
        const values1 = try inorderValues(testing.allocator, &root);
        defer testing.allocator.free(values1);
        try testing.expectEqual(@as(usize, 1), values1.len);
        try testing.expect(isSorted(&root));

        var left = Node{ .data = 2.0 };
        root.left = &left;
        var right = Node{ .data = 2.2 };
        root.right = &right;

        const values2 = try inorderValues(testing.allocator, &root);
        defer testing.allocator.free(values2);
        try testing.expectEqualSlices(f64, &[_]f64{ 2.0, 2.1, 2.2 }, values2);
        try testing.expect(isSorted(&root));
    }

    {
        var root = Node{ .data = 0 };
        var left = Node{ .data = 0 };
        var right = Node{ .data = 0 };
        root.left = &left;
        root.right = &right;
        try testing.expect(isSorted(&root));
    }

    {
        var root = Node{ .data = 5 };
        var left = Node{ .data = 1 };
        var right = Node{ .data = 4 };
        var right_left = Node{ .data = 3 };
        root.left = &left;
        root.right = &right;
        right.left = &right_left;
        try testing.expect(!isSorted(&root));
    }
}

test "is sorted: behavior parity for non-global constraint" {
    // Python implementation checks only local parent/child constraints recursively,
    // so this structure is considered sorted there as well.
    var root = Node{ .data = 10 };
    var right = Node{ .data = 12 };
    var right_left = Node{ .data = 9 };
    root.right = &right;
    right.left = &right_left;

    try testing.expect(isSorted(&root));
}

test "is sorted: boundary and extreme" {
    try testing.expect(isSorted(null));

    const n: usize = 50_000;
    const nodes = try testing.allocator.alloc(Node, n);
    defer testing.allocator.free(nodes);

    for (0..n) |i| {
        nodes[i] = .{ .data = @floatFromInt(i) };
    }
    for (0..n) |i| {
        nodes[i].right = if (i + 1 < n) &nodes[i + 1] else null;
    }

    try testing.expect(isSorted(&nodes[0]));

    nodes[n - 2].data = 123456.0;
    nodes[n - 1].data = 1.0;
    try testing.expect(!isSorted(&nodes[0]));
}
