//! Binary Tree Mirror - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/binary_tree_mirror.py

const std = @import("std");
const testing = std.testing;

/// Child value `0` means "no child".
pub const TreeEntry = struct {
    key: i64,
    children: [2]i64,
};

fn lessByKey(_: void, a: TreeEntry, b: TreeEntry) bool {
    return a.key < b.key;
}

/// Mirrors a dictionary-represented binary tree rooted at `root`.
/// Time complexity: O(n), Space complexity: O(n)
pub fn binaryTreeMirror(allocator: std.mem.Allocator, binary_tree: []const TreeEntry, root: i64) ![]TreeEntry {
    if (binary_tree.len == 0) return error.EmptyBinaryTree;

    var map = std.AutoHashMap(i64, [2]i64).init(allocator);
    defer map.deinit();

    for (binary_tree) |entry| {
        try map.put(entry.key, entry.children);
    }

    if (!map.contains(root)) return error.RootNotPresent;

    var stack = std.ArrayListUnmanaged(i64){};
    defer stack.deinit(allocator);
    try stack.append(allocator, root);

    while (stack.items.len > 0) {
        const node = stack.pop().?;
        if (node == 0) continue;

        const pair = map.get(node) orelse continue;
        const left_child = pair[0];
        const right_child = pair[1];

        try map.put(node, .{ right_child, left_child });

        if (left_child != 0) try stack.append(allocator, left_child);
        if (right_child != 0) try stack.append(allocator, right_child);
    }

    const out = try allocator.alloc(TreeEntry, map.count());
    var idx: usize = 0;
    var it = map.iterator();
    while (it.next()) |entry| : (idx += 1) {
        out[idx] = .{ .key = entry.key_ptr.*, .children = entry.value_ptr.* };
    }

    std.mem.sort(TreeEntry, out, {}, lessByKey);
    return out;
}

fn expectEntriesEqual(expected: []const TreeEntry, actual: []const TreeEntry) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try testing.expectEqual(e.key, a.key);
        try testing.expectEqualDeep(e.children, a.children);
    }
}

test "binary tree mirror: python examples" {
    {
        const input = [_]TreeEntry{
            .{ .key = 1, .children = .{ 2, 3 } },
            .{ .key = 2, .children = .{ 4, 5 } },
            .{ .key = 3, .children = .{ 6, 7 } },
            .{ .key = 7, .children = .{ 8, 9 } },
        };
        const out = try binaryTreeMirror(testing.allocator, input[0..], 1);
        defer testing.allocator.free(out);

        const expected = [_]TreeEntry{
            .{ .key = 1, .children = .{ 3, 2 } },
            .{ .key = 2, .children = .{ 5, 4 } },
            .{ .key = 3, .children = .{ 7, 6 } },
            .{ .key = 7, .children = .{ 9, 8 } },
        };
        try expectEntriesEqual(expected[0..], out);
    }

    {
        const input = [_]TreeEntry{
            .{ .key = 1, .children = .{ 2, 3 } },
            .{ .key = 2, .children = .{ 4, 5 } },
            .{ .key = 3, .children = .{ 6, 7 } },
            .{ .key = 4, .children = .{ 10, 11 } },
        };
        const out = try binaryTreeMirror(testing.allocator, input[0..], 1);
        defer testing.allocator.free(out);

        const expected = [_]TreeEntry{
            .{ .key = 1, .children = .{ 3, 2 } },
            .{ .key = 2, .children = .{ 5, 4 } },
            .{ .key = 3, .children = .{ 7, 6 } },
            .{ .key = 4, .children = .{ 11, 10 } },
        };
        try expectEntriesEqual(expected[0..], out);
    }
}

test "binary tree mirror: errors" {
    const empty = [_]TreeEntry{};
    try testing.expectError(error.EmptyBinaryTree, binaryTreeMirror(testing.allocator, empty[0..], 5));

    const input = [_]TreeEntry{
        .{ .key = 1, .children = .{ 2, 3 } },
        .{ .key = 2, .children = .{ 4, 5 } },
    };
    try testing.expectError(error.RootNotPresent, binaryTreeMirror(testing.allocator, input[0..], 99));
}

test "binary tree mirror: extreme large dictionary" {
    const internal_nodes: usize = 10_000;
    const entries = try testing.allocator.alloc(TreeEntry, internal_nodes);
    defer testing.allocator.free(entries);

    for (0..internal_nodes) |idx| {
        const node: i64 = @intCast(idx + 1);
        const left: i64 = @intCast((idx + 1) * 2);
        const right: i64 = @intCast((idx + 1) * 2 + 1);
        entries[idx] = .{ .key = node, .children = .{ left, right } };
    }

    const out = try binaryTreeMirror(testing.allocator, entries, 1);
    defer testing.allocator.free(out);

    try testing.expectEqual(internal_nodes, out.len);
    try testing.expectEqual(@as(i64, 1), out[0].key);
    try testing.expectEqualDeep([2]i64{ 3, 2 }, out[0].children);

    const last = out[out.len - 1];
    try testing.expectEqual(@as(i64, @intCast(internal_nodes)), last.key);
    try testing.expectEqualDeep([2]i64{ @as(i64, @intCast(internal_nodes * 2 + 1)), @as(i64, @intCast(internal_nodes * 2)) }, last.children);
}
