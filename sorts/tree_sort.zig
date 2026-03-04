//! Tree Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/tree_sort.py

const std = @import("std");
const testing = std.testing;

const Node = struct {
    val: i64,
    left: ?*Node = null,
    right: ?*Node = null,
};

fn insertNode(allocator: std.mem.Allocator, root: *Node, value: i64) !void {
    if (value < root.val) {
        if (root.left) |left| {
            try insertNode(allocator, left, value);
        } else {
            const node = try allocator.create(Node);
            node.* = .{ .val = value };
            root.left = node;
        }
    } else if (value > root.val) {
        if (root.right) |right| {
            try insertNode(allocator, right, value);
        } else {
            const node = try allocator.create(Node);
            node.* = .{ .val = value };
            root.right = node;
        }
    }
    // equal values are ignored, matching Python reference behavior
}

fn inorderCollect(root: ?*Node, out: *std.ArrayListUnmanaged(i64), allocator: std.mem.Allocator) !void {
    if (root) |node| {
        try inorderCollect(node.left, out, allocator);
        try out.append(allocator, node.val);
        try inorderCollect(node.right, out, allocator);
    }
}

fn destroyTree(allocator: std.mem.Allocator, root: ?*Node) void {
    if (root) |node| {
        destroyTree(allocator, node.left);
        destroyTree(allocator, node.right);
        allocator.destroy(node);
    }
}

/// Returns a sorted tuple-like slice via BST in-order traversal.
/// Duplicate values are removed to match Python's Node.insert semantics.
/// Caller owns returned slice.
/// Time complexity: O(n log n) average, O(n²) worst
/// Space complexity: O(n)
pub fn treeSort(allocator: std.mem.Allocator, arr: []const i64) ![]i64 {
    if (arr.len == 0) return try allocator.alloc(i64, 0);

    const root = try allocator.create(Node);
    root.* = .{ .val = arr[0] };
    defer {
        destroyTree(allocator, root);
    }

    for (arr[1..]) |item| try insertNode(allocator, root, item);

    var out = std.ArrayListUnmanaged(i64){};
    defer out.deinit(allocator);
    try inorderCollect(root, &out, allocator);
    return try out.toOwnedSlice(allocator);
}

test "tree sort: python reference examples" {
    const alloc = testing.allocator;

    const r1 = try treeSort(alloc, &[_]i64{});
    defer alloc.free(r1);
    try testing.expectEqual(@as(usize, 0), r1.len);

    const r2 = try treeSort(alloc, &[_]i64{1});
    defer alloc.free(r2);
    try testing.expectEqualSlices(i64, &[_]i64{1}, r2);

    const r3 = try treeSort(alloc, &[_]i64{ 1, 2 });
    defer alloc.free(r3);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2 }, r3);

    const r4 = try treeSort(alloc, &[_]i64{ 5, 2, 7 });
    defer alloc.free(r4);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 5, 7 }, r4);

    const r5 = try treeSort(alloc, &[_]i64{ 5, -4, 9, 2, 7 });
    defer alloc.free(r5);
    try testing.expectEqualSlices(i64, &[_]i64{ -4, 2, 5, 7, 9 }, r5);
}

test "tree sort: duplicate behavior and edge cases" {
    const alloc = testing.allocator;

    const r = try treeSort(alloc, &[_]i64{ 5, 6, 1, -1, 4, 37, 2, 7, 7, 6, 1 });
    defer alloc.free(r);
    // duplicates removed, matching python tree insert semantics
    try testing.expectEqualSlices(i64, &[_]i64{ -1, 1, 2, 4, 5, 6, 7, 37 }, r);
}

test "tree sort: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 30_000;
    const arr = try alloc.alloc(i64, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(88);
    var random = prng.random();
    for (arr) |*v| v.* = random.intRangeAtMost(i64, -20_000, 20_000);

    const out = try treeSort(alloc, arr);
    defer alloc.free(out);
    for (1..out.len) |i| try testing.expect(out[i - 1] <= out[i]);
}
