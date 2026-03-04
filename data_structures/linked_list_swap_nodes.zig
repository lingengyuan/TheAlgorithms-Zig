//! Linked List Swap Nodes - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/swap_nodes.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    data: i64,
    next: ?*Node,
};

pub const LinkedList = struct {
    allocator: std.mem.Allocator,
    head: ?*Node,
    len_: usize,

    pub fn init(allocator: std.mem.Allocator) LinkedList {
        return .{ .allocator = allocator, .head = null, .len_ = 0 };
    }

    pub fn deinit(self: *LinkedList) void {
        var node_opt = self.head;
        while (node_opt) |node| {
            const next = node.next;
            self.allocator.destroy(node);
            node_opt = next;
        }
        self.* = undefined;
    }

    /// Pushes at list head.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn push(self: *LinkedList, value: i64) !void {
        const next_len = @addWithOverflow(self.len_, @as(usize, 1));
        if (next_len[1] != 0) return error.Overflow;

        const node = try self.allocator.create(Node);
        node.* = .{ .data = value, .next = self.head };
        self.head = node;
        self.len_ = next_len[0];
    }

    /// Swaps two node values if both are present.
    /// Time complexity: O(n), Space complexity: O(1)
    pub fn swapNodes(self: *LinkedList, v1: i64, v2: i64) void {
        if (v1 == v2) return;

        var node1 = self.head;
        while (node1) |node| {
            if (node.data == v1) break;
            node1 = node.next;
        }

        var node2 = self.head;
        while (node2) |node| {
            if (node.data == v2) break;
            node2 = node.next;
        }

        if (node1 == null or node2 == null) return;

        const temp = node1.?.data;
        node1.?.data = node2.?.data;
        node2.?.data = temp;
    }

    pub fn toOwnedSlice(self: *const LinkedList, allocator: std.mem.Allocator) ![]i64 {
        const out = try allocator.alloc(i64, self.len_);
        var idx: usize = 0;
        var node_opt = self.head;
        while (node_opt) |node| {
            out[idx] = node.data;
            idx += 1;
            node_opt = node.next;
        }
        return out;
    }
};

test "linked list swap nodes: python-like behavior" {
    var list = LinkedList.init(testing.allocator);
    defer list.deinit();

    try list.push(5);
    try list.push(4);
    try list.push(3);
    try list.push(2);
    try list.push(1);

    list.swapNodes(1, 5);
    const after_swap = try list.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(after_swap);
    try testing.expectEqualSlices(i64, &[_]i64{ 5, 2, 3, 4, 1 }, after_swap);
}

test "linked list swap nodes: absent values and empty list" {
    var list = LinkedList.init(testing.allocator);
    defer list.deinit();

    try list.push(9);
    try list.push(8);
    try list.push(7);
    list.swapNodes(1, 6);

    const unchanged = try list.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(unchanged);
    try testing.expectEqualSlices(i64, &[_]i64{ 7, 8, 9 }, unchanged);

    var empty = LinkedList.init(testing.allocator);
    defer empty.deinit();
    empty.swapNodes(1, 3);
    try testing.expectEqual(@as(?*Node, null), empty.head);
}

test "linked list swap nodes: extreme long list" {
    var list = LinkedList.init(testing.allocator);
    defer list.deinit();

    const n: usize = 40_000;
    for (1..n + 1) |i| {
        try list.push(@intCast(i));
    }

    list.swapNodes(1, @intCast(n));

    const out = try list.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(out);

    try testing.expectEqual(@as(i64, 1), out[0]);
    try testing.expectEqual(@as(i64, @intCast(n)), out[out.len - 1]);
}
