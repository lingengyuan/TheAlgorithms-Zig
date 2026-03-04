//! Linked List Rotate To Right - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/rotate_to_the_right.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    data: i64,
    next: ?*Node,
};

pub const LinkedList = struct {
    allocator: std.mem.Allocator,
    head: ?*Node,
    tail: ?*Node,
    len_: usize,

    pub fn init(allocator: std.mem.Allocator) LinkedList {
        return .{ .allocator = allocator, .head = null, .tail = null, .len_ = 0 };
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

    /// Inserts node at list end.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn insertNode(self: *LinkedList, value: i64) !void {
        const next_len = @addWithOverflow(self.len_, @as(usize, 1));
        if (next_len[1] != 0) return error.Overflow;

        const node = try self.allocator.create(Node);
        node.* = .{ .data = value, .next = null };

        if (self.tail) |tail_node| {
            tail_node.next = node;
        } else {
            self.head = node;
        }
        self.tail = node;
        self.len_ = next_len[0];
    }

    /// Rotates linked list to the right by `places`.
    /// Time complexity: O(n), Space complexity: O(1)
    pub fn rotateToRight(self: *LinkedList, places: usize) !void {
        if (self.head == null) return error.EmptyList;
        if (self.head.?.next == null) return;

        const k = places % self.len_;
        if (k == 0) return;

        const split_index = self.len_ - k;
        var split_node = self.head.?;
        var i: usize = 1;
        while (i < split_index) : (i += 1) {
            split_node = split_node.next.?;
        }

        const new_head = split_node.next.?;
        split_node.next = null;

        self.tail.?.next = self.head;
        self.head = new_head;
        self.tail = split_node;
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

test "linked list rotate to right: empty and single" {
    var empty = LinkedList.init(testing.allocator);
    defer empty.deinit();
    try testing.expectError(error.EmptyList, empty.rotateToRight(1));

    var single = LinkedList.init(testing.allocator);
    defer single.deinit();
    try single.insertNode(1);
    try single.rotateToRight(10);
    const one = try single.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(one);
    try testing.expectEqualSlices(i64, &[_]i64{1}, one);
}

test "linked list rotate to right: python sample" {
    var list = LinkedList.init(testing.allocator);
    defer list.deinit();

    try list.insertNode(1);
    try list.insertNode(2);
    try list.insertNode(3);
    try list.insertNode(4);
    try list.insertNode(5);

    try list.rotateToRight(2);

    const out = try list.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(i64, &[_]i64{ 4, 5, 1, 2, 3 }, out);
}

test "linked list rotate to right: extreme long rotation" {
    var list = LinkedList.init(testing.allocator);
    defer list.deinit();

    const n: usize = 20_000;
    for (1..n + 1) |i| {
        try list.insertNode(@intCast(i));
    }

    try list.rotateToRight(123_456_789);

    const out = try list.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(out);

    const k = 123_456_789 % n;
    const first_expected: i64 = @intCast(n - k + 1);
    try testing.expectEqual(first_expected, out[0]);
    try testing.expectEqual(@as(i64, @intCast(n - k + 2)), out[1]);
}
