//! Floyd's Cycle Detection - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/floyds_cycle_detection.py

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
        // Assumes no cycle at deinit time.
        var node_opt = self.head;
        while (node_opt) |node| {
            const next = node.next;
            self.allocator.destroy(node);
            node_opt = next;
        }
        self.* = undefined;
    }

    /// Adds node at list tail.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn addNode(self: *LinkedList, data: i64) !void {
        const next_len = @addWithOverflow(self.len_, @as(usize, 1));
        if (next_len[1] != 0) return error.Overflow;

        const node = try self.allocator.create(Node);
        node.* = .{ .data = data, .next = null };

        if (self.tail) |tail| {
            tail.next = node;
        } else {
            self.head = node;
        }

        self.tail = node;
        self.len_ = next_len[0];
    }

    /// Detects whether linked list contains a cycle.
    /// Time complexity: O(n), Space complexity: O(1)
    pub fn detectCycle(self: *const LinkedList) bool {
        if (self.head == null) return false;

        var slow = self.head;
        var fast = self.head;

        while (fast != null and fast.?.next != null) {
            slow = slow.?.next;
            fast = fast.?.next.?.next;
            if (slow == fast) return true;
        }

        return false;
    }
};

test "floyd cycle detection: empty and simple list" {
    var list = LinkedList.init(testing.allocator);
    defer list.deinit();

    try testing.expect(!list.detectCycle());

    try list.addNode(1);
    try list.addNode(2);
    try list.addNode(3);
    try list.addNode(4);
    try testing.expect(!list.detectCycle());
}

test "floyd cycle detection: python sample cycle" {
    var list = LinkedList.init(testing.allocator);
    defer list.deinit();

    try list.addNode(1);
    try list.addNode(2);
    try list.addNode(3);
    try list.addNode(4);

    try testing.expect(!list.detectCycle());

    // Create cycle: tail -> second node.
    list.tail.?.next = list.head.?.next;
    try testing.expect(list.detectCycle());

    // Break cycle before deinit.
    list.tail.?.next = null;
}

test "floyd cycle detection: extreme long list" {
    var list = LinkedList.init(testing.allocator);
    defer list.deinit();

    const n: usize = 100_000;
    for (0..n) |i| {
        try list.addNode(@intCast(i));
    }

    try testing.expect(!list.detectCycle());

    // Create a large cycle: tail -> head.next.
    list.tail.?.next = list.head.?.next;
    try testing.expect(list.detectCycle());

    // Break cycle before deinit.
    list.tail.?.next = null;
}
