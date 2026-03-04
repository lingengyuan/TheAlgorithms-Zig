//! Linked List Has Loop - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/has_loop.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    data: i64,
    next: ?*Node,
};

pub const NodeArena = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(*Node),

    pub fn init(allocator: std.mem.Allocator) NodeArena {
        return .{ .allocator = allocator, .nodes = .{} };
    }

    pub fn deinit(self: *NodeArena) void {
        for (self.nodes.items) |node| {
            self.allocator.destroy(node);
        }
        self.nodes.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn create(self: *NodeArena, value: i64) !*Node {
        const node = try self.allocator.create(Node);
        node.* = .{ .data = value, .next = null };
        try self.nodes.append(self.allocator, node);
        return node;
    }
};

/// Detects whether linked list contains a loop using Floyd's algorithm.
/// Time complexity: O(n), Space complexity: O(1)
pub fn hasLoop(head: ?*Node) bool {
    var slow = head;
    var fast = head;

    while (fast != null and fast.?.next != null) {
        slow = slow.?.next;
        fast = fast.?.next.?.next;
        if (slow == fast) return true;
    }

    return false;
}

test "linked list has loop: python-style examples" {
    var arena = NodeArena.init(testing.allocator);
    defer arena.deinit();

    const n1 = try arena.create(1);
    const n2 = try arena.create(2);
    const n3 = try arena.create(3);
    const n4 = try arena.create(4);
    n1.next = n2;
    n2.next = n3;
    n3.next = n4;

    try testing.expect(!hasLoop(n1));

    n4.next = n2;
    try testing.expect(hasLoop(n1));
}

test "linked list has loop: repeated values are not loop" {
    var arena = NodeArena.init(testing.allocator);
    defer arena.deinit();

    const n1 = try arena.create(5);
    const n2 = try arena.create(6);
    const n3 = try arena.create(5);
    const n4 = try arena.create(6);

    n1.next = n2;
    n2.next = n3;
    n3.next = n4;

    try testing.expect(!hasLoop(n1));

    const single = try arena.create(42);
    try testing.expect(!hasLoop(single));
}

test "linked list has loop: extreme long list" {
    var arena = NodeArena.init(testing.allocator);
    defer arena.deinit();

    const n: usize = 50_000;
    var head: ?*Node = null;
    var prev: ?*Node = null;
    var mid: ?*Node = null;

    for (0..n) |i| {
        const node = try arena.create(@intCast(i));
        if (head == null) head = node;
        if (prev) |p| p.next = node;
        prev = node;
        if (i == n / 2) mid = node;
    }

    try testing.expect(!hasLoop(head));

    prev.?.next = mid;
    try testing.expect(hasLoop(head));
}
