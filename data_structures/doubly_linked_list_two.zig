//! Doubly Linked List (Double Ended Variant) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/doubly_linked_list_two.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    data: i64,
    previous: ?*Node = null,
    next: ?*Node = null,
};

pub const LinkedList = struct {
    head: ?*Node = null,
    tail: ?*Node = null,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) LinkedList {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *LinkedList) void {
        var current = self.head;
        while (current) |node| {
            const next = node.next;
            self.allocator.destroy(node);
            current = next;
        }
        self.head = null;
        self.tail = null;
    }

    fn createNode(self: *LinkedList, value: i64) !*Node {
        const node = try self.allocator.create(Node);
        node.* = .{ .data = value };
        return node;
    }

    pub fn isEmpty(self: *const LinkedList) bool {
        return self.head == null;
    }

    pub fn getHeadData(self: *const LinkedList) ?i64 {
        return if (self.head) |h| h.data else null;
    }

    pub fn getTailData(self: *const LinkedList) ?i64 {
        return if (self.tail) |t| t.data else null;
    }

    pub fn contains(self: *const LinkedList, value: i64) bool {
        var current = self.head;
        while (current) |node| {
            if (node.data == value) return true;
            current = node.next;
        }
        return false;
    }

    pub fn setHeadValue(self: *LinkedList, value: i64) !void {
        const node = try self.createNode(value);
        if (self.head == null) {
            self.head = node;
            self.tail = node;
            return;
        }
        self.insertBeforeNode(self.head.?, node);
    }

    pub fn setTailValue(self: *LinkedList, value: i64) !void {
        const node = try self.createNode(value);
        if (self.tail == null) {
            self.head = node;
            self.tail = node;
            return;
        }
        self.insertAfterNode(self.tail.?, node);
    }

    pub fn insert(self: *LinkedList, value: i64) !void {
        if (self.head == null) {
            try self.setHeadValue(value);
        } else {
            try self.setTailValue(value);
        }
    }

    pub fn insertBeforeNode(self: *LinkedList, node: *Node, node_to_insert: *Node) void {
        node_to_insert.next = node;
        node_to_insert.previous = node.previous;

        if (node.previous == null) {
            self.head = node_to_insert;
        } else {
            node.previous.?.next = node_to_insert;
        }

        node.previous = node_to_insert;
    }

    pub fn insertAfterNode(self: *LinkedList, node: *Node, node_to_insert: *Node) void {
        node_to_insert.previous = node;
        node_to_insert.next = node.next;

        if (node.next == null) {
            self.tail = node_to_insert;
        } else {
            node.next.?.previous = node_to_insert;
        }

        node.next = node_to_insert;
    }

    /// 1-based insertion position.
    /// Time complexity: O(n), Space complexity: O(1)
    pub fn insertAtPosition(self: *LinkedList, position: usize, value: i64) !void {
        const new_node = try self.createNode(value);

        var current_position: usize = 1;
        var node = self.head;
        while (node) |n| {
            if (current_position == position) {
                self.insertBeforeNode(n, new_node);
                return;
            }
            current_position += 1;
            node = n.next;
        }

        if (self.tail == null) {
            self.head = new_node;
            self.tail = new_node;
        } else {
            self.insertAfterNode(self.tail.?, new_node);
        }
    }

    pub fn getNode(self: *const LinkedList, value: i64) !*Node {
        var node = self.head;
        while (node) |n| {
            if (n.data == value) return n;
            node = n.next;
        }
        return error.NodeNotFound;
    }

    pub fn deleteValue(self: *LinkedList, value: i64) !bool {
        const node = self.getNode(value) catch return false;

        if (node == self.head) {
            self.head = node.next;
        }

        if (node == self.tail) {
            self.tail = node.previous;
        }

        self.removeNodePointers(node);
        self.allocator.destroy(node);
        return true;
    }

    pub fn removeNodePointers(self: *LinkedList, node: *Node) void {
        _ = self;
        if (node.next) |next| {
            next.previous = node.previous;
        }
        if (node.previous) |prev| {
            prev.next = node.next;
        }
        node.next = null;
        node.previous = null;
    }

    pub fn toSlice(self: *const LinkedList, allocator: std.mem.Allocator) ![]i64 {
        var out = std.ArrayListUnmanaged(i64){};
        errdefer out.deinit(allocator);

        var current = self.head;
        while (current) |node| {
            try out.append(allocator, node.data);
            current = node.next;
        }

        return out.toOwnedSlice(allocator);
    }
};

test "doubly linked list two: python create_linked_list flow" {
    var list = LinkedList.init(testing.allocator);
    defer list.deinit();

    try testing.expect(list.getHeadData() == null);
    try testing.expect(list.getTailData() == null);
    try testing.expect(list.isEmpty());

    try list.insert(10);
    try testing.expectEqual(@as(?i64, 10), list.getHeadData());
    try testing.expectEqual(@as(?i64, 10), list.getTailData());

    try list.insertAtPosition(3, 20);
    try testing.expectEqual(@as(?i64, 10), list.getHeadData());
    try testing.expectEqual(@as(?i64, 20), list.getTailData());

    try list.setHeadValue(1000);
    try list.setTailValue(2000);

    const values = try list.toSlice(testing.allocator);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(i64, &[_]i64{ 1000, 10, 20, 2000 }, values);

    try testing.expect(list.contains(10));
    try testing.expect(try list.deleteValue(10));
    try testing.expect(!list.contains(10));

    try testing.expect(try list.deleteValue(2000));
    try testing.expectEqual(@as(?i64, 20), list.getTailData());

    try testing.expect(try list.deleteValue(1000));
    try testing.expectEqual(@as(?i64, 20), list.getHeadData());

    try testing.expect(try list.deleteValue(20));
    try testing.expect(list.isEmpty());
}

test "doubly linked list two: insert_at_position examples" {
    var list = LinkedList.init(testing.allocator);
    defer list.deinit();

    try list.insertAtPosition(1, 10);
    try list.insertAtPosition(2, 20);
    try list.insertAtPosition(1, 30);
    try list.insertAtPosition(3, 40);
    try list.insertAtPosition(5, 50);

    const values = try list.toSlice(testing.allocator);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(i64, &[_]i64{ 30, 10, 40, 20, 50 }, values);
}

test "doubly linked list two: extreme bulk insert/delete" {
    var list = LinkedList.init(testing.allocator);
    defer list.deinit();

    const n: usize = 20_000;
    var i: usize = 1;
    while (i <= n) : (i += 1) {
        try list.insert(@intCast(i));
    }

    i = 1;
    while (i <= n) : (i += 2) {
        try testing.expect(try list.deleteValue(@intCast(i)));
    }

    const values = try list.toSlice(testing.allocator);
    defer testing.allocator.free(values);

    try testing.expectEqual(@as(usize, n / 2), values.len);
    try testing.expectEqual(@as(i64, 2), values[0]);
    try testing.expectEqual(@as(i64, @intCast(n)), values[values.len - 1]);
}
