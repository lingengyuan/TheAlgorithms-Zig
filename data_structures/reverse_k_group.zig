//! Reverse K Group - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/reverse_k_group.py

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

    pub fn append(self: *LinkedList, data: i64) !void {
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

    pub fn fromSlice(allocator: std.mem.Allocator, values: []const i64) !LinkedList {
        var list = LinkedList.init(allocator);
        for (values) |v| {
            try list.append(v);
        }
        return list;
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

    /// Reverses nodes within groups of size `group_size`.
    /// Time complexity: O(n), Space complexity: O(1)
    pub fn reverseKNodes(self: *LinkedList, group_size: usize) !void {
        if (group_size == 0) return error.InvalidGroupSize;
        if (group_size == 1 or self.head == null or self.head.?.next == null) return;

        var length = self.len_;
        var dummy = Node{ .data = 0, .next = self.head };
        var previous: *Node = &dummy;

        while (length >= group_size) {
            var current = previous.next.?;
            var next = current.next;
            var i: usize = 1;
            while (i < group_size) : (i += 1) {
                const next_ptr = next.?;
                current.next = next_ptr.next;
                next_ptr.next = previous.next;
                previous.next = next_ptr;
                next = current.next;
            }
            previous = current;
            length -= group_size;
        }

        self.head = dummy.next;

        // Refresh tail.
        self.tail = self.head;
        while (self.tail != null and self.tail.?.next != null) {
            self.tail = self.tail.?.next;
        }
    }
};

test "reverse k group: python example" {
    var list = try LinkedList.fromSlice(testing.allocator, &[_]i64{ 1, 2, 3, 4, 5 });
    defer list.deinit();

    try list.reverseKNodes(2);

    const out = try list.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 1, 4, 3, 5 }, out);
}

test "reverse k group: invalid and boundary cases" {
    var list = try LinkedList.fromSlice(testing.allocator, &[_]i64{ 1, 2, 3, 4, 5 });
    defer list.deinit();

    try testing.expectError(error.InvalidGroupSize, list.reverseKNodes(0));

    try list.reverseKNodes(1);
    {
        const out = try list.toOwnedSlice(testing.allocator);
        defer testing.allocator.free(out);
        try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 4, 5 }, out);
    }

    try list.reverseKNodes(10);
    {
        const out = try list.toOwnedSlice(testing.allocator);
        defer testing.allocator.free(out);
        try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 4, 5 }, out);
    }
}

test "reverse k group: extreme long list" {
    const n: usize = 20_000;
    var values = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(values);
    for (0..n) |i| values[i] = @intCast(i + 1);

    var list = try LinkedList.fromSlice(testing.allocator, values);
    defer list.deinit();

    try list.reverseKNodes(100);

    const out = try list.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(out);

    try testing.expectEqual(@as(i64, 100), out[0]);
    try testing.expectEqual(@as(i64, 99), out[1]);
    try testing.expectEqual(@as(i64, 1), out[99]);
    // Last full group [19901..20000] is reversed.
    try testing.expectEqual(@as(i64, 19_901), out[out.len - 1]);
}
