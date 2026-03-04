//! Middle Element Of Linked List - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/middle_element_of_linked_list.py

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

    /// Inserts at head.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn push(self: *LinkedList, value: i64) !i64 {
        const next_len = @addWithOverflow(self.len_, @as(usize, 1));
        if (next_len[1] != 0) return error.Overflow;

        const node = try self.allocator.create(Node);
        node.* = .{ .data = value, .next = self.head };
        self.head = node;
        self.len_ = next_len[0];
        return value;
    }

    /// Returns middle element (second middle for even length).
    /// Time complexity: O(n), Space complexity: O(1)
    pub fn middleElement(self: *const LinkedList) ?i64 {
        if (self.head == null) return null;

        var slow = self.head;
        var fast = self.head;

        while (fast != null and fast.?.next != null) {
            fast = fast.?.next.?.next;
            slow = slow.?.next;
        }

        return slow.?.data;
    }
};

test "middle element of linked list: empty" {
    var list = LinkedList.init(testing.allocator);
    defer list.deinit();

    try testing.expectEqual(@as(?i64, null), list.middleElement());
}

test "middle element of linked list: python example" {
    var list = LinkedList.init(testing.allocator);
    defer list.deinit();

    const input = [_]i64{ 5, 6, 8, 8, 10, 12, 17, 7, 3, 20, -20 };
    for (input) |v| {
        _ = try list.push(v);
    }

    try testing.expectEqual(@as(?i64, 12), list.middleElement());
}

test "middle element of linked list: even and extreme" {
    var even_list = LinkedList.init(testing.allocator);
    defer even_list.deinit();

    for (1..101) |i| {
        _ = try even_list.push(@intCast(i));
    }
    // list content is [100..1], second-middle is 50
    try testing.expectEqual(@as(?i64, 50), even_list.middleElement());

    var big = LinkedList.init(testing.allocator);
    defer big.deinit();

    const n: usize = 100_001;
    for (1..n + 1) |i| {
        _ = try big.push(@intCast(i));
    }
    // list content is [100001..1], middle value is 50001
    try testing.expectEqual(@as(?i64, 50_001), big.middleElement());
}
