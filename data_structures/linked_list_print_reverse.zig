//! Linked List Print Reverse - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/print_reverse.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Node = struct {
    data: i64,
    next: ?*Node,
};

pub const LinkedList = struct {
    allocator: Allocator,
    head: ?*Node,
    tail: ?*Node,
    len_: usize,

    pub fn init(allocator: Allocator) LinkedList {
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

    /// Appends value at tail.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn append(self: *LinkedList, value: i64) !void {
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

    pub fn extend(self: *LinkedList, items: []const i64) !void {
        for (items) |item| {
            try self.append(item);
        }
    }

    pub fn toOwnedSlice(self: *const LinkedList, allocator: Allocator) ![]i64 {
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

/// Creates a linked list from a sequence.
/// Time complexity: O(n), Space complexity: O(n)
pub fn makeLinkedList(allocator: Allocator, elements: []const i64) !LinkedList {
    if (elements.len == 0) return error.EmptyElements;
    var list = LinkedList.init(allocator);
    try list.extend(elements);
    return list;
}

/// Returns a reverse-order string like "73 <- 88 <- 69".
/// Time complexity: O(n), Space complexity: O(n)
pub fn inReverse(list: *const LinkedList, allocator: Allocator) ![]u8 {
    const values = try list.toOwnedSlice(allocator);
    defer allocator.free(values);

    if (values.len == 0) {
        return allocator.alloc(u8, 0);
    }

    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);
    const writer = out.writer(allocator);

    var i = values.len;
    while (i > 0) {
        i -= 1;
        try writer.print("{d}", .{values[i]});
        if (i != 0) {
            try out.appendSlice(allocator, " <- ");
        }
    }

    return try out.toOwnedSlice(allocator);
}

test "linked list print reverse: empty and examples" {
    var empty = LinkedList.init(testing.allocator);
    defer empty.deinit();

    const empty_rev = try inReverse(&empty, testing.allocator);
    defer testing.allocator.free(empty_rev);
    try testing.expectEqualStrings("", empty_rev);

    try testing.expectError(error.EmptyElements, makeLinkedList(testing.allocator, &[_]i64{}));

    var list = try makeLinkedList(testing.allocator, &[_]i64{ 69, 88, 73 });
    defer list.deinit();

    const rev = try inReverse(&list, testing.allocator);
    defer testing.allocator.free(rev);
    try testing.expectEqualStrings("73 <- 88 <- 69", rev);
}

test "linked list print reverse: extreme long sequence" {
    const n: usize = 5_000;
    var values = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(values);

    for (0..n) |i| {
        values[i] = @intCast(i);
    }

    var list = try makeLinkedList(testing.allocator, values);
    defer list.deinit();

    const rev = try inReverse(&list, testing.allocator);
    defer testing.allocator.free(rev);

    try testing.expect(std.mem.startsWith(u8, rev, "4999 <- 4998"));
    try testing.expect(std.mem.endsWith(u8, rev, " <- 1 <- 0"));
}
