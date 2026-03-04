//! Linked List From Sequence - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/from_sequence.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const LinkedListError = error{EmptyElements};

pub const Node = struct {
    data: i64,
    next: ?*Node,
};

pub const LinkedList = struct {
    allocator: Allocator,
    head: ?*Node,
    len_: usize,

    pub fn deinit(self: *LinkedList) void {
        var node_opt = self.head;
        while (node_opt) |node| {
            const next = node.next;
            self.allocator.destroy(node);
            node_opt = next;
        }
        self.* = undefined;
    }

    pub fn len(self: *const LinkedList) usize {
        return self.len_;
    }

    pub fn toOwnedSlice(self: *const LinkedList, allocator: Allocator) ![]i64 {
        const out = try allocator.alloc(i64, self.len_);
        var node_opt = self.head;
        var idx: usize = 0;
        while (node_opt) |node| {
            out[idx] = node.data;
            idx += 1;
            node_opt = node.next;
        }
        return out;
    }

    pub fn repr(self: *const LinkedList, allocator: Allocator) ![]u8 {
        var out = std.ArrayListUnmanaged(u8){};
        errdefer out.deinit(allocator);
        const writer = out.writer(allocator);

        var node_opt = self.head;
        while (node_opt) |node| {
            try writer.print("<{d}> ---> ", .{node.data});
            node_opt = node.next;
        }
        try out.appendSlice(allocator, "<END>");

        return try out.toOwnedSlice(allocator);
    }
};

/// Creates a linked list from a sequence and returns its head wrapper.
/// Time complexity: O(n), Space complexity: O(n)
pub fn makeLinkedList(allocator: Allocator, elements: []const i64) !LinkedList {
    if (elements.len == 0) return LinkedListError.EmptyElements;

    var head: ?*Node = null;
    var tail: ?*Node = null;

    for (elements) |value| {
        const node = try allocator.create(Node);
        node.* = .{ .data = value, .next = null };

        if (tail) |tail_node| {
            tail_node.next = node;
        } else {
            head = node;
        }
        tail = node;
    }

    return .{
        .allocator = allocator,
        .head = head,
        .len_ = elements.len,
    };
}

test "linked list from sequence: empty input" {
    try testing.expectError(LinkedListError.EmptyElements, makeLinkedList(testing.allocator, &[_]i64{}));
}

test "linked list from sequence: python examples" {
    var one = try makeLinkedList(testing.allocator, &[_]i64{1});
    defer one.deinit();

    const one_repr = try one.repr(testing.allocator);
    defer testing.allocator.free(one_repr);
    try testing.expectEqualStrings("<1> ---> <END>", one_repr);

    var many = try makeLinkedList(testing.allocator, &[_]i64{ 1, 3, 5, 32, 44, 12, 43 });
    defer many.deinit();

    const many_repr = try many.repr(testing.allocator);
    defer testing.allocator.free(many_repr);
    try testing.expectEqualStrings("<1> ---> <3> ---> <5> ---> <32> ---> <44> ---> <12> ---> <43> ---> <END>", many_repr);
}

test "linked list from sequence: extreme long sequence" {
    const n: usize = 20_000;
    var values = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(values);

    for (0..n) |i| {
        values[i] = @intCast(i);
    }

    var list = try makeLinkedList(testing.allocator, values);
    defer list.deinit();

    try testing.expectEqual(n, list.len());

    const out = try list.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(out);

    try testing.expectEqual(@as(i64, 0), out[0]);
    try testing.expectEqual(@as(i64, @intCast(n - 1)), out[out.len - 1]);
}
