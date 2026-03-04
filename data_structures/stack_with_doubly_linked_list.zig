//! Stack With Doubly Linked List - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/stacks/stack_with_doubly_linked_list.py

const std = @import("std");
const testing = std.testing;

pub fn DoublyLinkedStack(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            data: T,
            next: ?*Node,
            prev: ?*Node,
        };

        allocator: std.mem.Allocator,
        head: ?*Node,
        len_: usize,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .head = null,
                .len_ = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            self.clear();
            self.* = undefined;
        }

        pub fn len(self: *const Self) usize {
            return self.len_;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.head == null;
        }

        /// Pushes element onto stack top.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn push(self: *Self, data: T) !void {
            const next_len = @addWithOverflow(self.len_, @as(usize, 1));
            if (next_len[1] != 0) return error.Overflow;

            const node = try self.allocator.create(Node);
            node.* = .{
                .data = data,
                .next = self.head,
                .prev = null,
            };

            if (self.head) |old_head| {
                old_head.prev = node;
            }
            self.head = node;
            self.len_ = next_len[0];
        }

        /// Pops stack top. Returns null on empty stack.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn pop(self: *Self) ?T {
            const node = self.head orelse return null;
            const value = node.data;

            self.head = node.next;
            if (self.head) |new_head| {
                new_head.prev = null;
            }

            self.allocator.destroy(node);
            self.len_ -= 1;
            return value;
        }

        pub fn top(self: *const Self) ?T {
            if (self.head) |node| return node.data;
            return null;
        }

        pub fn clear(self: *Self) void {
            var node_opt = self.head;
            while (node_opt) |node| {
                const next = node.next;
                self.allocator.destroy(node);
                node_opt = next;
            }
            self.head = null;
            self.len_ = 0;
        }

        pub fn toOwnedSlice(self: *const Self, allocator: std.mem.Allocator) ![]T {
            const out = try allocator.alloc(T, self.len_);
            var node_opt = self.head;
            var idx: usize = 0;
            while (node_opt) |node| {
                out[idx] = node.data;
                idx += 1;
                node_opt = node.next;
            }
            return out;
        }
    };
}

test "stack with doubly linked list: python-style operations" {
    var stack = DoublyLinkedStack(i32).init(testing.allocator);
    defer stack.deinit();

    try testing.expect(stack.isEmpty());
    try testing.expectEqual(@as(?i32, null), stack.top());

    for (0..4) |i| {
        try stack.push(@intCast(i));
    }

    try testing.expect(!stack.isEmpty());
    try testing.expectEqual(@as(?i32, 3), stack.top());
    try testing.expectEqual(@as(usize, 4), stack.len());
    try testing.expectEqual(@as(?i32, 3), stack.pop());
    try testing.expectEqual(@as(?i32, 2), stack.pop());

    const values = try stack.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 0 }, values);
}

test "stack with doubly linked list: empty pop returns null" {
    var stack = DoublyLinkedStack([]const u8).init(testing.allocator);
    defer stack.deinit();

    try testing.expectEqual(@as(?[]const u8, null), stack.pop());
    try stack.push("zig");
    try testing.expectEqualStrings("zig", stack.pop().?);
    try testing.expectEqual(@as(?[]const u8, null), stack.pop());
}

test "stack with doubly linked list: extreme push pop" {
    var stack = DoublyLinkedStack(i64).init(testing.allocator);
    defer stack.deinit();

    const n: usize = 30_000;
    for (0..n) |i| {
        try stack.push(@intCast(i));
    }

    var expected: i64 = @intCast(n - 1);
    while (expected >= 0) : (expected -= 1) {
        try testing.expectEqual(@as(?i64, expected), stack.pop());
    }

    try testing.expect(stack.isEmpty());
}
