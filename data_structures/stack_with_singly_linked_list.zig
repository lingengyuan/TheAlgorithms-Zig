//! Stack With Singly Linked List - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/stacks/stack_with_singly_linked_list.py

const std = @import("std");
const testing = std.testing;

pub fn LinkedStack(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            data: T,
            next: ?*Node,
        };

        allocator: std.mem.Allocator,
        top: ?*Node,
        len_: usize,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .top = null,
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
            return self.top == null;
        }

        /// Pushes item to stack top.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn push(self: *Self, item: T) !void {
            const next_len = @addWithOverflow(self.len_, @as(usize, 1));
            if (next_len[1] != 0) return error.Overflow;

            const node = try self.allocator.create(Node);
            node.* = .{ .data = item, .next = self.top };
            self.top = node;
            self.len_ = next_len[0];
        }

        /// Pops and returns stack top.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn pop(self: *Self) !T {
            const node = self.top orelse return error.EmptyStack;
            const value = node.data;
            self.top = node.next;
            self.allocator.destroy(node);
            self.len_ -= 1;
            return value;
        }

        pub fn peek(self: *const Self) !T {
            const node = self.top orelse return error.EmptyStack;
            return node.data;
        }

        pub fn clear(self: *Self) void {
            var node_opt = self.top;
            while (node_opt) |node| {
                const next = node.next;
                self.allocator.destroy(node);
                node_opt = next;
            }
            self.top = null;
            self.len_ = 0;
        }

        pub fn toOwnedSlice(self: *const Self, allocator: std.mem.Allocator) ![]T {
            const out = try allocator.alloc(T, self.len_);
            var node_opt = self.top;
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

test "stack with singly linked list: python-style operations" {
    var stack = LinkedStack([]const u8).init(testing.allocator);
    defer stack.deinit();

    try testing.expect(stack.isEmpty());

    try stack.push("5");
    try stack.push("9");
    try stack.push("python");
    try testing.expect(!stack.isEmpty());

    try testing.expectEqualStrings("python", try stack.pop());
    try stack.push("algorithms");
    try testing.expectEqualStrings("algorithms", try stack.pop());
    try testing.expectEqualStrings("9", try stack.pop());
    try testing.expectEqualStrings("5", try stack.pop());
    try testing.expect(stack.isEmpty());
    try testing.expectError(error.EmptyStack, stack.pop());
}

test "stack with singly linked list: peek and clear" {
    var stack = LinkedStack(i32).init(testing.allocator);
    defer stack.deinit();

    try stack.push(1);
    try stack.push(2);
    try stack.push(3);
    try testing.expectEqual(@as(i32, 3), try stack.peek());
    try testing.expectEqual(@as(usize, 3), stack.len());

    const values = try stack.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(i32, &[_]i32{ 3, 2, 1 }, values);

    stack.clear();
    try testing.expect(stack.isEmpty());
    try testing.expectError(error.EmptyStack, stack.peek());
}

test "stack with singly linked list: extreme push pop" {
    var stack = LinkedStack(i64).init(testing.allocator);
    defer stack.deinit();

    const n: usize = 25_000;
    for (0..n) |i| {
        try stack.push(@intCast(i));
    }

    var expected: i64 = @intCast(n - 1);
    while (expected >= 0) : (expected -= 1) {
        try testing.expectEqual(expected, try stack.pop());
    }

    try testing.expect(stack.isEmpty());
}
