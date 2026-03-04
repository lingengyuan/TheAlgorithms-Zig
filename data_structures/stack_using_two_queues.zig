//! Stack Using Two Queues - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/stacks/stack_using_two_queues.py

const std = @import("std");
const testing = std.testing;
const queue_mod = @import("queue.zig");

pub fn StackWithQueues(comptime T: type) type {
    return struct {
        const Self = @This();

        main_queue: queue_mod.Queue(T),
        temp_queue: queue_mod.Queue(T),

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .main_queue = queue_mod.Queue(T).init(allocator),
                .temp_queue = queue_mod.Queue(T).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.main_queue.deinit();
            self.temp_queue.deinit();
            self.* = undefined;
        }

        pub fn len(self: *const Self) usize {
            return self.main_queue.count();
        }

        /// Pushes element onto stack top.
        /// Time complexity: O(n), Space complexity: O(1)
        pub fn push(self: *Self, item: T) !void {
            try self.temp_queue.enqueue(item);
            while (!self.main_queue.isEmpty()) {
                try self.temp_queue.enqueue((self.main_queue.dequeue()).?);
            }
            std.mem.swap(queue_mod.Queue(T), &self.main_queue, &self.temp_queue);
        }

        /// Pops element from stack top.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn pop(self: *Self) !T {
            return self.main_queue.dequeue() orelse error.EmptyStack;
        }

        pub fn peek(self: *const Self) ?T {
            return self.main_queue.peek();
        }
    };
}

test "stack using two queues: python sample" {
    var stack = StackWithQueues(i32).init(testing.allocator);
    defer stack.deinit();

    try stack.push(1);
    try stack.push(2);
    try stack.push(3);

    try testing.expectEqual(@as(?i32, 3), stack.peek());
    try testing.expectEqual(@as(i32, 3), try stack.pop());
    try testing.expectEqual(@as(?i32, 2), stack.peek());
    try testing.expectEqual(@as(i32, 2), try stack.pop());
    try testing.expectEqual(@as(i32, 1), try stack.pop());
    try testing.expectEqual(@as(?i32, null), stack.peek());
    try testing.expectError(error.EmptyStack, stack.pop());
}

test "stack using two queues: mixed operations" {
    var stack = StackWithQueues(i64).init(testing.allocator);
    defer stack.deinit();

    try stack.push(10);
    try stack.push(20);
    try testing.expectEqual(@as(i64, 20), try stack.pop());
    try stack.push(30);
    try testing.expectEqual(@as(i64, 30), try stack.pop());
    try testing.expectEqual(@as(i64, 10), try stack.pop());
}

test "stack using two queues: extreme long" {
    var stack = StackWithQueues(i32).init(testing.allocator);
    defer stack.deinit();

    for (0..20_000) |i| {
        try stack.push(@intCast(i));
    }

    var expected: i32 = 19_999;
    while (expected >= 0) : (expected -= 1) {
        try testing.expectEqual(expected, try stack.pop());
    }
}
