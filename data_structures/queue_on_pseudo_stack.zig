//! Queue On Pseudo Stack - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/queues/queue_on_pseudo_stack.py

const std = @import("std");
const testing = std.testing;

pub fn QueueOnPseudoStack(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        stack: std.ArrayListUnmanaged(T),

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .stack = .{},
            };
        }

        pub fn deinit(self: *Self) void {
            self.stack.deinit(self.allocator);
            self.* = undefined;
        }

        pub fn size(self: *const Self) usize {
            return self.stack.items.len;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.stack.items.len == 0;
        }

        /// Enqueues item.
        /// Time complexity: O(1) amortized, Space complexity: O(1)
        pub fn put(self: *Self, item: T) !void {
            try self.stack.append(self.allocator, item);
        }

        /// Rotates queue left by `rotation` steps.
        /// Time complexity: O(rotation * n), Space complexity: O(1)
        pub fn rotate(self: *Self, rotation: usize) !void {
            if (self.stack.items.len == 0) return;
            var i: usize = 0;
            while (i < rotation) : (i += 1) {
                const temp = self.stack.orderedRemove(0);
                try self.stack.append(self.allocator, temp);
            }
        }

        /// Dequeues item from queue front.
        /// Time complexity: O(n), Space complexity: O(1)
        pub fn get(self: *Self) !T {
            if (self.stack.items.len == 0) return error.EmptyQueue;

            const len_before = self.stack.items.len;
            try self.rotate(1);

            const dequeued = self.stack.pop().?;
            try self.rotate(len_before - 1);
            return dequeued;
        }

        /// Reads queue front without removing it.
        /// Time complexity: O(n), Space complexity: O(1)
        pub fn front(self: *Self) !T {
            const value = try self.get();
            try self.put(value);
            if (self.stack.items.len > 1) {
                try self.rotate(self.stack.items.len - 1);
            }
            return value;
        }
    };
}

test "queue on pseudo stack: python-like operations" {
    var queue = QueueOnPseudoStack(i32).init(testing.allocator);
    defer queue.deinit();

    try queue.put(1);
    try queue.put(2);
    try queue.put(3);
    try testing.expectEqual(@as(usize, 3), queue.size());

    try testing.expectEqual(@as(i32, 1), try queue.get());
    try testing.expectEqual(@as(i32, 2), try queue.front());
    try testing.expectEqual(@as(i32, 2), try queue.get());
    try testing.expectEqual(@as(i32, 3), try queue.get());

    try testing.expectError(error.EmptyQueue, queue.get());
    try testing.expectError(error.EmptyQueue, queue.front());
}

test "queue on pseudo stack: rotate semantics" {
    var queue = QueueOnPseudoStack(i64).init(testing.allocator);
    defer queue.deinit();

    for (10..14) |i| {
        try queue.put(@intCast(i));
    }

    try queue.rotate(1);
    try testing.expectEqual(@as(i64, 11), try queue.front());
    try queue.rotate(2);
    try testing.expectEqual(@as(i64, 13), try queue.front());
}

test "queue on pseudo stack: extreme long sequence" {
    var queue = QueueOnPseudoStack(i64).init(testing.allocator);
    defer queue.deinit();

    const n: usize = 8_000;
    for (0..n) |i| {
        try queue.put(@intCast(i));
    }

    for (0..n) |i| {
        try testing.expectEqual(@as(i64, @intCast(i)), try queue.get());
    }

    try testing.expect(queue.isEmpty());
}
