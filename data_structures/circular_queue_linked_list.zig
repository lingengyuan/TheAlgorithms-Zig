//! Circular Queue Linked List - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/queues/circular_queue_linked_list.py

const std = @import("std");
const testing = std.testing;

pub fn CircularQueueLinkedList(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            data: ?T,
            next: *Node,
            prev: *Node,
        };

        allocator: std.mem.Allocator,
        front: *Node,
        rear: *Node,
        len_: usize,
        capacity: usize,

        pub fn init(allocator: std.mem.Allocator, initial_capacity: usize) !Self {
            if (initial_capacity == 0) return error.InvalidCapacity;

            const first_node = try allocator.create(Node);
            first_node.* = .{ .data = null, .next = undefined, .prev = undefined };

            var previous = first_node;
            var i: usize = 1;
            while (i < initial_capacity) : (i += 1) {
                const current = try allocator.create(Node);
                current.* = .{ .data = null, .next = undefined, .prev = previous };
                previous.next = current;
                previous = current;
            }

            previous.next = first_node;
            first_node.prev = previous;

            return .{
                .allocator = allocator,
                .front = first_node,
                .rear = first_node,
                .len_ = 0,
                .capacity = initial_capacity,
            };
        }

        pub fn deinit(self: *Self) void {
            var node = self.front;
            var remaining = self.capacity;
            while (remaining > 0) : (remaining -= 1) {
                const next = node.next;
                self.allocator.destroy(node);
                node = next;
            }
            self.* = undefined;
        }

        pub fn len(self: *const Self) usize {
            return self.len_;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.len_ == 0;
        }

        /// Returns front element without removal.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn first(self: *const Self) !T {
            if (self.len_ == 0) return error.EmptyQueue;
            return self.front.data.?;
        }

        /// Inserts data at queue tail.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn enqueue(self: *Self, data: T) !void {
            if (self.len_ >= self.capacity) return error.FullQueue;
            if (self.len_ != 0) {
                self.rear = self.rear.next;
            }
            self.rear.data = data;
            if (self.len_ == 0) {
                self.front = self.rear;
            }
            self.len_ += 1;
        }

        /// Removes and returns queue head.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn dequeue(self: *Self) !T {
            if (self.len_ == 0) return error.EmptyQueue;

            const value = self.front.data.?;
            self.front.data = null;

            if (self.len_ == 1) {
                self.len_ = 0;
                self.rear = self.front;
                return value;
            }

            self.front = self.front.next;
            self.len_ -= 1;
            return value;
        }
    };
}

test "circular queue linked list: python capacity and empty/full cases" {
    var cq = try CircularQueueLinkedList([]const u8).init(testing.allocator, 2);
    defer cq.deinit();

    try cq.enqueue("a");
    try cq.enqueue("b");
    try testing.expectError(error.FullQueue, cq.enqueue("c"));

    try testing.expectEqualStrings("a", try cq.dequeue());
    try testing.expectEqualStrings("b", try cq.dequeue());
    try testing.expectError(error.EmptyQueue, cq.dequeue());
}

test "circular queue linked list: first and wraparound" {
    var cq = try CircularQueueLinkedList(i32).init(testing.allocator, 4);
    defer cq.deinit();

    try testing.expect(cq.isEmpty());
    try testing.expectError(error.EmptyQueue, cq.first());

    try cq.enqueue(1);
    try cq.enqueue(2);
    try cq.enqueue(3);
    try testing.expectEqual(@as(i32, 1), try cq.first());
    try testing.expectEqual(@as(i32, 1), try cq.dequeue());

    try cq.enqueue(4);
    try cq.enqueue(5);
    try testing.expectEqual(@as(i32, 2), try cq.dequeue());
    try testing.expectEqual(@as(i32, 3), try cq.dequeue());
    try testing.expectEqual(@as(i32, 4), try cq.dequeue());
    try testing.expectEqual(@as(i32, 5), try cq.dequeue());
    try testing.expect(cq.isEmpty());
}

test "circular queue linked list: extreme cycling" {
    var cq = try CircularQueueLinkedList(i64).init(testing.allocator, 128);
    defer cq.deinit();

    for (0..128) |i| {
        try cq.enqueue(@intCast(i));
    }

    var head_expected: i64 = 0;
    var next_in: i64 = 128;
    for (0..20_000) |_| {
        try testing.expectEqual(head_expected, try cq.dequeue());
        try cq.enqueue(next_in);
        head_expected += 1;
        next_in += 1;
    }

    while (!cq.isEmpty()) {
        try testing.expectEqual(head_expected, try cq.dequeue());
        head_expected += 1;
    }
}
