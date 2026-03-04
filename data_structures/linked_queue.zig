//! Linked Queue - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/queues/linked_queue.py

const std = @import("std");
const testing = std.testing;

pub fn LinkedQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            data: T,
            next: ?*Node,
        };

        allocator: std.mem.Allocator,
        front: ?*Node,
        rear: ?*Node,
        len_: usize,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .front = null,
                .rear = null,
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
            return self.len_ == 0;
        }

        /// Adds item to queue tail.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn put(self: *Self, item: T) !void {
            const next_len = @addWithOverflow(self.len_, @as(usize, 1));
            if (next_len[1] != 0) return error.Overflow;

            const node = try self.allocator.create(Node);
            node.* = .{ .data = item, .next = null };

            if (self.rear) |rear_node| {
                rear_node.next = node;
            } else {
                self.front = node;
            }
            self.rear = node;
            self.len_ = next_len[0];
        }

        /// Removes and returns queue head.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn get(self: *Self) !T {
            const node = self.front orelse return error.EmptyQueue;
            const value = node.data;

            self.front = node.next;
            if (self.front == null) {
                self.rear = null;
            }

            self.allocator.destroy(node);
            self.len_ -= 1;
            return value;
        }

        pub fn clear(self: *Self) void {
            var node_opt = self.front;
            while (node_opt) |node| {
                const next = node.next;
                self.allocator.destroy(node);
                node_opt = next;
            }
            self.front = null;
            self.rear = null;
            self.len_ = 0;
        }

        pub fn toOwnedSlice(self: *const Self, allocator: std.mem.Allocator) ![]T {
            const out = try allocator.alloc(T, self.len_);
            var node_opt = self.front;
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

test "linked queue: python-style operations" {
    var queue = LinkedQueue([]const u8).init(testing.allocator);
    defer queue.deinit();

    try testing.expect(queue.isEmpty());

    try queue.put("5");
    try queue.put("9");
    try queue.put("python");
    try testing.expect(!queue.isEmpty());

    try testing.expectEqualStrings("5", try queue.get());
    try queue.put("algorithms");
    try testing.expectEqualStrings("9", try queue.get());
    try testing.expectEqualStrings("python", try queue.get());
    try testing.expectEqualStrings("algorithms", try queue.get());

    try testing.expect(queue.isEmpty());
    try testing.expectError(error.EmptyQueue, queue.get());
}

test "linked queue: len and clear" {
    var queue = LinkedQueue(i32).init(testing.allocator);
    defer queue.deinit();

    for (1..6) |i| {
        try queue.put(@intCast(i));
    }
    try testing.expectEqual(@as(usize, 5), queue.len());

    const values = try queue.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, values);

    queue.clear();
    try testing.expect(queue.isEmpty());
    try testing.expectEqual(@as(usize, 0), queue.len());
}

test "linked queue: extreme long enqueue dequeue" {
    var queue = LinkedQueue(i64).init(testing.allocator);
    defer queue.deinit();

    const n: usize = 20_000;

    for (0..n) |i| {
        try queue.put(@intCast(i));
    }
    try testing.expectEqual(n, queue.len());

    for (0..n) |i| {
        try testing.expectEqual(@as(i64, @intCast(i)), try queue.get());
    }

    try testing.expect(queue.isEmpty());
    try testing.expectError(error.EmptyQueue, queue.get());
}
