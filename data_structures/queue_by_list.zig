//! Queue By List - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/queues/queue_by_list.py

const std = @import("std");
const testing = std.testing;

pub fn QueueByList(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        entries: std.ArrayListUnmanaged(T),

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .entries = .{},
            };
        }

        pub fn initFromSlice(allocator: std.mem.Allocator, items: []const T) !Self {
            var queue = init(allocator);
            try queue.entries.appendSlice(allocator, items);
            return queue;
        }

        pub fn deinit(self: *Self) void {
            self.entries.deinit(self.allocator);
            self.* = undefined;
        }

        pub fn len(self: *const Self) usize {
            return self.entries.items.len;
        }

        /// Adds item to queue tail.
        /// Time complexity: O(1) amortized, Space complexity: O(1)
        pub fn put(self: *Self, item: T) !void {
            try self.entries.append(self.allocator, item);
        }

        /// Removes and returns queue head.
        /// Time complexity: O(n), Space complexity: O(1)
        pub fn get(self: *Self) !T {
            if (self.entries.items.len == 0) return error.EmptyQueue;
            return self.entries.orderedRemove(0);
        }

        pub fn rotate(self: *Self, rotation: usize) !void {
            var i: usize = 0;
            while (i < rotation and self.entries.items.len > 0) : (i += 1) {
                const value = self.entries.orderedRemove(0);
                try self.entries.append(self.allocator, value);
            }
        }

        pub fn getFront(self: *const Self) !T {
            if (self.entries.items.len == 0) return error.EmptyQueue;
            return self.entries.items[0];
        }

        pub fn toOwnedSlice(self: *const Self, allocator: std.mem.Allocator) ![]T {
            const out = try allocator.alloc(T, self.entries.items.len);
            @memcpy(out, self.entries.items);
            return out;
        }
    };
}

test "queue by list: python construction and basic operations" {
    var queue = try QueueByList(i32).initFromSlice(testing.allocator, &[_]i32{ 10, 20, 30 });
    defer queue.deinit();

    try testing.expectEqual(@as(usize, 3), queue.len());
    try testing.expectEqual(@as(i32, 10), try queue.get());
    try queue.put(40);
    try testing.expectEqual(@as(i32, 20), try queue.get());
    try testing.expectEqual(@as(i32, 30), try queue.get());
    try testing.expectEqual(@as(usize, 1), queue.len());
    try testing.expectEqual(@as(i32, 40), try queue.get());
    try testing.expectError(error.EmptyQueue, queue.get());
}

test "queue by list: rotate and get front" {
    var queue = try QueueByList(i32).initFromSlice(testing.allocator, &[_]i32{ 10, 20, 30, 40 });
    defer queue.deinit();

    try testing.expectEqual(@as(i32, 10), try queue.getFront());
    try queue.rotate(1);
    try testing.expectEqual(@as(i32, 20), try queue.getFront());
    try queue.rotate(2);

    const values = try queue.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(i32, &[_]i32{ 40, 10, 20, 30 }, values);
}

test "queue by list: extreme repeated rotate and dequeue" {
    var queue = QueueByList(i64).init(testing.allocator);
    defer queue.deinit();

    const n: usize = 15_000;
    for (0..n) |i| {
        try queue.put(@intCast(i));
    }

    try queue.rotate(5_000);

    try testing.expectEqual(@as(i64, 5_000), try queue.get());

    for (0..n - 1) |_| {
        _ = try queue.get();
    }

    try testing.expectError(error.EmptyQueue, queue.get());
}
