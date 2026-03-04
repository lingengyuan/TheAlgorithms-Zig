//! Queue by Two Stacks - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/queues/queue_by_two_stacks.py

const std = @import("std");
const testing = std.testing;
const stack_mod = @import("stack.zig");

pub fn QueueByTwoStacks(comptime T: type) type {
    return struct {
        const Self = @This();

        stack1: stack_mod.Stack(T),
        stack2: stack_mod.Stack(T),

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .stack1 = stack_mod.Stack(T).init(allocator),
                .stack2 = stack_mod.Stack(T).init(allocator),
            };
        }

        pub fn initFromSlice(allocator: std.mem.Allocator, items: []const T) !Self {
            var q = init(allocator);
            for (items) |item| {
                try q.put(item);
            }
            return q;
        }

        pub fn deinit(self: *Self) void {
            self.stack1.deinit();
            self.stack2.deinit();
            self.* = undefined;
        }

        pub fn len(self: *const Self) usize {
            return self.stack1.count() + self.stack2.count();
        }

        pub fn put(self: *Self, item: T) !void {
            try self.stack1.push(item);
        }

        fn refill(self: *Self) !void {
            while (!self.stack1.isEmpty()) {
                try self.stack2.push((self.stack1.pop()).?);
            }
        }

        /// Removes and returns queue front value.
        /// Time complexity: amortized O(1), Space complexity: O(1)
        pub fn get(self: *Self) !T {
            if (self.stack2.isEmpty()) {
                try self.refill();
            }
            return self.stack2.pop() orelse error.EmptyQueue;
        }

        pub fn toOwnedSlice(self: *Self, allocator: std.mem.Allocator) ![]T {
            var out = std.ArrayListUnmanaged(T){};
            errdefer out.deinit(allocator);

            // stack2 top is queue front; iterate reversed storage order.
            var i = self.stack2.count();
            while (i > 0) {
                i -= 1;
                try out.append(allocator, self.stack2.data[i]);
            }
            for (self.stack1.data[0..self.stack1.count()]) |v| {
                try out.append(allocator, v);
            }

            return try out.toOwnedSlice(allocator);
        }
    };
}

test "queue by two stacks: python-like examples" {
    var q = QueueByTwoStacks(i32).init(testing.allocator);
    defer q.deinit();

    try testing.expectEqual(@as(usize, 0), q.len());

    try q.put(10);
    try q.put(20);
    try q.put(30);

    const items = try q.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(items);
    try testing.expectEqualSlices(i32, &[_]i32{ 10, 20, 30 }, items);

    try testing.expectEqual(@as(i32, 10), try q.get());
    try q.put(40);
    try testing.expectEqual(@as(i32, 20), try q.get());
    try testing.expectEqual(@as(i32, 30), try q.get());
    try testing.expectEqual(@as(usize, 1), q.len());
    try testing.expectEqual(@as(i32, 40), try q.get());
    try testing.expectError(error.EmptyQueue, q.get());
}

test "queue by two stacks: init from slice and mixed operations" {
    var q = try QueueByTwoStacks(i64).initFromSlice(testing.allocator, &[_]i64{ 1, 4, 9 });
    defer q.deinit();

    try testing.expectEqual(@as(i64, 1), try q.get());
    try q.put(16);
    try q.put(25);
    try testing.expectEqual(@as(i64, 4), try q.get());
    try testing.expectEqual(@as(i64, 9), try q.get());
    try testing.expectEqual(@as(i64, 16), try q.get());
    try testing.expectEqual(@as(i64, 25), try q.get());
}

test "queue by two stacks: extreme long sequence" {
    var q = QueueByTwoStacks(i32).init(testing.allocator);
    defer q.deinit();

    var in: i32 = 1;
    while (in <= 50_000) : (in += 1) {
        try q.put(in);
    }

    var expected: i32 = 1;
    while (expected <= 50_000) : (expected += 1) {
        try testing.expectEqual(expected, try q.get());
    }

    try testing.expectError(error.EmptyQueue, q.get());
}
