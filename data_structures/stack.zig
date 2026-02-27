//! Stack (array-based) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/stacks/stack.py

const std = @import("std");
const testing = std.testing;

/// Generic stack backed by a dynamically-resized array.
pub fn Stack(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        data: []T,
        len: usize,
        capacity: usize,
        limit: usize,

        pub fn init(allocator: std.mem.Allocator) Self {
            return initWithLimit(allocator, std.math.maxInt(usize));
        }

        pub fn initWithLimit(allocator: std.mem.Allocator, limit: usize) Self {
            return .{
                .allocator = allocator,
                .data = &[_]T{},
                .len = 0,
                .capacity = 0,
                .limit = limit,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.capacity > 0) {
                self.allocator.free(self.data[0..self.capacity]);
            }
            self.* = undefined;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.len == 0;
        }

        pub fn count(self: *const Self) usize {
            return self.len;
        }

        pub fn size(self: *const Self) usize {
            return self.len;
        }

        pub fn isFull(self: *const Self) bool {
            return self.len >= self.limit;
        }

        pub fn push(self: *Self, value: T) !void {
            if (self.len >= self.limit) return error.StackOverflow;
            try self.ensureCapacity(self.len + 1);
            self.data[self.len] = value;
            self.len += 1;
        }

        pub fn pop(self: *Self) ?T {
            if (self.len == 0) return null;
            self.len -= 1;
            return self.data[self.len];
        }

        pub fn peek(self: *const Self) ?T {
            if (self.len == 0) return null;
            return self.data[self.len - 1];
        }

        pub fn popOrError(self: *Self) !T {
            return self.pop() orelse error.StackUnderflow;
        }

        pub fn peekOrError(self: *const Self) !T {
            return self.peek() orelse error.StackUnderflow;
        }

        fn ensureCapacity(self: *Self, min_capacity: usize) !void {
            if (self.capacity >= min_capacity) return;

            var new_capacity: usize = if (self.capacity == 0) 4 else self.capacity * 2;
            if (new_capacity < min_capacity) new_capacity = min_capacity;

            const new_data = try self.allocator.alloc(T, new_capacity);
            if (self.len > 0) {
                @memcpy(new_data[0..self.len], self.data[0..self.len]);
            }

            if (self.capacity > 0) {
                self.allocator.free(self.data[0..self.capacity]);
            }

            self.data = new_data;
            self.capacity = new_capacity;
        }
    };
}

test "stack: push pop and peek" {
    var stack = Stack(i32).init(testing.allocator);
    defer stack.deinit();

    try testing.expect(stack.isEmpty());
    try testing.expectEqual(@as(?i32, null), stack.peek());

    try stack.push(10);
    try stack.push(20);
    try stack.push(30);

    try testing.expectEqual(@as(usize, 3), stack.count());
    try testing.expectEqual(@as(?i32, 30), stack.peek());
    try testing.expectEqual(@as(?i32, 30), stack.pop());
    try testing.expectEqual(@as(?i32, 20), stack.pop());
    try testing.expectEqual(@as(?i32, 10), stack.pop());
    try testing.expectEqual(@as(?i32, null), stack.pop());
    try testing.expect(stack.isEmpty());
}

test "stack: works with generic type" {
    var stack = Stack(f64).init(testing.allocator);
    defer stack.deinit();

    try stack.push(1.5);
    try stack.push(2.5);
    try testing.expectEqual(@as(?f64, 2.5), stack.peek());
    try testing.expectEqual(@as(?f64, 2.5), stack.pop());
    try testing.expectEqual(@as(?f64, 1.5), stack.pop());
}

test "stack: dynamic growth" {
    var stack = Stack(i32).init(testing.allocator);
    defer stack.deinit();

    for (0..100) |i| {
        try stack.push(@intCast(i));
    }
    try testing.expectEqual(@as(usize, 100), stack.count());

    var expected: i32 = 99;
    while (expected >= 0) : (expected -= 1) {
        try testing.expectEqual(@as(?i32, expected), stack.pop());
    }
}

test "stack: underflow and overflow" {
    var stack = Stack(i32).initWithLimit(testing.allocator, 1);
    defer stack.deinit();

    try testing.expectError(error.StackUnderflow, stack.popOrError());
    try stack.push(10);
    try testing.expect(stack.isFull());
    try testing.expectError(error.StackOverflow, stack.push(20));
    try testing.expectEqual(@as(i32, 10), try stack.peekOrError());
    try testing.expectEqual(@as(i32, 10), try stack.popOrError());
}
