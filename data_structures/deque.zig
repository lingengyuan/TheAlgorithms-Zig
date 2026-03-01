//! Deque (Double-Ended Queue) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/deque_doubly.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Deque = struct {
    const Self = @This();

    allocator: Allocator,
    buffer: []i64,
    head: usize,
    len_: usize,

    pub fn init(allocator: Allocator) !Self {
        const buffer = try allocator.alloc(i64, 8);
        return .{
            .allocator = allocator,
            .buffer = buffer,
            .head = 0,
            .len_ = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.buffer);
        self.* = undefined;
    }

    pub fn len(self: *const Self) usize {
        return self.len_;
    }

    pub fn isEmpty(self: *const Self) bool {
        return self.len_ == 0;
    }

    pub fn pushFront(self: *Self, value: i64) !void {
        const needed = @addWithOverflow(self.len_, @as(usize, 1));
        if (needed[1] != 0) return error.Overflow;
        try self.ensureCapacity(needed[0]);
        self.head = (self.head + self.buffer.len - 1) % self.buffer.len;
        self.buffer[self.head] = value;
        self.len_ = needed[0];
    }

    pub fn pushBack(self: *Self, value: i64) !void {
        const needed = @addWithOverflow(self.len_, @as(usize, 1));
        if (needed[1] != 0) return error.Overflow;
        try self.ensureCapacity(needed[0]);
        const idx = self.index(self.len_);
        self.buffer[idx] = value;
        self.len_ = needed[0];
    }

    pub fn popFront(self: *Self) ?i64 {
        if (self.len_ == 0) return null;
        const value = self.buffer[self.head];
        self.head = (self.head + 1) % self.buffer.len;
        self.len_ -= 1;
        return value;
    }

    pub fn popBack(self: *Self) ?i64 {
        if (self.len_ == 0) return null;
        const idx = self.index(self.len_ - 1);
        const value = self.buffer[idx];
        self.len_ -= 1;
        return value;
    }

    pub fn peekFront(self: *const Self) ?i64 {
        if (self.len_ == 0) return null;
        return self.buffer[self.head];
    }

    pub fn peekBack(self: *const Self) ?i64 {
        if (self.len_ == 0) return null;
        return self.buffer[self.index(self.len_ - 1)];
    }

    fn index(self: *const Self, offset: usize) usize {
        return (self.head + offset) % self.buffer.len;
    }

    fn ensureCapacity(self: *Self, min_capacity: usize) !void {
        if (self.buffer.len >= min_capacity) return;

        var new_cap = self.buffer.len;
        while (new_cap < min_capacity) {
            const doubled = @mulWithOverflow(new_cap, @as(usize, 2));
            if (doubled[1] != 0) return error.Overflow;
            new_cap = doubled[0];
        }
        const new_buffer = try self.allocator.alloc(i64, new_cap);

        for (0..self.len_) |i| {
            new_buffer[i] = self.buffer[self.index(i)];
        }

        self.allocator.free(self.buffer);
        self.buffer = new_buffer;
        self.head = 0;
    }
};

test "deque: push and pop both ends" {
    var dq = try Deque.init(testing.allocator);
    defer dq.deinit();

    try dq.pushBack(1);
    try dq.pushBack(2);
    try dq.pushFront(0);
    try testing.expectEqual(@as(?i64, 0), dq.popFront());
    try testing.expectEqual(@as(?i64, 2), dq.popBack());
    try testing.expectEqual(@as(?i64, 1), dq.popFront());
    try testing.expectEqual(@as(?i64, null), dq.popFront());
}

test "deque: peek operations" {
    var dq = try Deque.init(testing.allocator);
    defer dq.deinit();

    try testing.expectEqual(@as(?i64, null), dq.peekFront());
    try testing.expectEqual(@as(?i64, null), dq.peekBack());

    try dq.pushBack(10);
    try dq.pushBack(20);
    try dq.pushFront(5);
    try testing.expectEqual(@as(?i64, 5), dq.peekFront());
    try testing.expectEqual(@as(?i64, 20), dq.peekBack());
}

test "deque: grows and preserves order under wrap-around" {
    var dq = try Deque.init(testing.allocator);
    defer dq.deinit();

    for (0..16) |i| {
        try dq.pushBack(@intCast(i));
    }

    for (0..6) |_| {
        _ = dq.popFront();
    }

    for (100..120) |i| {
        try dq.pushBack(@intCast(i));
    }

    try dq.pushFront(-1);
    try dq.pushFront(-2);

    try testing.expectEqual(@as(?i64, -2), dq.popFront());
    try testing.expectEqual(@as(?i64, -1), dq.popFront());
    try testing.expectEqual(@as(?i64, 6), dq.popFront());
    try testing.expectEqual(@as(?i64, 119), dq.popBack());
}

test "deque: single element edge cases" {
    var dq = try Deque.init(testing.allocator);
    defer dq.deinit();

    try dq.pushFront(42);
    try testing.expectEqual(@as(usize, 1), dq.len());
    try testing.expectEqual(@as(?i64, 42), dq.peekFront());
    try testing.expectEqual(@as(?i64, 42), dq.peekBack());
    try testing.expectEqual(@as(?i64, 42), dq.popBack());
    try testing.expect(dq.isEmpty());
}

test "deque: empty pops return null" {
    var dq = try Deque.init(testing.allocator);
    defer dq.deinit();

    try testing.expectEqual(@as(?i64, null), dq.popFront());
    try testing.expectEqual(@as(?i64, null), dq.popBack());
}

test "deque: push overflow is reported" {
    var dq = try Deque.init(testing.allocator);
    defer dq.deinit();

    dq.len_ = std.math.maxInt(usize);
    try testing.expectError(error.Overflow, dq.pushBack(1));
}

test "deque: growth doubling overflow is reported" {
    var dq = try Deque.init(testing.allocator);
    const original = dq.buffer;
    defer dq.allocator.free(original);

    const fake_ptr: [*]i64 = @ptrFromInt(@alignOf(i64));
    dq.buffer = fake_ptr[0 .. std.math.maxInt(usize) - 1];
    dq.head = 0;
    dq.len_ = dq.buffer.len;
    try testing.expectError(error.Overflow, dq.pushBack(1));
}
