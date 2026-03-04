//! Double Ended Queue - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/queues/double_ended_queue.py

const std = @import("std");
const testing = std.testing;

pub fn Deque(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            value: T,
            next: ?*Node,
            prev: ?*Node,
        };

        pub const Iterator = struct {
            current: ?*Node,

            pub fn next(self: *Iterator) ?T {
                const node = self.current orelse return null;
                self.current = node.next;
                return node.value;
            }
        };

        allocator: std.mem.Allocator,
        front: ?*Node,
        back: ?*Node,
        len_: usize,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .front = null,
                .back = null,
                .len_ = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            var cursor = self.front;
            while (cursor) |node| {
                cursor = node.next;
                self.allocator.destroy(node);
            }
            self.front = null;
            self.back = null;
            self.len_ = 0;
        }

        /// Adds value to the back.
        /// Time complexity: O(1)
        pub fn append(self: *Self, value: T) !void {
            const node = try self.allocator.create(Node);
            node.* = .{
                .value = value,
                .next = null,
                .prev = self.back,
            };

            if (self.back) |back| {
                back.next = node;
            } else {
                self.front = node;
            }
            self.back = node;
            self.len_ += 1;
        }

        /// Adds value to the front.
        /// Time complexity: O(1)
        pub fn appendLeft(self: *Self, value: T) !void {
            const node = try self.allocator.create(Node);
            node.* = .{
                .value = value,
                .next = self.front,
                .prev = null,
            };

            if (self.front) |front| {
                front.prev = node;
            } else {
                self.back = node;
            }
            self.front = node;
            self.len_ += 1;
        }

        /// Appends all values to the back.
        /// Time complexity: O(n)
        pub fn extend(self: *Self, values: []const T) !void {
            for (values) |value| {
                try self.append(value);
            }
        }

        /// Appends all values to the front, one by one.
        /// Same order semantics as Python `extendleft`.
        /// Time complexity: O(n)
        pub fn extendLeft(self: *Self, values: []const T) !void {
            for (values) |value| {
                try self.appendLeft(value);
            }
        }

        /// Removes and returns the back element.
        /// Time complexity: O(1)
        pub fn pop(self: *Self) !T {
            const node = self.back orelse return error.EmptyDeque;
            const value = node.value;

            self.back = node.prev;
            if (self.back) |back| {
                back.next = null;
            } else {
                self.front = null;
            }

            self.allocator.destroy(node);
            self.len_ -= 1;
            return value;
        }

        /// Removes and returns the front element.
        /// Time complexity: O(1)
        pub fn popLeft(self: *Self) !T {
            const node = self.front orelse return error.EmptyDeque;
            const value = node.value;

            self.front = node.next;
            if (self.front) |front| {
                front.prev = null;
            } else {
                self.back = null;
            }

            self.allocator.destroy(node);
            self.len_ -= 1;
            return value;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.front == null;
        }

        pub fn len(self: *const Self) usize {
            return self.len_;
        }

        pub fn eql(self: *const Self, other: *const Self) bool {
            if (self.len_ != other.len_) return false;

            var left = self.front;
            var right = other.front;
            while (left != null and right != null) {
                if (!std.meta.eql(left.?.value, right.?.value)) return false;
                left = left.?.next;
                right = right.?.next;
            }
            return true;
        }

        pub fn iterator(self: *const Self) Iterator {
            return .{ .current = self.front };
        }

        pub fn toOwnedSlice(self: *const Self, allocator: std.mem.Allocator) ![]T {
            const out = try allocator.alloc(T, self.len_);
            var idx: usize = 0;
            var cursor = self.front;
            while (cursor) |node| {
                out[idx] = node.value;
                idx += 1;
                cursor = node.next;
            }
            return out;
        }
    };
}

fn expectDequeValues(comptime T: type, deque: *const Deque(T), expected: []const T) !void {
    const values = try deque.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(T, expected, values);
}

test "double ended queue: append appendLeft extend extendLeft" {
    var dq = Deque(i64).init(testing.allocator);
    defer dq.deinit();

    try dq.extend(&[_]i64{ 1, 2, 3 });
    try dq.append(4);
    try expectDequeValues(i64, &dq, &[_]i64{ 1, 2, 3, 4 });

    var dq2 = Deque(i64).init(testing.allocator);
    defer dq2.deinit();
    try dq2.extend(&[_]i64{ 2, 3 });
    try dq2.appendLeft(1);
    try expectDequeValues(i64, &dq2, &[_]i64{ 1, 2, 3 });

    try dq2.extend(&[_]i64{ 4, 5 });
    try expectDequeValues(i64, &dq2, &[_]i64{ 1, 2, 3, 4, 5 });

    try dq2.extendLeft(&[_]i64{ 0, -1 });
    try expectDequeValues(i64, &dq2, &[_]i64{ -1, 0, 1, 2, 3, 4, 5 });
}

test "double ended queue: pop and popleft behavior" {
    var single = Deque(i64).init(testing.allocator);
    defer single.deinit();

    try single.append(1);
    try testing.expectEqual(@as(i64, 1), try single.pop());
    try testing.expect(single.isEmpty());

    var dq = Deque(i64).init(testing.allocator);
    defer dq.deinit();

    try dq.extend(&[_]i64{ 15182, 1, 2, 3 });
    try testing.expectEqual(@as(i64, 3), try dq.pop());
    try testing.expectEqual(@as(i64, 15182), try dq.popLeft());
    try expectDequeValues(i64, &dq, &[_]i64{ 1, 2 });

    try testing.expectEqual(@as(i64, 2), try dq.pop());
    try testing.expectEqual(@as(i64, 1), try dq.popLeft());
    try testing.expectError(error.EmptyDeque, dq.pop());
    try testing.expectError(error.EmptyDeque, dq.popLeft());
}

test "double ended queue: iterator and equality" {
    var a = Deque(i32).init(testing.allocator);
    defer a.deinit();
    var b = Deque(i32).init(testing.allocator);
    defer b.deinit();
    var c = Deque(i32).init(testing.allocator);
    defer c.deinit();

    try a.extend(&[_]i32{ 1, 2, 3 });
    try b.extend(&[_]i32{ 1, 2, 3 });
    try c.extend(&[_]i32{ 1, 2 });

    try testing.expect(a.eql(&b));
    try testing.expect(!a.eql(&c));

    var it = a.iterator();
    try testing.expectEqual(@as(?i32, 1), it.next());
    try testing.expectEqual(@as(?i32, 2), it.next());
    try testing.expectEqual(@as(?i32, 3), it.next());
    try testing.expectEqual(@as(?i32, null), it.next());
}

test "double ended queue: extreme alternating operations" {
    var dq = Deque(i64).init(testing.allocator);
    defer dq.deinit();

    const n: usize = 100_000;
    for (0..n) |i| {
        if (i % 2 == 0) {
            try dq.append(@intCast(i));
        } else {
            try dq.appendLeft(@intCast(i));
        }
    }

    try testing.expectEqual(n, dq.len());

    var removals: usize = 0;
    while (!dq.isEmpty()) : (removals += 1) {
        if (removals % 2 == 0) {
            _ = try dq.pop();
        } else {
            _ = try dq.popLeft();
        }
    }

    try testing.expectEqual(n, removals);
    try testing.expectEqual(@as(usize, 0), dq.len());
    try testing.expect(dq.isEmpty());
}
