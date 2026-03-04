//! Circular Linked List - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/circular_linked_list.py

const std = @import("std");
const testing = std.testing;

pub fn CircularLinkedList(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            data: T,
            next: ?*Node,
        };

        allocator: std.mem.Allocator,
        head: ?*Node,
        tail: ?*Node,
        len: usize,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .head = null,
                .tail = null,
                .len = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.len > 0 and self.head != null) {
                var current = self.head.?;
                var remaining = self.len;
                while (remaining > 0) : (remaining -= 1) {
                    const next_ptr = current.next;
                    self.allocator.destroy(current);
                    if (next_ptr == null) break;
                    current = next_ptr.?;
                }
            }
            self.* = undefined;
        }

        pub fn count(self: *const Self) usize {
            return self.len;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.len == 0;
        }

        pub fn insertHead(self: *Self, data: T) !void {
            return self.insertNth(0, data);
        }

        pub fn insertTail(self: *Self, data: T) !void {
            return self.insertNth(self.len, data);
        }

        /// Inserts data at index in [0, len].
        /// Time complexity: O(n), Space complexity: O(1)
        pub fn insertNth(self: *Self, index: usize, data: T) !void {
            if (index > self.len) return error.IndexOutOfRange;

            const new_node = try self.allocator.create(Node);
            errdefer self.allocator.destroy(new_node);
            new_node.* = .{ .data = data, .next = null };

            if (self.head == null) {
                new_node.next = new_node;
                self.head = new_node;
                self.tail = new_node;
                self.len = 1;
                return;
            }

            if (index == 0) {
                new_node.next = self.head;
                self.head = new_node;
                self.tail.?.next = self.head;
                self.len += 1;
                return;
            }

            var prev = self.head.?;
            var step: usize = 0;
            while (step + 1 < index) : (step += 1) {
                prev = prev.next.?;
            }

            new_node.next = prev.next;
            prev.next = new_node;
            if (index == self.len) {
                self.tail = new_node;
            }
            self.len += 1;
            self.tail.?.next = self.head;
        }

        pub fn deleteFront(self: *Self) !T {
            return self.deleteNth(0);
        }

        pub fn deleteTail(self: *Self) !T {
            if (self.len == 0) return error.IndexOutOfRange;
            return self.deleteNth(self.len - 1);
        }

        /// Deletes node at index and returns its data.
        /// Time complexity: O(n), Space complexity: O(1)
        pub fn deleteNth(self: *Self, index: usize) !T {
            if (index >= self.len) return error.IndexOutOfRange;

            if (self.len == 1) {
                const node = self.head.?;
                const value = node.data;
                self.allocator.destroy(node);
                self.head = null;
                self.tail = null;
                self.len = 0;
                return value;
            }

            if (index == 0) {
                const node = self.head.?;
                const value = node.data;
                self.head = node.next;
                self.tail.?.next = self.head;
                self.allocator.destroy(node);
                self.len -= 1;
                return value;
            }

            var prev = self.head.?;
            var step: usize = 0;
            while (step + 1 < index) : (step += 1) {
                prev = prev.next.?;
            }

            const node = prev.next.?;
            const value = node.data;
            prev.next = node.next;
            if (index == self.len - 1) {
                self.tail = prev;
            }
            self.tail.?.next = self.head;
            self.allocator.destroy(node);
            self.len -= 1;
            return value;
        }

        pub fn toOwnedSlice(self: *const Self, allocator: std.mem.Allocator) ![]T {
            const out = try allocator.alloc(T, self.len);
            errdefer allocator.free(out);

            if (self.len == 0) return out;

            var node = self.head.?;
            var i: usize = 0;
            while (i < self.len) : (i += 1) {
                out[i] = node.data;
                node = node.next.?;
            }

            return out;
        }
    };
}

test "circular linked list: python-style operations" {
    var list = CircularLinkedList(i32).init(testing.allocator);
    defer list.deinit();

    try testing.expectEqual(@as(usize, 0), list.count());
    try testing.expect(list.isEmpty());
    try testing.expectError(error.IndexOutOfRange, list.deleteFront());
    try testing.expectError(error.IndexOutOfRange, list.deleteTail());

    for (0..5) |i| {
        try testing.expectEqual(i, list.count());
        try list.insertNth(i, @intCast(i + 1));
    }

    var slice = try list.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(slice);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, slice);

    try list.insertTail(6);
    try list.insertHead(0);
    testing.allocator.free(slice);
    slice = try list.toOwnedSlice(testing.allocator);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 1, 2, 3, 4, 5, 6 }, slice);

    try testing.expectEqual(@as(i32, 0), try list.deleteFront());
    try testing.expectEqual(@as(i32, 6), try list.deleteTail());
    try testing.expectEqual(@as(i32, 3), try list.deleteNth(2));

    try list.insertNth(2, 3);
    testing.allocator.free(slice);
    slice = try list.toOwnedSlice(testing.allocator);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, slice);
    try testing.expect(!list.isEmpty());
}

test "circular linked list: index bounds" {
    var list = CircularLinkedList(i32).init(testing.allocator);
    defer list.deinit();

    try testing.expectError(error.IndexOutOfRange, list.insertNth(1, 10));
    try list.insertHead(1);
    try testing.expectError(error.IndexOutOfRange, list.deleteNth(1));
}

test "circular linked list: extreme long insert/delete" {
    var list = CircularLinkedList(i64).init(testing.allocator);
    defer list.deinit();

    for (0..10_000) |i| {
        try list.insertTail(@intCast(i));
    }
    try testing.expectEqual(@as(usize, 10_000), list.count());

    var i: i64 = 0;
    while (i < 5_000) : (i += 1) {
        try testing.expectEqual(i, try list.deleteFront());
    }

    try testing.expectEqual(@as(usize, 5_000), list.count());

    const rem = try list.toOwnedSlice(testing.allocator);
    defer testing.allocator.free(rem);
    try testing.expectEqual(@as(i64, 5_000), rem[0]);
    try testing.expectEqual(@as(i64, 9_999), rem[rem.len - 1]);
}
