//! Doubly Linked List - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/doubly_linked_list.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// A generic doubly linked list with dynamic node allocation.
pub fn DoublyLinkedList(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Node = struct {
            data: T,
            prev: ?*Node = null,
            next: ?*Node = null,
        };

        head: ?*Node = null,
        tail: ?*Node = null,
        len: usize = 0,
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Free all nodes.
        pub fn deinit(self: *Self) void {
            var current = self.head;
            while (current) |node| {
                const next = node.next;
                self.allocator.destroy(node);
                current = next;
            }
            self.head = null;
            self.tail = null;
            self.len = 0;
        }

        /// Insert at head. O(1)
        pub fn insertHead(self: *Self, data: T) !void {
            const node = try self.allocator.create(Node);
            node.* = .{ .data = data, .prev = null, .next = self.head };

            if (self.head) |old_head| {
                old_head.prev = node;
            } else {
                self.tail = node;
            }
            self.head = node;
            self.len += 1;
        }

        /// Insert at tail. O(1)
        pub fn insertTail(self: *Self, data: T) !void {
            const node = try self.allocator.create(Node);
            node.* = .{ .data = data, .prev = self.tail, .next = null };

            if (self.tail) |old_tail| {
                old_tail.next = node;
            } else {
                self.head = node;
            }
            self.tail = node;
            self.len += 1;
        }

        /// Delete head and return its data. Returns null if empty.
        pub fn deleteHead(self: *Self) ?T {
            const head = self.head orelse return null;
            const data = head.data;

            self.head = head.next;
            if (self.head) |new_head| {
                new_head.prev = null;
            } else {
                self.tail = null;
            }

            self.allocator.destroy(head);
            self.len -= 1;
            return data;
        }

        /// Delete tail and return its data. Returns null if empty.
        pub fn deleteTail(self: *Self) ?T {
            const tail = self.tail orelse return null;
            const data = tail.data;

            self.tail = tail.prev;
            if (self.tail) |new_tail| {
                new_tail.next = null;
            } else {
                self.head = null;
            }

            self.allocator.destroy(tail);
            self.len -= 1;
            return data;
        }

        /// Get data at index (forward traversal). Returns null if out of bounds.
        pub fn get(self: *const Self, index: usize) ?T {
            if (index >= self.len) return null;
            var current = self.head;
            for (0..index) |_| {
                current = current.?.next;
            }
            return current.?.data;
        }

        /// Check if list is empty.
        pub fn isEmpty(self: *const Self) bool {
            return self.head == null;
        }

        /// Reverse the list in-place. O(n)
        pub fn reverse(self: *Self) void {
            var current = self.head;
            while (current) |node| {
                const next = node.next;
                node.next = node.prev;
                node.prev = next;
                current = next;
            }
            const old_head = self.head;
            self.head = self.tail;
            self.tail = old_head;
        }

        /// Collect all elements into a caller-owned slice (for test assertions).
        pub fn toSlice(self: *const Self, allocator: Allocator) ![]T {
            const slice = try allocator.alloc(T, self.len);
            var current = self.head;
            var i: usize = 0;
            while (current) |node| {
                slice[i] = node.data;
                i += 1;
                current = node.next;
            }
            return slice;
        }
    };
}

// ===== Tests =====

test "doubly linked list: insert head and tail" {
    const alloc = testing.allocator;
    var list = DoublyLinkedList(i32).init(alloc);
    defer list.deinit();

    try list.insertHead(2);
    try list.insertHead(1);
    try list.insertTail(3);
    try list.insertTail(4);

    try testing.expectEqual(@as(usize, 4), list.len);

    const slice = try list.toSlice(alloc);
    defer alloc.free(slice);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4 }, slice);
}

test "doubly linked list: delete head and tail" {
    const alloc = testing.allocator;
    var list = DoublyLinkedList(i32).init(alloc);
    defer list.deinit();

    try list.insertTail(1);
    try list.insertTail(2);
    try list.insertTail(3);

    try testing.expectEqual(@as(?i32, 1), list.deleteHead());
    try testing.expectEqual(@as(?i32, 3), list.deleteTail());
    try testing.expectEqual(@as(usize, 1), list.len);
    try testing.expectEqual(@as(?i32, 2), list.get(0));
}

test "doubly linked list: delete from empty" {
    const alloc = testing.allocator;
    var list = DoublyLinkedList(i32).init(alloc);
    defer list.deinit();

    try testing.expectEqual(@as(?i32, null), list.deleteHead());
    try testing.expectEqual(@as(?i32, null), list.deleteTail());
    try testing.expect(list.isEmpty());
}

test "doubly linked list: reverse" {
    const alloc = testing.allocator;
    var list = DoublyLinkedList(i32).init(alloc);
    defer list.deinit();

    try list.insertTail(1);
    try list.insertTail(2);
    try list.insertTail(3);
    try list.insertTail(4);
    list.reverse();

    const slice = try list.toSlice(alloc);
    defer alloc.free(slice);
    try testing.expectEqualSlices(i32, &[_]i32{ 4, 3, 2, 1 }, slice);

    // Verify tail pointer is correct after reverse
    try testing.expectEqual(@as(?i32, 1), list.deleteTail());
}

test "doubly linked list: get by index" {
    const alloc = testing.allocator;
    var list = DoublyLinkedList(i32).init(alloc);
    defer list.deinit();

    try list.insertTail(10);
    try list.insertTail(20);
    try list.insertTail(30);

    try testing.expectEqual(@as(?i32, 10), list.get(0));
    try testing.expectEqual(@as(?i32, 20), list.get(1));
    try testing.expectEqual(@as(?i32, 30), list.get(2));
    try testing.expectEqual(@as(?i32, null), list.get(3));
}

test "doubly linked list: single element" {
    const alloc = testing.allocator;
    var list = DoublyLinkedList(i32).init(alloc);
    defer list.deinit();

    try list.insertHead(42);
    try testing.expectEqual(@as(?i32, 42), list.deleteHead());
    try testing.expect(list.isEmpty());

    try list.insertTail(99);
    try testing.expectEqual(@as(?i32, 99), list.deleteTail());
    try testing.expect(list.isEmpty());
}

test "doubly linked list: interleaved operations" {
    const alloc = testing.allocator;
    var list = DoublyLinkedList(i32).init(alloc);
    defer list.deinit();

    try list.insertTail(1);
    try list.insertTail(2);
    try list.insertTail(3);
    try list.insertTail(4);
    try list.insertTail(5);

    _ = list.deleteHead(); // remove 1
    _ = list.deleteTail(); // remove 5
    // remaining: 2, 3, 4

    try list.insertHead(0);
    try list.insertTail(6);
    // now: 0, 2, 3, 4, 6

    const slice = try list.toSlice(alloc);
    defer alloc.free(slice);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 3, 4, 6 }, slice);
}
