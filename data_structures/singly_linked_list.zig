//! Singly Linked List - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/singly_linked_list.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// A generic singly linked list with dynamic node allocation.
pub fn SinglyLinkedList(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Node = struct {
            data: T,
            next: ?*Node = null,
        };

        head: ?*Node = null,
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
            self.len = 0;
        }

        /// Insert at head. O(1)
        pub fn insertHead(self: *Self, data: T) !void {
            const node = try self.allocator.create(Node);
            node.* = .{ .data = data, .next = self.head };
            self.head = node;
            self.len += 1;
        }

        /// Insert at tail. O(n)
        pub fn insertTail(self: *Self, data: T) !void {
            const node = try self.allocator.create(Node);
            node.* = .{ .data = data, .next = null };

            if (self.head) |_| {
                var current = self.head;
                while (current.?.next) |_| {
                    current = current.?.next;
                }
                current.?.next = node;
            } else {
                self.head = node;
            }
            self.len += 1;
        }

        /// Delete head and return its data. Returns null if empty.
        pub fn deleteHead(self: *Self) ?T {
            const head = self.head orelse return null;
            const data = head.data;
            self.head = head.next;
            self.allocator.destroy(head);
            self.len -= 1;
            return data;
        }

        /// Delete tail and return its data. Returns null if empty.
        pub fn deleteTail(self: *Self) ?T {
            const head = self.head orelse return null;

            // Single node
            if (head.next == null) {
                const data = head.data;
                self.allocator.destroy(head);
                self.head = null;
                self.len -= 1;
                return data;
            }

            // Walk to second-to-last
            var current = head;
            while (current.next.?.next) |_| {
                current = current.next.?;
            }
            const tail = current.next.?;
            const data = tail.data;
            current.next = null;
            self.allocator.destroy(tail);
            self.len -= 1;
            return data;
        }

        /// Get data at index. Returns null if out of bounds.
        pub fn get(self: *const Self, index: usize) ?T {
            if (index >= self.len) return null;
            var current = self.head;
            for (0..index) |_| {
                current = current.?.next;
            }
            return current.?.data;
        }

        /// Reverse the list in-place. O(n)
        pub fn reverse(self: *Self) void {
            var prev: ?*Node = null;
            var current = self.head;
            while (current) |node| {
                const next = node.next;
                node.next = prev;
                prev = node;
                current = next;
            }
            self.head = prev;
        }

        /// Check if list is empty.
        pub fn isEmpty(self: *const Self) bool {
            return self.head == null;
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

test "singly linked list: insert head and tail" {
    const alloc = testing.allocator;
    var list = SinglyLinkedList(i32).init(alloc);
    defer list.deinit();

    try list.insertTail(1);
    try list.insertTail(2);
    try list.insertTail(3);
    try list.insertHead(0);

    try testing.expectEqual(@as(usize, 4), list.len);

    const slice = try list.toSlice(alloc);
    defer alloc.free(slice);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 1, 2, 3 }, slice);
}

test "singly linked list: delete head and tail" {
    const alloc = testing.allocator;
    var list = SinglyLinkedList(i32).init(alloc);
    defer list.deinit();

    try list.insertTail(1);
    try list.insertTail(2);
    try list.insertTail(3);

    try testing.expectEqual(@as(?i32, 1), list.deleteHead());
    try testing.expectEqual(@as(?i32, 3), list.deleteTail());
    try testing.expectEqual(@as(usize, 1), list.len);
    try testing.expectEqual(@as(?i32, 2), list.get(0));
}

test "singly linked list: delete from empty" {
    const alloc = testing.allocator;
    var list = SinglyLinkedList(i32).init(alloc);
    defer list.deinit();

    try testing.expectEqual(@as(?i32, null), list.deleteHead());
    try testing.expectEqual(@as(?i32, null), list.deleteTail());
    try testing.expect(list.isEmpty());
}

test "singly linked list: reverse" {
    const alloc = testing.allocator;
    var list = SinglyLinkedList(i32).init(alloc);
    defer list.deinit();

    try list.insertTail(1);
    try list.insertTail(2);
    try list.insertTail(3);
    list.reverse();

    const slice = try list.toSlice(alloc);
    defer alloc.free(slice);
    try testing.expectEqualSlices(i32, &[_]i32{ 3, 2, 1 }, slice);
}

test "singly linked list: get by index" {
    const alloc = testing.allocator;
    var list = SinglyLinkedList(i32).init(alloc);
    defer list.deinit();

    try list.insertTail(10);
    try list.insertTail(20);
    try list.insertTail(30);

    try testing.expectEqual(@as(?i32, 10), list.get(0));
    try testing.expectEqual(@as(?i32, 20), list.get(1));
    try testing.expectEqual(@as(?i32, 30), list.get(2));
    try testing.expectEqual(@as(?i32, null), list.get(3));
}

test "singly linked list: single element operations" {
    const alloc = testing.allocator;
    var list = SinglyLinkedList(i32).init(alloc);
    defer list.deinit();

    try list.insertHead(42);
    try testing.expectEqual(@as(usize, 1), list.len);
    try testing.expectEqual(@as(?i32, 42), list.deleteTail());
    try testing.expect(list.isEmpty());
}
