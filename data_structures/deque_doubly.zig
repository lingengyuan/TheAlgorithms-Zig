//! Deque Doubly - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/deque_doubly.py

const std = @import("std");
const testing = std.testing;

pub fn LinkedDeque(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            data: ?T,
            prev: ?*Node,
            next: ?*Node,
        };

        allocator: std.mem.Allocator,
        header: *Node,
        trailer: *Node,
        size: usize,

        pub fn init(allocator: std.mem.Allocator) !Self {
            const header = try allocator.create(Node);
            errdefer allocator.destroy(header);
            const trailer = try allocator.create(Node);
            errdefer allocator.destroy(trailer);

            header.* = .{ .data = null, .prev = null, .next = trailer };
            trailer.* = .{ .data = null, .prev = header, .next = null };

            return .{
                .allocator = allocator,
                .header = header,
                .trailer = trailer,
                .size = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            var node_opt: ?*Node = self.header;
            while (node_opt) |node| {
                const next = node.next;
                self.allocator.destroy(node);
                node_opt = next;
            }
            self.* = undefined;
        }

        pub fn len(self: *const Self) usize {
            return self.size;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.size == 0;
        }

        pub fn first(self: *const Self) !T {
            if (self.isEmpty()) return error.EmptyList;
            return self.header.next.?.data.?;
        }

        pub fn last(self: *const Self) !T {
            if (self.isEmpty()) return error.EmptyList;
            return self.trailer.prev.?.data.?;
        }

        /// Inserts element at front.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn addFirst(self: *Self, element: T) !void {
            try self.insert(self.header, element, self.header.next.?);
        }

        /// Inserts element at back.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn addLast(self: *Self, element: T) !void {
            try self.insert(self.trailer.prev.?, element, self.trailer);
        }

        /// Removes and returns front element.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn removeFirst(self: *Self) !T {
            if (self.isEmpty()) return error.EmptyList;
            return self.delete(self.header.next.?);
        }

        /// Removes and returns back element.
        /// Time complexity: O(1), Space complexity: O(1)
        pub fn removeLast(self: *Self) !T {
            if (self.isEmpty()) return error.EmptyList;
            return self.delete(self.trailer.prev.?);
        }

        fn insert(self: *Self, predecessor: *Node, element: T, successor: *Node) !void {
            const next_size = @addWithOverflow(self.size, @as(usize, 1));
            if (next_size[1] != 0) return error.Overflow;

            const new_node = try self.allocator.create(Node);
            new_node.* = .{
                .data = element,
                .prev = predecessor,
                .next = successor,
            };

            predecessor.next = new_node;
            successor.prev = new_node;
            self.size = next_size[0];
        }

        fn delete(self: *Self, node: *Node) T {
            const predecessor = node.prev.?;
            const successor = node.next.?;

            predecessor.next = successor;
            successor.prev = predecessor;
            self.size -= 1;

            const value = node.data.?;
            self.allocator.destroy(node);
            return value;
        }
    };
}

test "deque doubly: first and last" {
    var deque = try LinkedDeque([]const u8).init(testing.allocator);
    defer deque.deinit();

    try deque.addFirst("A");
    try testing.expectEqualStrings("A", try deque.first());
    try deque.addFirst("B");
    try testing.expectEqualStrings("B", try deque.first());

    try deque.addLast("C");
    try testing.expectEqualStrings("C", try deque.last());
    try testing.expectEqual(@as(usize, 3), deque.len());
}

test "deque doubly: remove empty and remove both ends" {
    var deque = try LinkedDeque(i32).init(testing.allocator);
    defer deque.deinit();

    try testing.expectError(error.EmptyList, deque.removeFirst());
    try testing.expectError(error.EmptyList, deque.removeLast());

    try deque.addFirst(1);
    try deque.addLast(2);
    try deque.addFirst(0);

    try testing.expectEqual(@as(i32, 0), try deque.removeFirst());
    try testing.expectEqual(@as(i32, 2), try deque.removeLast());
    try testing.expectEqual(@as(i32, 1), try deque.removeFirst());
    try testing.expect(deque.isEmpty());
}

test "deque doubly: extreme alternating operations" {
    var deque = try LinkedDeque(i64).init(testing.allocator);
    defer deque.deinit();

    const n: usize = 20_000;
    for (0..n) |i| {
        if (i % 2 == 0) {
            try deque.addFirst(@intCast(i));
        } else {
            try deque.addLast(@intCast(i));
        }
    }

    var removed: usize = 0;
    while (!deque.isEmpty()) {
        if (removed % 2 == 0) {
            _ = try deque.removeFirst();
        } else {
            _ = try deque.removeLast();
        }
        removed += 1;
    }

    try testing.expectEqual(n, removed);
}
