//! Min Heap - Zig implementation (array-based)
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/heap/min_heap.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// A generic min-heap backed by a dynamic array.
pub fn MinHeap(comptime T: type) type {
    return struct {
        const Self = @This();

        items: std.ArrayListUnmanaged(T) = .{},
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        pub fn deinit(self: *Self) void {
            self.items.deinit(self.allocator);
        }

        /// Build a heap from an existing slice. O(n)
        pub fn fromSlice(allocator: Allocator, data: []const T) !Self {
            var self = Self.init(allocator);
            try self.items.appendSlice(allocator, data);
            // Heapify from last parent down to root
            if (self.items.items.len > 1) {
                var i: usize = parentIdx(self.items.items.len - 1);
                while (true) {
                    self.siftDown(i);
                    if (i == 0) break;
                    i -= 1;
                }
            }
            return self;
        }

        /// Insert a value. O(log n)
        pub fn insert(self: *Self, value: T) !void {
            try self.items.append(self.allocator, value);
            self.siftUp(self.items.items.len - 1);
        }

        /// Peek at the minimum without removing. Returns null if empty.
        pub fn peek(self: *const Self) ?T {
            if (self.items.items.len == 0) return null;
            return self.items.items[0];
        }

        /// Remove and return the minimum. Returns null if empty. O(log n)
        pub fn extractMin(self: *Self) ?T {
            const len = self.items.items.len;
            if (len == 0) return null;
            const min_val = self.items.items[0];
            self.items.items[0] = self.items.items[len - 1];
            _ = self.items.pop();
            if (self.items.items.len > 0) {
                self.siftDown(0);
            }
            return min_val;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.items.items.len == 0;
        }

        pub fn size(self: *const Self) usize {
            return self.items.items.len;
        }

        fn siftUp(self: *Self, idx_: usize) void {
            var idx = idx_;
            while (idx > 0) {
                const p = parentIdx(idx);
                if (self.items.items[idx] < self.items.items[p]) {
                    const tmp = self.items.items[idx];
                    self.items.items[idx] = self.items.items[p];
                    self.items.items[p] = tmp;
                    idx = p;
                } else break;
            }
        }

        fn siftDown(self: *Self, idx_: usize) void {
            var idx = idx_;
            const len = self.items.items.len;
            while (true) {
                var smallest = idx;
                const left = 2 * idx + 1;
                const right = 2 * idx + 2;

                if (left < len and self.items.items[left] < self.items.items[smallest]) {
                    smallest = left;
                }
                if (right < len and self.items.items[right] < self.items.items[smallest]) {
                    smallest = right;
                }
                if (smallest != idx) {
                    const tmp = self.items.items[idx];
                    self.items.items[idx] = self.items.items[smallest];
                    self.items.items[smallest] = tmp;
                    idx = smallest;
                } else break;
            }
        }

        fn parentIdx(idx: usize) usize {
            return (idx - 1) / 2;
        }
    };
}

// ===== Tests =====

test "min heap: insert and extract" {
    const alloc = testing.allocator;
    var heap = MinHeap(i32).init(alloc);
    defer heap.deinit();

    try heap.insert(5);
    try heap.insert(3);
    try heap.insert(8);
    try heap.insert(1);
    try heap.insert(4);

    try testing.expectEqual(@as(?i32, 1), heap.peek());
    try testing.expectEqual(@as(?i32, 1), heap.extractMin());
    try testing.expectEqual(@as(?i32, 3), heap.extractMin());
    try testing.expectEqual(@as(?i32, 4), heap.extractMin());
    try testing.expectEqual(@as(?i32, 5), heap.extractMin());
    try testing.expectEqual(@as(?i32, 8), heap.extractMin());
    try testing.expectEqual(@as(?i32, null), heap.extractMin());
}

test "min heap: from slice (heapify)" {
    const alloc = testing.allocator;
    var heap = try MinHeap(i32).fromSlice(alloc, &[_]i32{ 6, 3, -1, 1, 4 });
    defer heap.deinit();

    try testing.expectEqual(@as(?i32, -1), heap.extractMin());
    try testing.expectEqual(@as(?i32, 1), heap.extractMin());
    try testing.expectEqual(@as(?i32, 3), heap.extractMin());
    try testing.expectEqual(@as(?i32, 4), heap.extractMin());
    try testing.expectEqual(@as(?i32, 6), heap.extractMin());
}

test "min heap: empty" {
    const alloc = testing.allocator;
    var heap = MinHeap(i32).init(alloc);
    defer heap.deinit();

    try testing.expect(heap.isEmpty());
    try testing.expectEqual(@as(?i32, null), heap.peek());
    try testing.expectEqual(@as(?i32, null), heap.extractMin());
}

test "min heap: single element" {
    const alloc = testing.allocator;
    var heap = MinHeap(i32).init(alloc);
    defer heap.deinit();

    try heap.insert(42);
    try testing.expectEqual(@as(usize, 1), heap.size());
    try testing.expectEqual(@as(?i32, 42), heap.extractMin());
    try testing.expect(heap.isEmpty());
}

test "min heap: duplicates" {
    const alloc = testing.allocator;
    var heap = MinHeap(i32).init(alloc);
    defer heap.deinit();

    try heap.insert(3);
    try heap.insert(3);
    try heap.insert(1);
    try heap.insert(1);

    try testing.expectEqual(@as(?i32, 1), heap.extractMin());
    try testing.expectEqual(@as(?i32, 1), heap.extractMin());
    try testing.expectEqual(@as(?i32, 3), heap.extractMin());
    try testing.expectEqual(@as(?i32, 3), heap.extractMin());
}
