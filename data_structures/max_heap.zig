//! Max Heap - Zig implementation (array-based)
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/heap/max_heap.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// A generic max-heap backed by a dynamic array.
pub fn MaxHeap(comptime T: type) type {
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

        /// Peek at the maximum without removing. Returns null if empty.
        pub fn peek(self: *const Self) ?T {
            if (self.items.items.len == 0) return null;
            return self.items.items[0];
        }

        /// Remove and return the maximum. Returns null if empty. O(log n)
        pub fn extractMax(self: *Self) ?T {
            const len = self.items.items.len;
            if (len == 0) return null;
            const max_val = self.items.items[0];
            self.items.items[0] = self.items.items[len - 1];
            _ = self.items.pop();
            if (self.items.items.len > 0) {
                self.siftDown(0);
            }
            return max_val;
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
                if (self.items.items[idx] > self.items.items[p]) {
                    std.mem.swap(T, &self.items.items[idx], &self.items.items[p]);
                    idx = p;
                } else break;
            }
        }

        fn siftDown(self: *Self, idx_: usize) void {
            var idx = idx_;
            const len = self.items.items.len;
            while (true) {
                var largest = idx;
                const left = 2 * idx + 1;
                const right = 2 * idx + 2;

                if (left < len and self.items.items[left] > self.items.items[largest]) {
                    largest = left;
                }
                if (right < len and self.items.items[right] > self.items.items[largest]) {
                    largest = right;
                }
                if (largest != idx) {
                    std.mem.swap(T, &self.items.items[idx], &self.items.items[largest]);
                    idx = largest;
                } else break;
            }
        }

        fn parentIdx(idx: usize) usize {
            return (idx - 1) / 2;
        }
    };
}

test "max heap: insert and extract" {
    var heap = MaxHeap(i32).init(testing.allocator);
    defer heap.deinit();

    try heap.insert(5);
    try heap.insert(3);
    try heap.insert(8);
    try heap.insert(1);
    try heap.insert(4);

    try testing.expectEqual(@as(?i32, 8), heap.peek());
    try testing.expectEqual(@as(?i32, 8), heap.extractMax());
    try testing.expectEqual(@as(?i32, 5), heap.extractMax());
    try testing.expectEqual(@as(?i32, 4), heap.extractMax());
    try testing.expectEqual(@as(?i32, 3), heap.extractMax());
    try testing.expectEqual(@as(?i32, 1), heap.extractMax());
    try testing.expectEqual(@as(?i32, null), heap.extractMax());
}

test "max heap: from slice (heapify)" {
    var heap = try MaxHeap(i32).fromSlice(testing.allocator, &[_]i32{ 6, 3, -1, 1, 4 });
    defer heap.deinit();

    try testing.expectEqual(@as(?i32, 6), heap.extractMax());
    try testing.expectEqual(@as(?i32, 4), heap.extractMax());
    try testing.expectEqual(@as(?i32, 3), heap.extractMax());
    try testing.expectEqual(@as(?i32, 1), heap.extractMax());
    try testing.expectEqual(@as(?i32, -1), heap.extractMax());
}

test "max heap: empty" {
    var heap = MaxHeap(i32).init(testing.allocator);
    defer heap.deinit();

    try testing.expect(heap.isEmpty());
    try testing.expectEqual(@as(?i32, null), heap.peek());
    try testing.expectEqual(@as(?i32, null), heap.extractMax());
}

test "max heap: duplicates" {
    var heap = MaxHeap(i32).init(testing.allocator);
    defer heap.deinit();

    try heap.insert(3);
    try heap.insert(3);
    try heap.insert(1);
    try heap.insert(1);

    try testing.expectEqual(@as(?i32, 3), heap.extractMax());
    try testing.expectEqual(@as(?i32, 3), heap.extractMax());
    try testing.expectEqual(@as(?i32, 1), heap.extractMax());
    try testing.expectEqual(@as(?i32, 1), heap.extractMax());
}
