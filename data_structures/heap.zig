//! Heap (Max Heap) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/heap/heap.py

const std = @import("std");
const testing = std.testing;

pub const Heap = struct {
    h: std.ArrayListUnmanaged(i64) = .{},
    heap_size: usize = 0,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Heap {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Heap) void {
        self.h.deinit(self.allocator);
        self.heap_size = 0;
    }

    pub fn parentIndex(child_idx: usize) ?usize {
        return if (child_idx > 0) (child_idx - 1) / 2 else null;
    }

    fn leftChildIdx(self: *const Heap, parent_idx: usize) ?usize {
        const idx = 2 * parent_idx + 1;
        return if (idx < self.heap_size) idx else null;
    }

    fn rightChildIdx(self: *const Heap, parent_idx: usize) ?usize {
        const idx = 2 * parent_idx + 2;
        return if (idx < self.heap_size) idx else null;
    }

    fn maxHeapify(self: *Heap, index: usize) void {
        var idx = index;
        while (idx < self.heap_size) {
            var violation = idx;
            if (self.leftChildIdx(idx)) |left| {
                if (self.h.items[left] > self.h.items[violation]) violation = left;
            }
            if (self.rightChildIdx(idx)) |right| {
                if (self.h.items[right] > self.h.items[violation]) violation = right;
            }
            if (violation == idx) break;
            std.mem.swap(i64, &self.h.items[idx], &self.h.items[violation]);
            idx = violation;
        }
    }

    /// Builds max heap from collection.
    /// Time complexity: O(n), Space complexity: O(n)
    pub fn buildMaxHeap(self: *Heap, collection: []const i64) !void {
        self.h.clearAndFree(self.allocator);
        try self.h.appendSlice(self.allocator, collection);
        self.heap_size = self.h.items.len;

        if (self.heap_size > 1) {
            var i: isize = @as(isize, @intCast(self.heap_size / 2)) - 1;
            while (i >= 0) : (i -= 1) {
                self.maxHeapify(@intCast(i));
            }
        }
    }

    /// Extracts current maximum.
    /// Time complexity: O(log n), Space complexity: O(1)
    pub fn extractMax(self: *Heap) !i64 {
        if (self.heap_size == 0) return error.EmptyHeap;

        if (self.heap_size == 1) {
            self.heap_size = 0;
            return self.h.pop().?;
        }

        const max_value = self.h.items[0];
        self.h.items[0] = self.h.pop().?;
        self.heap_size -= 1;
        self.maxHeapify(0);
        return max_value;
    }

    /// Inserts one value.
    /// Time complexity: O(log n), Space complexity: O(1)
    pub fn insert(self: *Heap, value: i64) !void {
        try self.h.append(self.allocator, value);
        self.heap_size += 1;

        var idx = self.heap_size - 1;
        while (parentIndex(idx)) |p| {
            if (self.h.items[p] >= self.h.items[idx]) break;
            std.mem.swap(i64, &self.h.items[p], &self.h.items[idx]);
            idx = p;
        }
    }

    /// In-place heapsort (ascending order).
    /// Time complexity: O(n log n), Space complexity: O(1)
    pub fn heapSort(self: *Heap) void {
        const size = self.heap_size;
        var j: isize = @as(isize, @intCast(size)) - 1;
        while (j > 0) : (j -= 1) {
            const ju: usize = @intCast(j);
            std.mem.swap(i64, &self.h.items[0], &self.h.items[ju]);
            self.heap_size -= 1;
            self.maxHeapify(0);
        }
        self.heap_size = size;
    }

    pub fn toSlice(self: *const Heap, allocator: std.mem.Allocator) ![]i64 {
        const out = try allocator.alloc(i64, self.h.items.len);
        @memcpy(out, self.h.items);
        return out;
    }
};

test "heap: python core behavior" {
    var h = Heap.init(testing.allocator);
    defer h.deinit();

    const unsorted = [_]i64{ 103, 9, 1, 7, 11, 15, 25, 201, 209, 107, 5 };
    try h.buildMaxHeap(&unsorted);

    try testing.expectEqual(@as(i64, 209), try h.extractMax());

    try h.insert(100);
    h.heapSort();

    const sorted = try h.toSlice(testing.allocator);
    defer testing.allocator.free(sorted);

    try testing.expectEqualSlices(i64, &[_]i64{ 1, 5, 7, 9, 11, 15, 25, 100, 103, 107, 201 }, sorted);
}

test "heap: parent index and boundaries" {
    try testing.expect(Heap.parentIndex(0) == null);
    try testing.expectEqual(@as(?usize, 0), Heap.parentIndex(1));
    try testing.expectEqual(@as(?usize, 0), Heap.parentIndex(2));
    try testing.expectEqual(@as(?usize, 1), Heap.parentIndex(3));

    var h = Heap.init(testing.allocator);
    defer h.deinit();
    try testing.expectError(error.EmptyHeap, h.extractMax());
}

test "heap: extreme extract order" {
    const n: usize = 50_000;
    var values = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(values);

    var prng = std.Random.DefaultPrng.init(0xABCD0123EF);
    const random = prng.random();

    for (0..n) |i| {
        values[i] = @intCast(random.intRangeAtMost(i32, -1_000_000, 1_000_000));
    }

    var h = Heap.init(testing.allocator);
    defer h.deinit();
    try h.buildMaxHeap(values);

    var prev = try h.extractMax();
    var count: usize = 1;
    while (h.heap_size > 0) : (count += 1) {
        const cur = try h.extractMax();
        try testing.expect(prev >= cur);
        prev = cur;
    }

    try testing.expectEqual(n, count);
}
