//! Lazy Segment Tree (Range Assign + Range Max) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/lazy_segment_tree.py

const std = @import("std");
const testing = std.testing;

pub const LazySegmentTree = struct {
    size: usize,
    segment_tree: []i64,
    lazy: []i64,
    flag: []bool,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, size: usize) !LazySegmentTree {
        if (size == 0) return error.EmptySize;

        const cap = 4 * size;
        const segment_tree = try allocator.alloc(i64, cap);
        errdefer allocator.free(segment_tree);

        const lazy = try allocator.alloc(i64, cap);
        errdefer allocator.free(lazy);

        const flag = try allocator.alloc(bool, cap);

        @memset(segment_tree, 0);
        @memset(lazy, 0);
        @memset(flag, false);

        return .{
            .size = size,
            .segment_tree = segment_tree,
            .lazy = lazy,
            .flag = flag,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LazySegmentTree) void {
        self.allocator.free(self.segment_tree);
        self.allocator.free(self.lazy);
        self.allocator.free(self.flag);
    }

    fn left(idx: usize) usize {
        return idx * 2;
    }

    fn right(idx: usize) usize {
        return idx * 2 + 1;
    }

    fn apply(self: *LazySegmentTree, idx: usize, left_element: usize, right_element: usize, value: i64) void {
        _ = left_element;
        _ = right_element;
        self.segment_tree[idx] = value;
        self.lazy[idx] = value;
        self.flag[idx] = true;
    }

    fn push(self: *LazySegmentTree, idx: usize, left_element: usize, right_element: usize) void {
        if (!self.flag[idx]) return;

        const value = self.lazy[idx];
        self.flag[idx] = false;

        if (left_element != right_element) {
            const l = left(idx);
            const r = right(idx);
            self.apply(l, left_element, (left_element + right_element) / 2, value);
            self.apply(r, (left_element + right_element) / 2 + 1, right_element, value);
        }
    }

    fn buildRec(self: *LazySegmentTree, idx: usize, left_element: usize, right_element: usize, arr: []const i64) void {
        if (left_element == right_element) {
            self.segment_tree[idx] = arr[left_element - 1];
            return;
        }

        const mid = (left_element + right_element) / 2;
        self.buildRec(left(idx), left_element, mid, arr);
        self.buildRec(right(idx), mid + 1, right_element, arr);
        self.segment_tree[idx] = @max(self.segment_tree[left(idx)], self.segment_tree[right(idx)]);
    }

    /// Builds tree from input values.
    /// Time complexity: O(n), Space complexity: O(n)
    pub fn build(self: *LazySegmentTree, arr: []const i64) !void {
        if (arr.len != self.size) return error.InvalidInputLength;
        self.buildRec(1, 1, self.size, arr);
    }

    fn updateRec(
        self: *LazySegmentTree,
        idx: usize,
        left_element: usize,
        right_element: usize,
        a: usize,
        b: usize,
        val: i64,
    ) void {
        self.push(idx, left_element, right_element);

        if (right_element < a or left_element > b) return;

        if (left_element >= a and right_element <= b) {
            self.apply(idx, left_element, right_element, val);
            return;
        }

        const mid = (left_element + right_element) / 2;
        self.updateRec(left(idx), left_element, mid, a, b, val);
        self.updateRec(right(idx), mid + 1, right_element, a, b, val);
        self.segment_tree[idx] = @max(self.segment_tree[left(idx)], self.segment_tree[right(idx)]);
    }

    /// Assigns `val` to all elements in inclusive range [a, b], 1-indexed.
    /// Time complexity: O(log n), Space complexity: O(log n)
    pub fn rangeAssign(self: *LazySegmentTree, a: usize, b: usize, val: i64) !void {
        if (a < 1 or b < a or b > self.size) return error.InvalidRange;
        self.updateRec(1, 1, self.size, a, b, val);
    }

    fn queryRec(
        self: *LazySegmentTree,
        idx: usize,
        left_element: usize,
        right_element: usize,
        a: usize,
        b: usize,
    ) i64 {
        self.push(idx, left_element, right_element);

        if (right_element < a or left_element > b) return std.math.minInt(i64);
        if (left_element >= a and right_element <= b) return self.segment_tree[idx];

        const mid = (left_element + right_element) / 2;
        const q1 = self.queryRec(left(idx), left_element, mid, a, b);
        const q2 = self.queryRec(right(idx), mid + 1, right_element, a, b);
        return @max(q1, q2);
    }

    /// Queries maximum in inclusive range [a, b], 1-indexed.
    /// Time complexity: O(log n), Space complexity: O(log n)
    pub fn rangeMax(self: *LazySegmentTree, a: usize, b: usize) !i64 {
        if (a < 1 or b < a or b > self.size) return error.InvalidRange;
        return self.queryRec(1, 1, self.size, a, b);
    }
};

test "lazy segment tree: python examples" {
    const A = [_]i64{ 1, 2, -4, 7, 3, -5, 6, 11, -20, 9, 14, 15, 5, 2, -8 };

    var st = try LazySegmentTree.init(testing.allocator, A.len);
    defer st.deinit();

    try st.build(&A);

    try testing.expectEqual(@as(i64, 7), try st.rangeMax(4, 6));
    try testing.expectEqual(@as(i64, 14), try st.rangeMax(7, 11));
    try testing.expectEqual(@as(i64, 15), try st.rangeMax(7, 12));

    try st.rangeAssign(1, 3, 111);
    try testing.expectEqual(@as(i64, 111), try st.rangeMax(1, 15));

    try st.rangeAssign(7, 8, 235);
    try testing.expectEqual(@as(i64, 235), try st.rangeMax(1, 15));
    try testing.expectEqual(@as(i64, 235), try st.rangeMax(7, 8));
}

test "lazy segment tree: boundary" {
    var st = try LazySegmentTree.init(testing.allocator, 3);
    defer st.deinit();

    try st.build(&[_]i64{ 5, -2, 9 });

    try testing.expectError(error.InvalidRange, st.rangeAssign(0, 1, 7));
    try testing.expectError(error.InvalidRange, st.rangeAssign(3, 2, 7));
    try testing.expectError(error.InvalidRange, st.rangeMax(0, 1));
    try testing.expectError(error.InvalidRange, st.rangeMax(1, 4));
    try testing.expectError(error.InvalidInputLength, st.build(&[_]i64{1}));
}

test "lazy segment tree: extreme large range assignments" {
    const n: usize = 50_000;

    var values = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(values);

    for (0..n) |i| {
        values[i] = @as(i64, @intCast(i % 1000)) - 500;
    }

    var st = try LazySegmentTree.init(testing.allocator, n);
    defer st.deinit();
    try st.build(values);

    try st.rangeAssign(1, n, 111);
    try testing.expectEqual(@as(i64, 111), try st.rangeMax(1, n));

    try st.rangeAssign(n / 3, n / 2, 999);
    try testing.expectEqual(@as(i64, 999), try st.rangeMax(1, n));
    try testing.expectEqual(@as(i64, 999), try st.rangeMax(n / 3, n / 2));
    try testing.expectEqual(@as(i64, 111), try st.rangeMax(n / 2 + 1, n));

    try st.rangeAssign(1, 1, -777);
    try testing.expectEqual(@as(i64, -777), try st.rangeMax(1, 1));
    try testing.expectEqual(@as(i64, 999), try st.rangeMax(1, n));
}
