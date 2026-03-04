//! Non-Recursive Segment Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/non_recursive_segment_tree.py

const std = @import("std");
const testing = std.testing;

pub fn sumCombine(a: i64, b: i64) i64 {
    return a + b;
}

pub fn minCombine(a: i64, b: i64) i64 {
    return @min(a, b);
}

pub fn maxCombine(a: i64, b: i64) i64 {
    return @max(a, b);
}

pub const SegmentTree = struct {
    n: usize,
    st: []i64,
    combine: *const fn (i64, i64) i64,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        arr: []const i64,
        combine: *const fn (i64, i64) i64,
    ) !SegmentTree {
        const n = arr.len;
        const st = try allocator.alloc(i64, n * 2);

        @memset(st, 0);
        @memcpy(st[n..], arr);

        var tree = SegmentTree{ .n = n, .st = st, .combine = combine, .allocator = allocator };
        tree.build();
        return tree;
    }

    pub fn deinit(self: *SegmentTree) void {
        self.allocator.free(self.st);
    }

    fn build(self: *SegmentTree) void {
        if (self.n == 0) return;
        var p: usize = self.n - 1;
        while (p > 0) : (p -= 1) {
            self.st[p] = self.combine(self.st[p * 2], self.st[p * 2 + 1]);
        }
    }

    /// Updates one position.
    /// Time complexity: O(log n), Space complexity: O(1)
    pub fn update(self: *SegmentTree, p_in: usize, v: i64) !void {
        if (p_in >= self.n) return error.IndexOutOfBounds;

        var p = p_in + self.n;
        self.st[p] = v;

        while (p > 1) {
            p /= 2;
            self.st[p] = self.combine(self.st[p * 2], self.st[p * 2 + 1]);
        }
    }

    /// Queries combined value in inclusive range [left, right].
    /// Time complexity: O(log n), Space complexity: O(1)
    pub fn query(self: *const SegmentTree, left_in: usize, right_in: usize) !i64 {
        if (self.n == 0) return error.EmptyTree;
        if (left_in > right_in or right_in >= self.n) return error.InvalidRange;

        var left = left_in + self.n;
        var right = right_in + self.n;

        var has_res = false;
        var res: i64 = 0;

        while (left <= right) {
            if (left % 2 == 1) {
                res = if (!has_res) self.st[left] else self.combine(res, self.st[left]);
                has_res = true;
            }
            if (right % 2 == 0) {
                res = if (!has_res) self.st[right] else self.combine(res, self.st[right]);
                has_res = true;
            }
            left = (left + 1) / 2;
            if (right == 0) break;
            right = (right - 1) / 2;
        }

        return res;
    }
};

fn naiveSum(arr: []const i64, left: usize, right: usize) i64 {
    var total: i64 = 0;
    for (arr[left .. right + 1]) |v| total += v;
    return total;
}

test "non recursive segment tree: python sum/min/max style examples" {
    {
        var st = try SegmentTree.init(testing.allocator, &[_]i64{ 1, 2, 3 }, sumCombine);
        defer st.deinit();
        try testing.expectEqual(@as(i64, 6), try st.query(0, 2));
    }

    {
        var st = try SegmentTree.init(testing.allocator, &[_]i64{ 3, 1, 2 }, minCombine);
        defer st.deinit();
        try testing.expectEqual(@as(i64, 1), try st.query(0, 2));
    }

    {
        var st = try SegmentTree.init(testing.allocator, &[_]i64{ 2, 3, 1 }, maxCombine);
        defer st.deinit();
        try testing.expectEqual(@as(i64, 3), try st.query(0, 2));
    }

    {
        var st = try SegmentTree.init(testing.allocator, &[_]i64{ 1, 5, 7, -1, 6 }, sumCombine);
        defer st.deinit();

        try st.update(1, -1);
        try st.update(2, 3);
        try testing.expectEqual(@as(i64, 2), try st.query(1, 2));
        try testing.expectEqual(@as(i64, -1), try st.query(1, 1));

        try st.update(4, 1);
        try testing.expectEqual(@as(i64, 0), try st.query(3, 4));
    }
}

test "non recursive segment tree: boundary" {
    var st = try SegmentTree.init(testing.allocator, &[_]i64{ 10, 20, 30 }, sumCombine);
    defer st.deinit();

    try testing.expectError(error.IndexOutOfBounds, st.update(3, 100));
    try testing.expectError(error.InvalidRange, st.query(2, 1));
    try testing.expectError(error.InvalidRange, st.query(0, 3));

    var empty = try SegmentTree.init(testing.allocator, &[_]i64{}, sumCombine);
    defer empty.deinit();
    try testing.expectError(error.EmptyTree, empty.query(0, 0));
}

test "non recursive segment tree: extreme randomized parity" {
    const n: usize = 25_000;
    var values = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(values);

    var prng = std.Random.DefaultPrng.init(0xA51A9);
    const random = prng.random();

    for (0..n) |i| {
        values[i] = @intCast(random.intRangeAtMost(i32, -5000, 5000));
    }

    var st = try SegmentTree.init(testing.allocator, values, sumCombine);
    defer st.deinit();

    var iter: usize = 0;
    while (iter < 8_000) : (iter += 1) {
        const idx = random.uintLessThan(usize, n);
        const val: i64 = @intCast(random.intRangeAtMost(i32, -5000, 5000));
        values[idx] = val;
        try st.update(idx, val);
    }

    var q: usize = 0;
    while (q < 700) : (q += 1) {
        const a = random.uintLessThan(usize, n);
        const b = random.uintLessThan(usize, n);
        const left = @min(a, b);
        const right = @max(a, b);
        try testing.expectEqual(naiveSum(values, left, right), try st.query(left, right));
    }
}
