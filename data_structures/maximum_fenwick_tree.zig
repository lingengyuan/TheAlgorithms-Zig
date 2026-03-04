//! Maximum Fenwick Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/maximum_fenwick_tree.py

const std = @import("std");
const testing = std.testing;

pub const MaxFenwickTree = struct {
    size: usize,
    arr: []i64,
    tree: []i64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, size: usize) !MaxFenwickTree {
        const arr = try allocator.alloc(i64, size);
        errdefer allocator.free(arr);
        const tree = try allocator.alloc(i64, size);

        @memset(arr, 0);
        @memset(tree, 0);

        return .{
            .size = size,
            .arr = arr,
            .tree = tree,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MaxFenwickTree) void {
        self.allocator.free(self.arr);
        self.allocator.free(self.tree);
    }

    pub fn getNext(index: usize) usize {
        return index | (index + 1);
    }

    pub fn getPrev(index: usize) isize {
        return @as(isize, @intCast(index & (index + 1))) - 1;
    }

    /// Sets `index` to `value`.
    /// Time complexity: O(log^2 n), Space complexity: O(1)
    pub fn update(self: *MaxFenwickTree, index: usize, value: i64) !void {
        if (index >= self.size) return error.IndexOutOfBounds;

        self.arr[index] = value;

        var i = index;
        while (i < self.size) : (i = getNext(i)) {
            const left_border: usize = @intCast(getPrev(i) + 1);
            var best = self.arr[i];

            if (left_border < i) {
                var j: isize = @as(isize, @intCast(i)) - 1;
                while (j >= @as(isize, @intCast(left_border))) : (j -= 1) {
                    best = @max(best, self.arr[@intCast(j)]);
                }
            }

            self.tree[i] = best;
        }
    }

    /// Returns maximum value in range [left, right).
    /// Time complexity: O(log^2 n), Space complexity: O(1)
    pub fn query(self: *const MaxFenwickTree, left: usize, right: usize) !i64 {
        if (left > right or right > self.size) return error.InvalidRange;
        if (left == right) return 0;

        var r: isize = @as(isize, @intCast(right)) - 1;
        const l: isize = @intCast(left);
        var result: i64 = 0;

        while (r >= l) {
            const current_left = getPrev(@intCast(r));
            if (current_left >= l) {
                result = @max(result, self.tree[@intCast(r)]);
                r = current_left;
            } else {
                result = @max(result, self.arr[@intCast(r)]);
                r -= 1;
            }
        }

        return result;
    }
};

fn naiveRangeMax(values: []const i64, left: usize, right: usize) i64 {
    if (left >= right) return 0;

    var best: i64 = 0;
    for (values[left..right]) |v| {
        best = @max(best, v);
    }
    return best;
}

test "maximum fenwick tree: python doctest behavior" {
    var ft = try MaxFenwickTree.init(testing.allocator, 5);
    defer ft.deinit();

    try testing.expectEqual(@as(i64, 0), try ft.query(0, 5));
    try ft.update(4, 100);
    try testing.expectEqual(@as(i64, 100), try ft.query(0, 5));

    try ft.update(4, 0);
    try ft.update(2, 20);
    try testing.expectEqual(@as(i64, 20), try ft.query(0, 5));

    try ft.update(4, 10);
    try testing.expectEqual(@as(i64, 20), try ft.query(2, 5));
    try testing.expectEqual(@as(i64, 20), try ft.query(1, 5));

    try ft.update(2, 0);
    try testing.expectEqual(@as(i64, 10), try ft.query(0, 5));
}

test "maximum fenwick tree: boundary" {
    var ft = try MaxFenwickTree.init(testing.allocator, 6);
    defer ft.deinit();

    try ft.update(5, 1);
    try testing.expectEqual(@as(i64, 1), try ft.query(5, 6));

    var ft2 = try MaxFenwickTree.init(testing.allocator, 6);
    defer ft2.deinit();
    try ft2.update(0, 1000);
    try testing.expectEqual(@as(i64, 1000), try ft2.query(0, 1));

    try testing.expectError(error.IndexOutOfBounds, ft2.update(6, 1));
    try testing.expectError(error.InvalidRange, ft2.query(3, 7));
    try testing.expectError(error.InvalidRange, ft2.query(4, 3));
    try testing.expectEqual(@as(i64, 0), try ft2.query(2, 2));
}

test "maximum fenwick tree: extreme randomized parity" {
    const n: usize = 20_000;
    var ft = try MaxFenwickTree.init(testing.allocator, n);
    defer ft.deinit();

    var expected = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(expected);
    @memset(expected, 0);

    var prng = std.Random.DefaultPrng.init(0xC0FFEE1234);
    const random = prng.random();

    var updates: usize = 0;
    while (updates < 30_000) : (updates += 1) {
        const idx = random.uintLessThan(usize, n);
        const val: i64 = @intCast(random.uintLessThan(u32, 1_000_000));
        try ft.update(idx, val);
        expected[idx] = val;
    }

    var q: usize = 0;
    while (q < 1_000) : (q += 1) {
        const a = random.uintLessThan(usize, n);
        const b = random.uintLessThan(usize, n);
        const left = @min(a, b);
        const right = @max(a, b) + 1;

        const got = try ft.query(left, right);
        const want = naiveRangeMax(expected, left, right);
        try testing.expectEqual(want, got);
    }
}
