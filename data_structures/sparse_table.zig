//! Sparse Table (Range Minimum Query) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/arrays/sparse_table.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const SparseTableError = error{
    EmptyNumberList,
    IndexOutOfRange,
    InvalidRange,
};

pub const SparseTable = struct {
    allocator: Allocator,
    rows: usize,
    cols: usize,
    table: []i64,

    pub fn deinit(self: *SparseTable) void {
        self.allocator.free(self.table);
        self.* = undefined;
    }

    pub fn at(self: *const SparseTable, row: usize, col: usize) i64 {
        return self.table[row * self.cols + col];
    }

    fn set(self: *SparseTable, row: usize, col: usize, value: i64) void {
        self.table[row * self.cols + col] = value;
    }
};

/// Precomputes range-minimum sparse table for static input list.
/// Time complexity: O(n log n), Space complexity: O(n log n)
pub fn buildSparseTable(allocator: Allocator, number_list: []const i64) !SparseTable {
    if (number_list.len == 0) return SparseTableError.EmptyNumberList;

    const length = number_list.len;
    const rows = std.math.log2_int(usize, length) + 1;

    const table = try allocator.alloc(i64, rows * length);
    errdefer allocator.free(table);
    @memset(table, 0);

    var st = SparseTable{
        .allocator = allocator,
        .rows = rows,
        .cols = length,
        .table = table,
    };

    for (number_list, 0..) |value, i| {
        st.set(0, i, value);
    }

    var j: usize = 1;
    while ((@as(usize, 1) << @as(u6, @intCast(j))) <= length) : (j += 1) {
        var i: usize = 0;
        const width = @as(usize, 1) << @as(u6, @intCast(j));
        while ((i + width - 1) < length) : (i += 1) {
            const left = st.at(j - 1, i);
            const right = st.at(j - 1, i + (@as(usize, 1) << @as(u6, @intCast(j - 1))));
            st.set(j, i, @min(left, right));
        }
    }

    return st;
}

/// Answers minimum query for range [left_bound, right_bound].
/// Time complexity: O(1), Space complexity: O(1)
pub fn query(st: *const SparseTable, left_bound: usize, right_bound: usize) !i64 {
    if (left_bound >= st.cols or right_bound >= st.cols) return SparseTableError.IndexOutOfRange;
    if (left_bound > right_bound) return SparseTableError.InvalidRange;

    const len = right_bound - left_bound + 1;
    const j = std.math.log2_int(usize, len);

    const width = @as(usize, 1) << @as(u6, @intCast(j));
    const a = st.at(j, (right_bound + 1) - width);
    const b = st.at(j, left_bound);
    return @min(a, b);
}

test "sparse table: python build samples" {
    var st1 = try buildSparseTable(testing.allocator, &[_]i64{ 8, 1, 0, 3, 4, 9, 3 });
    defer st1.deinit();

    try testing.expectEqual(@as(usize, 3), st1.rows);
    try testing.expectEqualSlices(i64, &[_]i64{ 8, 1, 0, 3, 4, 9, 3 }, st1.table[0..7]);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 0, 0, 3, 4, 3, 0 }, st1.table[7..14]);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 0, 0, 3, 0, 0, 0 }, st1.table[14..21]);

    var st2 = try buildSparseTable(testing.allocator, &[_]i64{ 3, 1, 9 });
    defer st2.deinit();
    try testing.expectEqual(@as(usize, 2), st2.rows);
    try testing.expectEqualSlices(i64, &[_]i64{ 3, 1, 9 }, st2.table[0..3]);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 1, 0 }, st2.table[3..6]);
}

test "sparse table: python query samples" {
    var st = try buildSparseTable(testing.allocator, &[_]i64{ 8, 1, 0, 3, 4, 9, 3 });
    defer st.deinit();

    try testing.expectEqual(@as(i64, 0), try query(&st, 0, 4));
    try testing.expectEqual(@as(i64, 3), try query(&st, 4, 6));

    var st2 = try buildSparseTable(testing.allocator, &[_]i64{ 3, 1, 9 });
    defer st2.deinit();
    try testing.expectEqual(@as(i64, 9), try query(&st2, 2, 2));
    try testing.expectEqual(@as(i64, 1), try query(&st2, 0, 1));
}

test "sparse table: invalid input and extreme" {
    try testing.expectError(SparseTableError.EmptyNumberList, buildSparseTable(testing.allocator, &[_]i64{}));

    var st = try buildSparseTable(testing.allocator, &[_]i64{ 8, 1, 0, 3, 4, 9, 3 });
    defer st.deinit();
    try testing.expectError(SparseTableError.IndexOutOfRange, query(&st, 0, 11));
    try testing.expectError(SparseTableError.InvalidRange, query(&st, 4, 1));

    const n: usize = 8192;
    const arr = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(arr);
    for (arr, 0..) |*v, i| {
        v.* = @as(i64, @intCast((i * 37) % 997)) - 400;
    }

    var big = try buildSparseTable(testing.allocator, arr);
    defer big.deinit();

    var l: usize = 0;
    while (l < n) : (l += 257) {
        const r = @min(n - 1, l + 511);
        var expected = arr[l];
        var idx = l;
        while (idx <= r) : (idx += 1) {
            expected = @min(expected, arr[idx]);
        }
        try testing.expectEqual(expected, try query(&big, l, r));
    }
}
