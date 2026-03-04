//! Range Sum Query (Prefix Sum) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/range_sum_query.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const RangeSumQueryError = error{
    EmptyArray,
    InvalidQuery,
    Overflow,
};

pub const Query = struct {
    left: usize,
    right: usize,
};

/// Answers inclusive range-sum queries using a prefix sum array.
/// Time complexity: O(n + q), Space complexity: O(n + q)
pub fn prefixSumQueries(
    allocator: Allocator,
    array: []const i64,
    queries: []const Query,
) (RangeSumQueryError || Allocator.Error)![]i64 {
    if (array.len == 0) return RangeSumQueryError.EmptyArray;

    const prefix = try allocator.alloc(i64, array.len);
    defer allocator.free(prefix);

    prefix[0] = array[0];
    for (1..array.len) |i| {
        const next = @addWithOverflow(prefix[i - 1], array[i]);
        if (next[1] != 0) return RangeSumQueryError.Overflow;
        prefix[i] = next[0];
    }

    for (queries) |query| {
        if (query.left > query.right) return RangeSumQueryError.InvalidQuery;
        if (query.right >= array.len) return RangeSumQueryError.InvalidQuery;
    }

    const result = try allocator.alloc(i64, queries.len);

    for (queries, 0..) |query, i| {
        var res = prefix[query.right];
        if (query.left > 0) {
            const diff = @subWithOverflow(res, prefix[query.left - 1]);
            if (diff[1] != 0) return RangeSumQueryError.Overflow;
            res = diff[0];
        }
        result[i] = res;
    }

    return result;
}

test "range sum query: python examples" {
    const arr1 = [_]i64{ 1, 4, 6, 2, 61, 12 };
    const q1 = [_]Query{
        .{ .left = 2, .right = 5 },
        .{ .left = 1, .right = 5 },
        .{ .left = 3, .right = 4 },
    };
    const out1 = try prefixSumQueries(testing.allocator, &arr1, &q1);
    defer testing.allocator.free(out1);
    try testing.expectEqualSlices(i64, &[_]i64{ 81, 85, 63 }, out1);

    const arr2 = [_]i64{ 4, 2, 1, 6, 3 };
    const q2 = [_]Query{
        .{ .left = 3, .right = 4 },
        .{ .left = 1, .right = 3 },
        .{ .left = 0, .right = 2 },
    };
    const out2 = try prefixSumQueries(testing.allocator, &arr2, &q2);
    defer testing.allocator.free(out2);
    try testing.expectEqualSlices(i64, &[_]i64{ 9, 9, 7 }, out2);
}

test "range sum query: boundary and invalid queries" {
    const arr = [_]i64{42};
    const q = [_]Query{.{ .left = 0, .right = 0 }};
    const out = try prefixSumQueries(testing.allocator, &arr, &q);
    defer testing.allocator.free(out);
    try testing.expectEqualSlices(i64, &[_]i64{42}, out);

    try testing.expectError(RangeSumQueryError.EmptyArray, prefixSumQueries(testing.allocator, &[_]i64{}, &[_]Query{}));
    try testing.expectError(RangeSumQueryError.InvalidQuery, prefixSumQueries(testing.allocator, &arr, &[_]Query{.{ .left = 1, .right = 0 }}));
    try testing.expectError(RangeSumQueryError.InvalidQuery, prefixSumQueries(testing.allocator, &arr, &[_]Query{.{ .left = 0, .right = 1 }}));
}

test "range sum query: extreme many queries" {
    var arr: [2048]i64 = undefined;
    for (0..arr.len) |i| arr[i] = @intCast(i % 17);

    var qs: [1024]Query = undefined;
    for (0..qs.len) |i| {
        const left = i % (arr.len / 2);
        const right = left + (i % (arr.len / 2));
        qs[i] = .{ .left = left, .right = right };
    }

    const out = try prefixSumQueries(testing.allocator, &arr, &qs);
    defer testing.allocator.free(out);
    try testing.expectEqual(@as(usize, qs.len), out.len);
}
