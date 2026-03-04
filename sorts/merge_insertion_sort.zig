//! Merge-Insertion Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/merge_insertion_sort.py

const std = @import("std");
const testing = std.testing;

const Pair = struct {
    first: i64,
    second: i64,
};

fn binarySearchInsert(list: *std.ArrayListUnmanaged(i64), allocator: std.mem.Allocator, item: i64) !void {
    var left: usize = 0;
    var right: usize = list.items.len;
    while (left < right) {
        const mid = left + (right - left) / 2;
        if (list.items[mid] < item) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    try list.insert(allocator, left, item);
}

fn mergePairs(allocator: std.mem.Allocator, left: []const Pair, right: []const Pair) ![]Pair {
    const out = try allocator.alloc(Pair, left.len + right.len);
    var i: usize = 0;
    var li: usize = 0;
    var ri: usize = 0;

    while (li < left.len and ri < right.len) : (i += 1) {
        if (left[li].first < right[ri].first) {
            out[i] = left[li];
            li += 1;
        } else {
            out[i] = right[ri];
            ri += 1;
        }
    }
    while (li < left.len) : (li += 1) {
        out[i] = left[li];
        i += 1;
    }
    while (ri < right.len) : (ri += 1) {
        out[i] = right[ri];
        i += 1;
    }
    return out;
}

fn sortPairs(allocator: std.mem.Allocator, pairs: []const Pair) ![]Pair {
    if (pairs.len <= 1) {
        const out = try allocator.alloc(Pair, pairs.len);
        @memcpy(out, pairs);
        return out;
    }

    const mid = pairs.len / 2;
    const left = try sortPairs(allocator, pairs[0..mid]);
    defer allocator.free(left);
    const right = try sortPairs(allocator, pairs[mid..]);
    defer allocator.free(right);
    return try mergePairs(allocator, left, right);
}

/// Returns sorted copy using merge-insertion strategy.
/// Caller owns returned slice.
/// Time complexity: O(n log n), Space complexity: O(n)
pub fn mergeInsertionSort(allocator: std.mem.Allocator, collection: []const i64) ![]i64 {
    if (collection.len <= 1) {
        const out = try allocator.alloc(i64, collection.len);
        @memcpy(out, collection);
        return out;
    }

    var pairs = std.ArrayListUnmanaged(Pair){};
    defer pairs.deinit(allocator);

    var has_last_odd = false;
    var last_odd: i64 = 0;

    var i: usize = 0;
    while (i < collection.len) : (i += 2) {
        if (i + 1 >= collection.len) {
            has_last_odd = true;
            last_odd = collection[i];
        } else {
            const a = collection[i];
            const b = collection[i + 1];
            if (a <= b) {
                try pairs.append(allocator, .{ .first = a, .second = b });
            } else {
                try pairs.append(allocator, .{ .first = b, .second = a });
            }
        }
    }

    const sorted_pairs = try sortPairs(allocator, pairs.items);
    defer allocator.free(sorted_pairs);

    var result = std.ArrayListUnmanaged(i64){};
    errdefer result.deinit(allocator);

    if (sorted_pairs.len > 0) {
        for (sorted_pairs) |p| try result.append(allocator, p.first);
        try result.append(allocator, sorted_pairs[sorted_pairs.len - 1].second);
        for (sorted_pairs[0 .. sorted_pairs.len - 1]) |p| {
            try binarySearchInsert(&result, allocator, p.second);
        }
    }

    if (has_last_odd) {
        try binarySearchInsert(&result, allocator, last_odd);
    }

    return try result.toOwnedSlice(allocator);
}

test "merge insertion sort: python reference examples" {
    const alloc = testing.allocator;

    const r1 = try mergeInsertionSort(alloc, &[_]i64{ 0, 5, 3, 2, 2 });
    defer alloc.free(r1);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 2, 2, 3, 5 }, r1);

    const r2 = try mergeInsertionSort(alloc, &[_]i64{99});
    defer alloc.free(r2);
    try testing.expectEqualSlices(i64, &[_]i64{99}, r2);

    const r3 = try mergeInsertionSort(alloc, &[_]i64{ -2, -5, -45 });
    defer alloc.free(r3);
    try testing.expectEqualSlices(i64, &[_]i64{ -45, -5, -2 }, r3);
}

test "merge insertion sort: edge and permutation-like check" {
    const alloc = testing.allocator;

    const r1 = try mergeInsertionSort(alloc, &[_]i64{});
    defer alloc.free(r1);
    try testing.expectEqual(@as(usize, 0), r1.len);

    const r2 = try mergeInsertionSort(alloc, &[_]i64{ 4, 1, 0, 3, 2 });
    defer alloc.free(r2);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 1, 2, 3, 4 }, r2);
}

test "merge insertion sort: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 12_000;
    const arr = try alloc.alloc(i64, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(909);
    var random = prng.random();
    for (arr) |*v| v.* = random.intRangeAtMost(i64, -1_000_000, 1_000_000);

    const out = try mergeInsertionSort(alloc, arr);
    defer alloc.free(out);
    for (1..out.len) |idx| try testing.expect(out[idx - 1] <= out[idx]);
}
