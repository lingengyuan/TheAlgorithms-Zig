//! Tim Sort (Educational Variant) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/tim_sort.py

const std = @import("std");
const testing = std.testing;

fn binarySearch(lst: []const i64, item: i64, start: usize, end: usize) usize {
    if (lst.len == 0) return 0;
    if (start == end) return if (lst[start] > item) start else start + 1;
    if (start > end) return start;

    const mid = (start + end) / 2;
    if (lst[mid] < item) {
        return binarySearch(lst, item, mid + 1, end);
    } else if (lst[mid] > item) {
        if (mid == 0) return 0;
        return binarySearch(lst, item, start, mid - 1);
    } else {
        return mid;
    }
}

fn insertionSort(allocator: std.mem.Allocator, input: []const i64) ![]i64 {
    var lst = try allocator.alloc(i64, input.len);
    @memcpy(lst, input);

    const length = lst.len;
    var index: usize = 1;
    while (index < length) : (index += 1) {
        const value = lst[index];
        const pos = binarySearch(lst[0..index], value, 0, index - 1);

        var j = index;
        while (j > pos) : (j -= 1) {
            lst[j] = lst[j - 1];
        }
        lst[pos] = value;
    }
    return lst;
}

fn merge(allocator: std.mem.Allocator, left: []const i64, right: []const i64) ![]i64 {
    const out = try allocator.alloc(i64, left.len + right.len);
    var li: usize = 0;
    var ri: usize = 0;
    var i: usize = 0;

    while (li < left.len and ri < right.len) : (i += 1) {
        if (left[li] < right[ri]) {
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

/// Educational tim-sort-like implementation:
/// detect monotonic runs, insertion-sort each run, then merge.
/// Caller owns returned slice.
pub fn timSort(allocator: std.mem.Allocator, lst: []const i64) ![]i64 {
    if (lst.len == 0) return try allocator.alloc(i64, 0);
    if (lst.len == 1) {
        const one = try allocator.alloc(i64, 1);
        one[0] = lst[0];
        return one;
    }

    var runs = std.ArrayListUnmanaged(struct { start: usize, end: usize }){};
    defer runs.deinit(allocator);

    var run_start: usize = 0;
    var i: usize = 1;
    while (i < lst.len) : (i += 1) {
        if (lst[i] < lst[i - 1]) {
            try runs.append(allocator, .{ .start = run_start, .end = i });
            run_start = i;
        }
    }
    try runs.append(allocator, .{ .start = run_start, .end = lst.len });

    var sorted_array = try allocator.alloc(i64, 0);
    for (runs.items) |run| {
        const sorted_run = try insertionSort(allocator, lst[run.start..run.end]);
        defer allocator.free(sorted_run);

        const merged = try merge(allocator, sorted_array, sorted_run);
        allocator.free(sorted_array);
        sorted_array = merged;
    }

    return sorted_array;
}

test "tim sort: python reference examples" {
    const alloc = testing.allocator;

    const r1 = try timSort(alloc, &[_]i64{ 3, 2, 1 });
    defer alloc.free(r1);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3 }, r1);

    const r2 = try timSort(alloc, &[_]i64{ 1, 0, 3, 2, -1, -1 });
    defer alloc.free(r2);
    try testing.expectEqualSlices(i64, &[_]i64{ -1, -1, 0, 1, 2, 3 }, r2);

    const r3 = try timSort(alloc, &[_]i64{ 6, 5, 4, 3, 2, 1, 0 });
    defer alloc.free(r3);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 1, 2, 3, 4, 5, 6 }, r3);
}

test "tim sort: edge cases" {
    const alloc = testing.allocator;

    const empty = try timSort(alloc, &[_]i64{});
    defer alloc.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const one = try timSort(alloc, &[_]i64{5});
    defer alloc.free(one);
    try testing.expectEqualSlices(i64, &[_]i64{5}, one);
}

test "tim sort: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 50_000;
    const arr = try alloc.alloc(i64, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(9087);
    var random = prng.random();
    for (arr) |*v| v.* = random.intRangeAtMost(i64, -500_000, 500_000);

    const sorted = try timSort(alloc, arr);
    defer alloc.free(sorted);
    for (1..sorted.len) |k| try testing.expect(sorted[k - 1] <= sorted[k]);
}
