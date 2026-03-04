//! IntroSort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/intro_sort.py

const std = @import("std");
const testing = std.testing;

fn insertionSortRange(comptime T: type, arr: []T, start: usize, end: usize) void {
    if (end <= start + 1) return;
    var i = start + 1;
    while (i < end) : (i += 1) {
        const value = arr[i];
        var j = i;
        while (j > start and value < arr[j - 1]) : (j -= 1) {
            arr[j] = arr[j - 1];
        }
        arr[j] = value;
    }
}

fn heapify(comptime T: type, arr: []T, n: usize, i: usize) void {
    var largest = i;
    while (true) {
        const left = 2 * largest + 1;
        const right = 2 * largest + 2;
        var candidate = largest;

        if (left < n and arr[left] > arr[candidate]) candidate = left;
        if (right < n and arr[right] > arr[candidate]) candidate = right;
        if (candidate == largest) break;
        std.mem.swap(T, &arr[largest], &arr[candidate]);
        largest = candidate;
    }
}

fn heapSortRange(comptime T: type, arr: []T, start: usize, end: usize) void {
    if (end <= start + 1) return;
    const sub = arr[start..end];
    const n = sub.len;

    var i = n / 2;
    while (i > 0) {
        i -= 1;
        heapify(T, sub, n, i);
    }

    i = n;
    while (i > 1) {
        i -= 1;
        std.mem.swap(T, &sub[0], &sub[i]);
        heapify(T, sub, i, 0);
    }
}

fn medianOf3(comptime T: type, a: T, b: T, c: T) T {
    if ((a > b) != (a > c)) return a;
    if ((b > a) != (b > c)) return b;
    return c;
}

fn partition(comptime T: type, arr: []T, low: usize, high: usize, pivot: T) usize {
    var i = low;
    var j = high;
    while (true) {
        while (arr[i] < pivot) : (i += 1) {}
        j -= 1;
        while (pivot < arr[j]) {
            j -= 1;
        }
        if (i >= j) return i;
        std.mem.swap(T, &arr[i], &arr[j]);
        i += 1;
    }
}

fn introSortRange(comptime T: type, arr: []T, start_in: usize, end_in: usize, size_threshold: usize, max_depth_in: usize) void {
    const start = start_in;
    var end = end_in;
    var max_depth = max_depth_in;

    while (end - start > size_threshold) {
        if (max_depth == 0) {
            heapSortRange(T, arr, start, end);
            return;
        }
        max_depth -= 1;
        const mid = start + ((end - start) / 2);
        const pivot = medianOf3(T, arr[start], arr[mid], arr[end - 1]);
        const p = partition(T, arr, start, end, pivot);
        introSortRange(T, arr, p, end, size_threshold, max_depth);
        end = p;
    }

    insertionSortRange(T, arr, start, end);
}

fn depthLimit(n: usize) usize {
    if (n <= 1) return 0;
    var power: usize = 1;
    var lg: usize = 0;
    while (power < n) : (lg += 1) {
        power <<= 1;
    }
    return 2 * lg;
}

/// In-place introsort: quicksort + heapsort fallback + insertion sort for tiny ranges.
/// Time complexity: O(n log n) average and worst
/// Space complexity: O(log n) recursion
pub fn introSort(comptime T: type, arr: []T) void {
    if (arr.len == 0) return;
    const max_depth = depthLimit(arr.len);
    introSortRange(T, arr, 0, arr.len, 16, max_depth);
}

fn expectSortedAscending(comptime T: type, arr: []const T) !void {
    if (arr.len <= 1) return;
    for (1..arr.len) |i| try testing.expect(arr[i - 1] <= arr[i]);
}

test "intro sort: python reference examples" {
    var a1 = [_]i64{ 4, 2, 6, 8, 1, 7, 8, 22, 14, 56, 27, 79, 23, 45, 14, 12 };
    introSort(i64, &a1);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 4, 6, 7, 8, 8, 12, 14, 14, 22, 23, 27, 45, 56, 79 }, &a1);

    var a2 = [_]i64{ -1, -5, -3, -13, -44 };
    introSort(i64, &a2);
    try testing.expectEqualSlices(i64, &[_]i64{ -44, -13, -5, -3, -1 }, &a2);

    var a3 = [_]f64{ 1.7, 1.0, 3.3, 2.1, 0.3 };
    introSort(f64, &a3);
    try testing.expectEqualSlices(f64, &[_]f64{ 0.3, 1.0, 1.7, 2.1, 3.3 }, &a3);

    var a4 = [_]u8{ 'd', 'a', 'b', 'e', 'c' };
    introSort(u8, &a4);
    try testing.expectEqualSlices(u8, &[_]u8{ 'a', 'b', 'c', 'd', 'e' }, &a4);
}

test "intro sort: edge cases" {
    var empty = [_]i32{};
    introSort(i32, &empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    var one = [_]i32{5};
    introSort(i32, &one);
    try testing.expectEqualSlices(i32, &[_]i32{5}, &one);
}

test "intro sort: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 80_000;
    const arr = try alloc.alloc(i64, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(4242);
    var random = prng.random();
    for (arr) |*v| v.* = random.intRangeAtMost(i64, -9_000_000, 9_000_000);

    introSort(i64, arr);
    try expectSortedAscending(i64, arr);
}
