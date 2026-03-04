//! Strand Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/strand_sort.py

const std = @import("std");
const testing = std.testing;

fn lessForReverse(comptime T: type, reverse: bool, a: T, b: T) bool {
    return if (reverse) a < b else a > b;
}

fn mergeSortedSlices(comptime T: type, allocator: std.mem.Allocator, left: []const T, right: []const T, reverse: bool) ![]T {
    const out = try allocator.alloc(T, left.len + right.len);
    var i: usize = 0;
    var li: usize = 0;
    var ri: usize = 0;

    while (li < left.len and ri < right.len) : (i += 1) {
        const pick_left = if (reverse) left[li] > right[ri] else left[li] < right[ri];
        if (pick_left) {
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

/// Returns sorted copy using strand-sort style extraction.
/// Caller owns returned slice.
/// Time complexity: O(n²) average, Space complexity: O(n)
pub fn strandSort(comptime T: type, allocator: std.mem.Allocator, input: []const T, reverse: bool) ![]T {
    if (input.len == 0) return try allocator.alloc(T, 0);

    var remaining = try allocator.alloc(T, input.len);
    defer allocator.free(remaining);
    @memcpy(remaining, input);
    var remaining_len = input.len;

    var solution = try allocator.alloc(T, 0);
    errdefer allocator.free(solution);

    while (remaining_len > 0) {
        var sub = std.ArrayListUnmanaged(T){};
        defer sub.deinit(allocator);
        var next_remaining = std.ArrayListUnmanaged(T){};
        defer next_remaining.deinit(allocator);

        try sub.append(allocator, remaining[0]);
        for (remaining[1..remaining_len]) |item| {
            if (lessForReverse(T, reverse, item, sub.items[sub.items.len - 1])) {
                try sub.append(allocator, item);
            } else {
                try next_remaining.append(allocator, item);
            }
        }

        const merged = try mergeSortedSlices(T, allocator, solution, sub.items, reverse);
        allocator.free(solution);
        solution = merged;

        remaining_len = next_remaining.items.len;
        if (remaining_len > 0) {
            @memcpy(remaining[0..remaining_len], next_remaining.items);
        }
    }

    return solution;
}

test "strand sort: python reference examples" {
    const alloc = testing.allocator;

    const a1 = [_]i32{ 4, 2, 5, 3, 0, 1 };
    const r1 = try strandSort(i32, alloc, &a1, false);
    defer alloc.free(r1);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 1, 2, 3, 4, 5 }, r1);

    const r2 = try strandSort(i32, alloc, &a1, true);
    defer alloc.free(r2);
    try testing.expectEqualSlices(i32, &[_]i32{ 5, 4, 3, 2, 1, 0 }, r2);
}

test "strand sort: edge cases" {
    const alloc = testing.allocator;

    const empty = [_]i32{};
    const r1 = try strandSort(i32, alloc, &empty, false);
    defer alloc.free(r1);
    try testing.expectEqual(@as(usize, 0), r1.len);

    const one = [_]i32{7};
    const r2 = try strandSort(i32, alloc, &one, false);
    defer alloc.free(r2);
    try testing.expectEqualSlices(i32, &[_]i32{7}, r2);
}

test "strand sort: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 10_000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(64);
    var random = prng.random();
    for (arr) |*v| v.* = random.intRangeAtMost(i32, -500_000, 500_000);

    const out = try strandSort(i32, alloc, arr, false);
    defer alloc.free(out);
    for (1..out.len) |i| try testing.expect(out[i - 1] <= out[i]);
}
