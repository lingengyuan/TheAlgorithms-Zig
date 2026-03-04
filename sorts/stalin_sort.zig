//! Stalin Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/stalin_sort.py

const std = @import("std");
const testing = std.testing;

/// Returns non-decreasing subsequence by discarding out-of-order elements.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn stalinSort(comptime T: type, allocator: std.mem.Allocator, sequence: []const T) ![]T {
    if (sequence.len == 0) return try allocator.alloc(T, 0);

    var result = std.ArrayListUnmanaged(T){};
    errdefer result.deinit(allocator);

    try result.append(allocator, sequence[0]);
    for (sequence[1..]) |element| {
        if (element >= result.items[result.items.len - 1]) {
            try result.append(allocator, element);
        }
    }

    return try result.toOwnedSlice(allocator);
}

test "stalin sort: python reference examples" {
    const alloc = testing.allocator;

    const r1 = try stalinSort(i32, alloc, &[_]i32{ 4, 3, 5, 2, 1, 7 });
    defer alloc.free(r1);
    try testing.expectEqualSlices(i32, &[_]i32{ 4, 5, 7 }, r1);

    const r2 = try stalinSort(i32, alloc, &[_]i32{ 1, 2, 3, 4 });
    defer alloc.free(r2);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4 }, r2);

    const r3 = try stalinSort(i32, alloc, &[_]i32{ 4, 5, 5, 2, 3 });
    defer alloc.free(r3);
    try testing.expectEqualSlices(i32, &[_]i32{ 4, 5, 5 }, r3);

    const r4 = try stalinSort(i32, alloc, &[_]i32{ 6, 11, 12, 4, 1, 5 });
    defer alloc.free(r4);
    try testing.expectEqualSlices(i32, &[_]i32{ 6, 11, 12 }, r4);
}

test "stalin sort: edge cases" {
    const alloc = testing.allocator;

    const r1 = try stalinSort(i32, alloc, &[_]i32{ 5, 0, 4, 3 });
    defer alloc.free(r1);
    try testing.expectEqualSlices(i32, &[_]i32{5}, r1);

    const r2 = try stalinSort(i32, alloc, &[_]i32{ 1, 2, 8, 7, 6 });
    defer alloc.free(r2);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 8 }, r2);

    const r3 = try stalinSort(i32, alloc, &[_]i32{});
    defer alloc.free(r3);
    try testing.expectEqual(@as(usize, 0), r3.len);
}

test "stalin sort: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 50_000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(2025);
    var random = prng.random();
    for (arr) |*v| {
        v.* = random.intRangeAtMost(i32, -50_000, 50_000);
    }

    const out = try stalinSort(i32, alloc, arr);
    defer alloc.free(out);
    for (1..out.len) |i| {
        try testing.expect(out[i - 1] <= out[i]);
    }
}
