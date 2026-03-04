//! Unknown Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/unknown_sort.py

const std = @import("std");
const testing = std.testing;

fn removeFirst(list: *std.ArrayListUnmanaged(i64), value: i64) !void {
    for (list.items, 0..) |v, i| {
        if (v == value) {
            _ = list.orderedRemove(i);
            return;
        }
    }
    return error.NotFound;
}

/// Returns a sorted copy using min/max extraction strategy from Python reference.
/// Caller owns returned slice.
/// Time complexity: O(n²), Space complexity: O(n)
pub fn unknownSort(allocator: std.mem.Allocator, input: []const i64) ![]i64 {
    var collection = std.ArrayListUnmanaged(i64){};
    defer collection.deinit(allocator);
    try collection.appendSlice(allocator, input);

    var start = std.ArrayListUnmanaged(i64){};
    defer start.deinit(allocator);
    var end = std.ArrayListUnmanaged(i64){};
    defer end.deinit(allocator);

    while (collection.items.len > 1) {
        var min_one = collection.items[0];
        var max_one = collection.items[0];
        for (collection.items[1..]) |v| {
            if (v < min_one) min_one = v;
            if (v > max_one) max_one = v;
        }

        try start.append(allocator, min_one);
        try end.append(allocator, max_one);
        try removeFirst(&collection, min_one);
        try removeFirst(&collection, max_one);
    }

    std.mem.reverse(i64, end.items);
    const out = try allocator.alloc(i64, start.items.len + collection.items.len + end.items.len);
    var idx: usize = 0;
    @memcpy(out[idx..][0..start.items.len], start.items);
    idx += start.items.len;
    if (collection.items.len == 1) {
        out[idx] = collection.items[0];
        idx += 1;
    }
    @memcpy(out[idx..][0..end.items.len], end.items);
    return out;
}

test "unknown sort: python reference examples" {
    const alloc = testing.allocator;

    const r1 = try unknownSort(alloc, &[_]i64{ 0, 5, 3, 2, 2 });
    defer alloc.free(r1);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 2, 2, 3, 5 }, r1);

    const r2 = try unknownSort(alloc, &[_]i64{});
    defer alloc.free(r2);
    try testing.expectEqual(@as(usize, 0), r2.len);

    const r3 = try unknownSort(alloc, &[_]i64{ -2, -5, -45 });
    defer alloc.free(r3);
    try testing.expectEqualSlices(i64, &[_]i64{ -45, -5, -2 }, r3);
}

test "unknown sort: edge and duplicate-heavy cases" {
    const alloc = testing.allocator;
    const r = try unknownSort(alloc, &[_]i64{ 5, 5, 5, 5, 5, 1, 1, 2, 2 });
    defer alloc.free(r);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 1, 2, 2, 5, 5, 5, 5, 5 }, r);
}

test "unknown sort: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 8_000;
    const arr = try alloc.alloc(i64, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(404);
    var random = prng.random();
    for (arr) |*v| v.* = random.intRangeAtMost(i64, -100_000, 100_000);

    const out = try unknownSort(alloc, arr);
    defer alloc.free(out);
    for (1..out.len) |i| try testing.expect(out[i - 1] <= out[i]);
}
