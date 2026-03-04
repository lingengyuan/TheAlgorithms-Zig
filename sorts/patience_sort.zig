//! Patience Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/patience_sort.py

const std = @import("std");
const testing = std.testing;

const Pile = struct {
    items: std.ArrayListUnmanaged(i64) = .{},
};

fn deinitPiles(allocator: std.mem.Allocator, piles: []Pile) void {
    for (piles) |*p| p.items.deinit(allocator);
}

/// In-place patience sort for integers.
/// Time complexity: O(n log n) pile building + O(n * p) merge scan, p=#piles
/// Space complexity: O(n)
pub fn patienceSort(allocator: std.mem.Allocator, collection: []i64) !void {
    if (collection.len <= 1) return;

    var piles = std.ArrayListUnmanaged(Pile){};
    defer {
        deinitPiles(allocator, piles.items);
        piles.deinit(allocator);
    }

    // Build piles with bisect-left on pile tops.
    for (collection) |element| {
        var left: usize = 0;
        var right: usize = piles.items.len;
        while (left < right) {
            const mid = left + (right - left) / 2;
            const top = piles.items[mid].items.items[piles.items[mid].items.items.len - 1];
            if (top < element) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        if (left < piles.items.len) {
            try piles.items[left].items.append(allocator, element);
        } else {
            var p = Pile{};
            try p.items.append(allocator, element);
            try piles.append(allocator, p);
        }
    }

    // Merge by repeatedly taking smallest exposed card (end of each pile).
    var write_index: usize = 0;
    while (write_index < collection.len) : (write_index += 1) {
        var best_pile: ?usize = null;
        var best_value: i64 = 0;

        for (piles.items, 0..) |p, idx| {
            if (p.items.items.len == 0) continue;
            const candidate = p.items.items[p.items.items.len - 1];
            if (best_pile == null or candidate < best_value) {
                best_pile = idx;
                best_value = candidate;
            }
        }

        const chosen = best_pile.?;
        collection[write_index] = best_value;
        _ = piles.items[chosen].items.pop();
    }
}

test "patience sort: python reference examples" {
    const alloc = testing.allocator;

    var a1 = [_]i64{ 1, 9, 5, 21, 17, 6 };
    try patienceSort(alloc, &a1);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 5, 6, 9, 17, 21 }, &a1);

    var a2 = [_]i64{};
    try patienceSort(alloc, &a2);
    try testing.expectEqual(@as(usize, 0), a2.len);

    var a3 = [_]i64{ -3, -17, -48 };
    try patienceSort(alloc, &a3);
    try testing.expectEqualSlices(i64, &[_]i64{ -48, -17, -3 }, &a3);
}

test "patience sort: edge cases" {
    const alloc = testing.allocator;

    var one = [_]i64{4};
    try patienceSort(alloc, &one);
    try testing.expectEqualSlices(i64, &[_]i64{4}, &one);

    var dup = [_]i64{ 3, 3, 3, 3, 3 };
    try patienceSort(alloc, &dup);
    try testing.expectEqualSlices(i64, &[_]i64{ 3, 3, 3, 3, 3 }, &dup);
}

test "patience sort: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 30_000;
    const arr = try alloc.alloc(i64, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(1001);
    var random = prng.random();
    for (arr) |*v| v.* = random.intRangeAtMost(i64, -1_000_000, 1_000_000);

    try patienceSort(alloc, arr);
    for (1..arr.len) |i| try testing.expect(arr[i - 1] <= arr[i]);
}
