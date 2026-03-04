//! Pairs With Given Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/arrays/pairs_with_given_sum.py

const std = @import("std");
const testing = std.testing;

/// Counts pairs (i < j) where arr[i] + arr[j] == req_sum.
/// Time complexity: O(n), Space complexity: O(n)
pub fn pairsWithSum(allocator: std.mem.Allocator, arr: []const i64, req_sum: i64) !u64 {
    var counts = std.AutoHashMap(i64, u64).init(allocator);
    defer counts.deinit();

    var total: u64 = 0;
    for (arr) |v| {
        const need = req_sum - v;
        if (counts.get(need)) |c| {
            total += c;
        }

        const existing = counts.get(v) orelse 0;
        try counts.put(v, existing + 1);
    }

    return total;
}

test "pairs with given sum: python examples" {
    try testing.expectEqual(@as(u64, 2), try pairsWithSum(testing.allocator, &[_]i64{ 1, 5, 7, 1 }, 6));
    try testing.expectEqual(@as(u64, 28), try pairsWithSum(testing.allocator, &[_]i64{ 1, 1, 1, 1, 1, 1, 1, 1 }, 2));
    try testing.expectEqual(@as(u64, 4), try pairsWithSum(testing.allocator, &[_]i64{ 1, 7, 6, 2, 5, 4, 3, 1, 9, 8 }, 7));
}

test "pairs with given sum: empty and extreme" {
    try testing.expectEqual(@as(u64, 0), try pairsWithSum(testing.allocator, &[_]i64{}, 10));

    const n: usize = 100_000;
    const values = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(values);
    @memset(values, 1);

    const count = try pairsWithSum(testing.allocator, values, 2);
    // n choose 2
    try testing.expectEqual(@as(u64, @intCast((n * (n - 1)) / 2)), count);
}
