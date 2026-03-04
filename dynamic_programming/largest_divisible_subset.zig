//! Largest Divisible Subset - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/largest_divisible_subset.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const LargestDivisibleSubsetError = error{
    Overflow,
};

/// Finds a largest subset such that for every pair (x, y),
/// either x divides y or y divides x.
/// The returned order follows Python reconstruction order (largest to smallest).
/// Time complexity: O(n^2), Space complexity: O(n)
pub fn largestDivisibleSubset(
    allocator: Allocator,
    items: []const i64,
) (LargestDivisibleSubsetError || Allocator.Error)![]i64 {
    if (items.len == 0) return allocator.alloc(i64, 0);

    const sorted = try allocator.dupe(i64, items);
    defer allocator.free(sorted);
    std.mem.sort(i64, sorted, {}, std.sort.asc(i64));

    const n = sorted.len;
    const lengths = try allocator.alloc(usize, n);
    defer allocator.free(lengths);
    const parent = try allocator.alloc(usize, n);
    defer allocator.free(parent);

    @memset(lengths, 1);
    for (0..n) |i| parent[i] = i;

    for (0..n) |i| {
        for (0..i) |prev| {
            // Python condition: ((prev != 0 and item % prev) == 0)
            // In Python, when prev == 0, (False == 0) is True.
            const divisible = if (sorted[prev] == 0) true else (@mod(sorted[i], sorted[prev]) == 0);
            if (!divisible) continue;

            const candidate = @addWithOverflow(lengths[prev], @as(usize, 1));
            if (candidate[1] != 0) return LargestDivisibleSubsetError.Overflow;
            if (candidate[0] > lengths[i]) {
                lengths[i] = candidate[0];
                parent[i] = prev;
            }
        }
    }

    var best_len: usize = 0;
    var last_index: usize = 0;
    for (0..n) |i| {
        if (lengths[i] > best_len) {
            best_len = lengths[i];
            last_index = i;
        }
    }

    const result = try allocator.alloc(i64, best_len);
    var pos: usize = 0;
    var idx = last_index;
    while (true) {
        result[pos] = sorted[idx];
        pos += 1;
        if (parent[idx] == idx) break;
        idx = parent[idx];
    }

    return result;
}

test "largest divisible subset: python examples" {
    const result1 = try largestDivisibleSubset(testing.allocator, &[_]i64{ 1, 16, 7, 8, 4 });
    defer testing.allocator.free(result1);
    try testing.expectEqualSlices(i64, &[_]i64{ 16, 8, 4, 1 }, result1);

    const result2 = try largestDivisibleSubset(testing.allocator, &[_]i64{ 1, 2, 3 });
    defer testing.allocator.free(result2);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 1 }, result2);

    const result3 = try largestDivisibleSubset(testing.allocator, &[_]i64{ -1, -2, -3 });
    defer testing.allocator.free(result3);
    try testing.expectEqualSlices(i64, &[_]i64{-3}, result3);
}

test "largest divisible subset: duplicates and zeros" {
    const result1 = try largestDivisibleSubset(testing.allocator, &[_]i64{ 1, 1, 1 });
    defer testing.allocator.free(result1);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 1, 1 }, result1);

    const result2 = try largestDivisibleSubset(testing.allocator, &[_]i64{ 0, 0, 0 });
    defer testing.allocator.free(result2);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 0, 0 }, result2);

    const result3 = try largestDivisibleSubset(testing.allocator, &[_]i64{ -1, -1, -1 });
    defer testing.allocator.free(result3);
    try testing.expectEqualSlices(i64, &[_]i64{ -1, -1, -1 }, result3);
}

test "largest divisible subset: empty and power-of-two extreme" {
    const empty = try largestDivisibleSubset(testing.allocator, &[_]i64{});
    defer testing.allocator.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    var values: [32]i64 = undefined;
    for (0..values.len) |i| values[i] = @as(i64, 1) << @intCast(i);
    const result = try largestDivisibleSubset(testing.allocator, &values);
    defer testing.allocator.free(result);
    try testing.expectEqual(@as(usize, 32), result.len);
    try testing.expectEqual(@as(i64, 1) << 31, result[0]);
    try testing.expectEqual(@as(i64, 1), result[result.len - 1]);
}
