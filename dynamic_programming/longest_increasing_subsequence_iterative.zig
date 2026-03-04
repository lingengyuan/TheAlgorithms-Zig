//! Longest Increasing Subsequence (Iterative O(n^2), non-strict) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/longest_increasing_subsequence_iterative.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const LisIterativeError = error{
    Overflow,
};

/// Returns one longest non-decreasing subsequence (uses <= relation),
/// matching Python reference behavior.
/// Time complexity: O(n^2), Space complexity: O(n)
pub fn longestSubsequenceIterative(
    allocator: Allocator,
    array: []const i64,
) (LisIterativeError || Allocator.Error)![]i64 {
    if (array.len == 0) return allocator.alloc(i64, 0);

    const n = array.len;
    const lengths = try allocator.alloc(usize, n);
    defer allocator.free(lengths);
    const parent = try allocator.alloc(usize, n);
    defer allocator.free(parent);

    for (0..n) |i| {
        lengths[i] = 1;
        parent[i] = i;
    }

    for (1..n) |i| {
        for (0..i) |prev| {
            if (array[prev] <= array[i]) {
                const candidate = @addWithOverflow(lengths[prev], @as(usize, 1));
                if (candidate[1] != 0) return LisIterativeError.Overflow;
                if (candidate[0] > lengths[i]) {
                    lengths[i] = candidate[0];
                    parent[i] = prev;
                }
            }
        }
    }

    var best_index: usize = 0;
    for (1..n) |i| {
        if (lengths[i] > lengths[best_index]) best_index = i;
    }

    const result = try allocator.alloc(i64, lengths[best_index]);
    var pos = result.len;
    var idx = best_index;
    while (true) {
        pos -= 1;
        result[pos] = array[idx];
        if (parent[idx] == idx) break;
        idx = parent[idx];
    }

    return result;
}

test "lis iterative: python examples" {
    const r1 = try longestSubsequenceIterative(testing.allocator, &[_]i64{ 10, 22, 9, 33, 21, 50, 41, 60, 80 });
    defer testing.allocator.free(r1);
    try testing.expectEqualSlices(i64, &[_]i64{ 10, 22, 33, 50, 60, 80 }, r1);

    const r2 = try longestSubsequenceIterative(testing.allocator, &[_]i64{ 4, 8, 7, 5, 1, 12, 2, 3, 9 });
    defer testing.allocator.free(r2);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 9 }, r2);

    const r3 = try longestSubsequenceIterative(testing.allocator, &[_]i64{ 9, 8, 7, 6, 5, 7 });
    defer testing.allocator.free(r3);
    try testing.expectEqualSlices(i64, &[_]i64{ 7, 7 }, r3);
}

test "lis iterative: boundaries" {
    const r1 = try longestSubsequenceIterative(testing.allocator, &[_]i64{ 1, 1, 1 });
    defer testing.allocator.free(r1);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 1, 1 }, r1);

    const r2 = try longestSubsequenceIterative(testing.allocator, &[_]i64{});
    defer testing.allocator.free(r2);
    try testing.expectEqual(@as(usize, 0), r2.len);
}

test "lis iterative: extreme non-decreasing array" {
    var arr: [4096]i64 = undefined;
    for (0..arr.len) |i| arr[i] = @intCast(i / 4);
    const r = try longestSubsequenceIterative(testing.allocator, &arr);
    defer testing.allocator.free(r);
    try testing.expectEqual(@as(usize, arr.len), r.len);
}
