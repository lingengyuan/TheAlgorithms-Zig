//! Kth Order Statistic - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/divide_and_conquer/kth_order_statistic.py

const std = @import("std");
const testing = std.testing;

pub const KthError = error{ EmptyInput, KOutOfRange };

fn kthRecursive(
    allocator: std.mem.Allocator,
    items: []const i64,
    k: usize,
) std.mem.Allocator.Error!i64 {
    if (items.len == 1) return items[0];

    const pivot = items[items.len / 2];

    var small_count: usize = 0;
    var equal_count: usize = 0;
    var big_count: usize = 0;

    for (items) |value| {
        if (value < pivot) {
            small_count += 1;
        } else if (value > pivot) {
            big_count += 1;
        } else {
            equal_count += 1;
        }
    }

    if (k <= small_count) {
        const small = try allocator.alloc(i64, small_count);
        defer allocator.free(small);

        var idx: usize = 0;
        for (items) |value| {
            if (value < pivot) {
                small[idx] = value;
                idx += 1;
            }
        }
        return kthRecursive(allocator, small, k);
    }

    if (k <= small_count + equal_count) {
        return pivot;
    }

    const new_k = k - small_count - equal_count;
    const big = try allocator.alloc(i64, big_count);
    defer allocator.free(big);

    var idx: usize = 0;
    for (items) |value| {
        if (value > pivot) {
            big[idx] = value;
            idx += 1;
        }
    }

    return kthRecursive(allocator, big, new_k);
}

/// Returns the k-th smallest element (1-based index).
///
/// Time complexity: O(n) average, O(n^2) worst
/// Space complexity: O(n) average (temporary partitions)
pub fn kthNumber(
    allocator: std.mem.Allocator,
    items: []const i64,
    k: usize,
) (KthError || std.mem.Allocator.Error)!i64 {
    if (items.len == 0) return KthError.EmptyInput;
    if (k == 0 or k > items.len) return KthError.KOutOfRange;
    return kthRecursive(allocator, items, k);
}

test "kth order statistic: python examples" {
    const alloc = testing.allocator;

    try testing.expectEqual(@as(i64, 3), try kthNumber(alloc, &[_]i64{ 2, 1, 3, 4, 5 }, 3));
    try testing.expectEqual(@as(i64, 1), try kthNumber(alloc, &[_]i64{ 2, 1, 3, 4, 5 }, 1));
    try testing.expectEqual(@as(i64, 5), try kthNumber(alloc, &[_]i64{ 2, 1, 3, 4, 5 }, 5));
    try testing.expectEqual(@as(i64, 3), try kthNumber(alloc, &[_]i64{ 3, 2, 5, 6, 7, 8 }, 2));
    try testing.expectEqual(@as(i64, 43), try kthNumber(alloc, &[_]i64{ 25, 21, 98, 100, 76, 22, 43, 60, 89, 87 }, 4));
}

test "kth order statistic: duplicates and boundaries" {
    const alloc = testing.allocator;

    try testing.expectEqual(@as(i64, 5), try kthNumber(alloc, &[_]i64{ 5, 1, 5, 5, 2 }, 4));
    try testing.expectEqual(@as(i64, -10), try kthNumber(alloc, &[_]i64{-10}, 1));
}

test "kth order statistic: invalid input" {
    const alloc = testing.allocator;

    try testing.expectError(KthError.EmptyInput, kthNumber(alloc, &[_]i64{}, 1));
    try testing.expectError(KthError.KOutOfRange, kthNumber(alloc, &[_]i64{ 1, 2, 3 }, 0));
    try testing.expectError(KthError.KOutOfRange, kthNumber(alloc, &[_]i64{ 1, 2, 3 }, 4));
}
