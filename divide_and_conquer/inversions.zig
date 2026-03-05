//! Inversion Count - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/divide_and_conquer/inversions.py

const std = @import("std");
const testing = std.testing;

pub const InversionResult = struct {
    sorted: []i64,
    inversions: u64,

    pub fn deinit(self: InversionResult, allocator: std.mem.Allocator) void {
        allocator.free(self.sorted);
    }
};

fn mergeAndCount(arr: []i64, temp: []i64, left: usize, mid: usize, right: usize) u64 {
    var i = left;
    var j = mid;
    var k = left;
    var inv_count: u64 = 0;

    while (i < mid and j < right) {
        if (arr[i] <= arr[j]) {
            temp[k] = arr[i];
            i += 1;
        } else {
            temp[k] = arr[j];
            inv_count += @as(u64, @intCast(mid - i));
            j += 1;
        }
        k += 1;
    }

    while (i < mid) : (i += 1) {
        temp[k] = arr[i];
        k += 1;
    }
    while (j < right) : (j += 1) {
        temp[k] = arr[j];
        k += 1;
    }

    for (left..right) |idx| {
        arr[idx] = temp[idx];
    }

    return inv_count;
}

fn sortAndCount(arr: []i64, temp: []i64, left: usize, right: usize) u64 {
    if (right - left <= 1) return 0;

    const mid = left + (right - left) / 2;
    const left_inv = sortAndCount(arr, temp, left, mid);
    const right_inv = sortAndCount(arr, temp, mid, right);
    const cross_inv = mergeAndCount(arr, temp, left, mid, right);

    return left_inv + right_inv + cross_inv;
}

/// Counts inversions via brute force O(n^2).
pub fn countInversionsBruteForce(items: []const i64) u64 {
    var count: u64 = 0;
    for (0..items.len) |i| {
        var j = i + 1;
        while (j < items.len) : (j += 1) {
            if (items[i] > items[j]) count += 1;
        }
    }
    return count;
}

/// Returns sorted copy and inversion count via divide-and-conquer.
///
/// Time complexity: O(n log n)
/// Space complexity: O(n)
pub fn countInversionsRecursive(
    allocator: std.mem.Allocator,
    items: []const i64,
) std.mem.Allocator.Error!InversionResult {
    const sorted = try allocator.dupe(i64, items);
    errdefer allocator.free(sorted);

    const temp = try allocator.alloc(i64, sorted.len);
    defer allocator.free(temp);

    const inv_count = sortAndCount(sorted, temp, 0, sorted.len);
    return .{ .sorted = sorted, .inversions = inv_count };
}

test "inversions: python examples" {
    const alloc = testing.allocator;

    var r1 = try countInversionsRecursive(alloc, &[_]i64{ 1, 4, 2, 4, 1 });
    defer r1.deinit(alloc);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 1, 2, 4, 4 }, r1.sorted);
    try testing.expectEqual(@as(u64, 4), r1.inversions);

    var r2 = try countInversionsRecursive(alloc, &[_]i64{ 1, 1, 2, 4, 4 });
    defer r2.deinit(alloc);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 1, 2, 4, 4 }, r2.sorted);
    try testing.expectEqual(@as(u64, 0), r2.inversions);

    var r3 = try countInversionsRecursive(alloc, &[_]i64{});
    defer r3.deinit(alloc);
    try testing.expectEqual(@as(usize, 0), r3.sorted.len);
    try testing.expectEqual(@as(u64, 0), r3.inversions);
}

test "inversions: brute force matches recursive" {
    const alloc = testing.allocator;
    const arr = [_]i64{ 10, 2, 1, 5, 5, 2, 11 };

    var recursive = try countInversionsRecursive(alloc, &arr);
    defer recursive.deinit(alloc);

    const brute = countInversionsBruteForce(&arr);
    try testing.expectEqual(brute, recursive.inversions);
    try testing.expectEqual(@as(u64, 8), brute);
}

test "inversions: extreme descending array" {
    const alloc = testing.allocator;

    var arr: [64]i64 = undefined;
    for (0..arr.len) |i| {
        arr[i] = @as(i64, @intCast(arr.len - i));
    }

    var recursive = try countInversionsRecursive(alloc, arr[0..]);
    defer recursive.deinit(alloc);

    const expected = (@as(u64, arr.len) * (@as(u64, arr.len) - 1)) / 2;
    try testing.expectEqual(expected, recursive.inversions);
}
