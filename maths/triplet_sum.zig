//! Triplet Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/triplet_sum.py

const std = @import("std");
const testing = std.testing;

pub const Triplet = [3]i64;

/// Naive O(n^3) triplet search.
/// Returns sorted triplet or (0,0,0) when not found.
/// Time complexity: O(n^3), Space complexity: O(1)
pub fn tripletSum1(arr: []const i64, target: i64) Triplet {
    var i: usize = 0;
    while (i < arr.len) : (i += 1) {
        var j: usize = 0;
        while (j < arr.len) : (j += 1) {
            if (j == i) continue;
            var k: usize = 0;
            while (k < arr.len) : (k += 1) {
                if (k == i or k == j) continue;
                const sum = @as(i128, arr[i]) + arr[j] + arr[k];
                if (sum == target) {
                    var out: Triplet = .{ arr[i], arr[j], arr[k] };
                    sortTriplet(&out);
                    return out;
                }
            }
        }
    }
    return .{ 0, 0, 0 };
}

/// Optimized O(n^2) triplet search.
/// Returns sorted triplet or (0,0,0) when not found.
/// Time complexity: O(n^2), Space complexity: O(n)
pub fn tripletSum2(allocator: std.mem.Allocator, arr: []const i64, target: i64) std.mem.Allocator.Error!Triplet {
    if (arr.len < 3) return .{ 0, 0, 0 };

    const sorted = try allocator.alloc(i64, arr.len);
    defer allocator.free(sorted);
    @memcpy(sorted, arr);
    std.mem.sort(i64, sorted, {}, std.sort.asc(i64));

    var i: usize = 0;
    while (i + 1 < sorted.len) : (i += 1) {
        var left = i + 1;
        var right = sorted.len - 1;
        while (left < right) {
            const sum = @as(i128, sorted[i]) + sorted[left] + sorted[right];
            if (sum == target) return .{ sorted[i], sorted[left], sorted[right] };
            if (sum < target) {
                left += 1;
            } else {
                right -= 1;
            }
        }
    }
    return .{ 0, 0, 0 };
}

fn sortTriplet(triple: *Triplet) void {
    if (triple[0] > triple[1]) std.mem.swap(i64, &triple[0], &triple[1]);
    if (triple[1] > triple[2]) std.mem.swap(i64, &triple[1], &triple[2]);
    if (triple[0] > triple[1]) std.mem.swap(i64, &triple[0], &triple[1]);
}

test "triplet sum: python reference examples" {
    try testing.expectEqualSlices(i64, &[_]i64{ 5, 7, 23 }, &tripletSum1(&[_]i64{ 13, 29, 7, 23, 5 }, 35));
    try testing.expectEqualSlices(i64, &[_]i64{ 9, 19, 37 }, &tripletSum1(&[_]i64{ 37, 9, 19, 50, 44 }, 65));
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 0, 0 }, &tripletSum1(&[_]i64{ 6, 47, 27, 1, 15 }, 11));

    try testing.expectEqualSlices(i64, &[_]i64{ 5, 7, 23 }, &(try tripletSum2(testing.allocator, &[_]i64{ 13, 29, 7, 23, 5 }, 35)));
    try testing.expectEqualSlices(i64, &[_]i64{ 9, 19, 37 }, &(try tripletSum2(testing.allocator, &[_]i64{ 37, 9, 19, 50, 44 }, 65)));
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 0, 0 }, &(try tripletSum2(testing.allocator, &[_]i64{ 6, 47, 27, 1, 15 }, 11)));
}

test "triplet sum: edge and extreme cases" {
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 0, 0 }, &(try tripletSum2(testing.allocator, &[_]i64{}, 0)));
    try testing.expectEqualSlices(i64, &[_]i64{ -3, 1, 2 }, &(try tripletSum2(testing.allocator, &[_]i64{ -3, 1, 2, 9, 15 }, 0)));
    const extreme = [_]i64{ std.math.maxInt(i64), std.math.maxInt(i64), std.math.minInt(i64), 2, 5 };
    try testing.expectEqualSlices(i64, &[_]i64{ std.math.minInt(i64), 2, std.math.maxInt(i64) }, &tripletSum1(&extreme, 1));
    try testing.expectEqualSlices(i64, &[_]i64{ std.math.minInt(i64), 2, std.math.maxInt(i64) }, &(try tripletSum2(testing.allocator, &extreme, 1)));
}
