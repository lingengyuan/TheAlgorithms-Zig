//! Three Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/three_sum.py

const std = @import("std");
const testing = std.testing;

pub const Triplet = [3]i64;

/// Returns unique triplets that sum to zero.
/// Caller owns returned slice.
/// Time complexity: O(n^2), Space complexity: O(n)
pub fn threeSum(allocator: std.mem.Allocator, nums: []const i64) std.mem.Allocator.Error![]Triplet {
    if (nums.len < 3) return try allocator.alloc(Triplet, 0);

    const sorted = try allocator.alloc(i64, nums.len);
    defer allocator.free(sorted);
    @memcpy(sorted, nums);
    std.mem.sort(i64, sorted, {}, std.sort.asc(i64));

    var results = std.ArrayListUnmanaged(Triplet){};
    errdefer results.deinit(allocator);

    var i: usize = 0;
    while (i + 2 < sorted.len) : (i += 1) {
        if (i > 0 and sorted[i] == sorted[i - 1]) continue;

        var low = i + 1;
        var high = sorted.len - 1;
        const target = -sorted[i];

        while (low < high) {
            const sum = sorted[low] + sorted[high];
            if (sum == target) {
                try results.append(allocator, .{ sorted[i], sorted[low], sorted[high] });

                while (low < high and sorted[low] == sorted[low + 1]) low += 1;
                while (low < high and sorted[high] == sorted[high - 1]) high -= 1;
                low += 1;
                high -= 1;
            } else if (sum < target) {
                low += 1;
            } else {
                high -= 1;
            }
        }
    }

    return try results.toOwnedSlice(allocator);
}

test "three sum: python reference examples" {
    const r1 = try threeSum(testing.allocator, &[_]i64{ -1, 0, 1, 2, -1, -4 });
    defer testing.allocator.free(r1);
    try testing.expectEqual(@as(usize, 2), r1.len);
    try testing.expectEqualSlices(i64, &[_]i64{ -1, -1, 2 }, &r1[0]);
    try testing.expectEqualSlices(i64, &[_]i64{ -1, 0, 1 }, &r1[1]);

    const r2 = try threeSum(testing.allocator, &[_]i64{ 1, 2, 3, 4 });
    defer testing.allocator.free(r2);
    try testing.expectEqual(@as(usize, 0), r2.len);
}

test "three sum: edge and extreme cases" {
    const r1 = try threeSum(testing.allocator, &[_]i64{});
    defer testing.allocator.free(r1);
    try testing.expectEqual(@as(usize, 0), r1.len);

    const r2 = try threeSum(testing.allocator, &[_]i64{ 0, 0, 0, 0, 0 });
    defer testing.allocator.free(r2);
    try testing.expectEqual(@as(usize, 1), r2.len);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 0, 0 }, &r2[0]);
}
