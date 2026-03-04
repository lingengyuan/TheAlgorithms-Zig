//! Two Pointer Two-Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/two_pointer.py

const std = @import("std");
const testing = std.testing;

/// Returns indices of two numbers summing to target in sorted array, or empty slice.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(1)
pub fn twoPointer(
    allocator: std.mem.Allocator,
    nums: []const i64,
    target: i64,
) std.mem.Allocator.Error![]usize {
    if (nums.len < 2) return try allocator.alloc(usize, 0);

    var i: usize = 0;
    var j: usize = nums.len - 1;

    while (i < j) {
        const sum_i128 = @as(i128, nums[i]) + @as(i128, nums[j]);
        if (sum_i128 == target) {
            const out = try allocator.alloc(usize, 2);
            out[0] = i;
            out[1] = j;
            return out;
        } else if (sum_i128 < target) {
            i += 1;
        } else {
            j -= 1;
        }
    }

    return try allocator.alloc(usize, 0);
}

test "two pointer: python reference examples" {
    const alloc = testing.allocator;

    const r1 = try twoPointer(alloc, &[_]i64{ 2, 7, 11, 15 }, 9);
    defer alloc.free(r1);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1 }, r1);

    const r2 = try twoPointer(alloc, &[_]i64{ 2, 7, 11, 15 }, 17);
    defer alloc.free(r2);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 3 }, r2);

    const r3 = try twoPointer(alloc, &[_]i64{ 2, 7, 11, 15 }, 18);
    defer alloc.free(r3);
    try testing.expectEqualSlices(usize, &[_]usize{ 1, 2 }, r3);

    const r4 = try twoPointer(alloc, &[_]i64{ 2, 7, 11, 15 }, 26);
    defer alloc.free(r4);
    try testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, r4);

    const r5 = try twoPointer(alloc, &[_]i64{ 1, 3, 3 }, 6);
    defer alloc.free(r5);
    try testing.expectEqualSlices(usize, &[_]usize{ 1, 2 }, r5);
}

test "two pointer: no-solution and extreme cases" {
    const r1 = try twoPointer(testing.allocator, &[_]i64{ 2, 7, 11, 15 }, 8);
    defer testing.allocator.free(r1);
    try testing.expectEqual(@as(usize, 0), r1.len);

    var nums: [10]i64 = undefined;
    for (&nums, 0..) |*slot, idx| slot.* = @as(i64, @intCast(idx)) * 3;
    const r2 = try twoPointer(testing.allocator, &nums, 19);
    defer testing.allocator.free(r2);
    try testing.expectEqual(@as(usize, 0), r2.len);

    const r3 = try twoPointer(testing.allocator, &[_]i64{ 1, 2, 3 }, 6);
    defer testing.allocator.free(r3);
    try testing.expectEqual(@as(usize, 0), r3.len);
}
