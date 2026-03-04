//! Two Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/two_sum.py

const std = @import("std");
const testing = std.testing;

/// Returns indices of two numbers summing to target, or empty slice if no solution.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn twoSum(
    allocator: std.mem.Allocator,
    nums: []const i64,
    target: i64,
) std.mem.Allocator.Error![]usize {
    var map = std.AutoHashMap(i64, usize).init(allocator);
    defer map.deinit();

    for (nums, 0..) |value, index| {
        const compl_i128 = @as(i128, target) - @as(i128, value);
        if (compl_i128 >= std.math.minInt(i64) and compl_i128 <= std.math.maxInt(i64)) {
            const complement: i64 = @intCast(compl_i128);
            if (map.get(complement)) |match_idx| {
                const out = try allocator.alloc(usize, 2);
                out[0] = match_idx;
                out[1] = index;
                return out;
            }
        }
        try map.put(value, index);
    }

    return try allocator.alloc(usize, 0);
}

test "two sum: python reference examples" {
    const alloc = testing.allocator;

    const r1 = try twoSum(alloc, &[_]i64{ 2, 7, 11, 15 }, 9);
    defer alloc.free(r1);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1 }, r1);

    const r2 = try twoSum(alloc, &[_]i64{ 15, 2, 11, 7 }, 13);
    defer alloc.free(r2);
    try testing.expectEqualSlices(usize, &[_]usize{ 1, 2 }, r2);

    const r3 = try twoSum(alloc, &[_]i64{ 2, 7, 11, 15 }, 17);
    defer alloc.free(r3);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 3 }, r3);

    const r4 = try twoSum(alloc, &[_]i64{ 7, 15, 11, 2 }, 18);
    defer alloc.free(r4);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 2 }, r4);

    const r5 = try twoSum(alloc, &[_]i64{ 2, 7, 11, 15 }, 26);
    defer alloc.free(r5);
    try testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, r5);
}

test "two sum: no-solution and extreme cases" {
    const r1 = try twoSum(testing.allocator, &[_]i64{ 2, 7, 11, 15 }, 8);
    defer testing.allocator.free(r1);
    try testing.expectEqual(@as(usize, 0), r1.len);

    var nums: [10]i64 = undefined;
    for (&nums, 0..) |*slot, i| slot.* = @as(i64, @intCast(i)) * 3;
    const r2 = try twoSum(testing.allocator, &nums, 19);
    defer testing.allocator.free(r2);
    try testing.expectEqual(@as(usize, 0), r2.len);

    const r3 = try twoSum(testing.allocator, &[_]i64{ std.math.maxInt(i64), -1, 0 }, std.math.maxInt(i64) - 1);
    defer testing.allocator.free(r3);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1 }, r3);
}
