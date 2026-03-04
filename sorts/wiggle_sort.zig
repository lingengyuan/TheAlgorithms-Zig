//! Wiggle Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/wiggle_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place wiggle sort using the same adjacent-swap rule as Python reference.
/// Produces pattern: nums[0] <= nums[1] >= nums[2] <= nums[3] ...
/// Time complexity: O(n), Space complexity: O(1)
pub fn wiggleSort(comptime T: type, nums: []T) void {
    if (nums.len == 0) return;

    for (0..nums.len) |i| {
        const prev = if (i == 0) nums.len - 1 else i - 1;
        const is_odd = (i % 2) == 1;
        const prev_gt_cur = nums[prev] > nums[i];
        if (is_odd == prev_gt_cur) {
            std.mem.swap(T, &nums[prev], &nums[i]);
        }
    }
}

fn expectWiggle(comptime T: type, nums: []const T) !void {
    if (nums.len <= 1) return;
    for (1..nums.len) |i| {
        if ((i % 2) == 1) {
            try testing.expect(nums[i - 1] <= nums[i]);
        } else {
            try testing.expect(nums[i - 1] >= nums[i]);
        }
    }
}

test "wiggle sort: python reference examples" {
    var a1 = [_]i32{ 0, 5, 3, 2, 2 };
    wiggleSort(i32, &a1);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 5, 2, 3, 2 }, &a1);

    var a2 = [_]i32{};
    wiggleSort(i32, &a2);
    try testing.expectEqual(@as(usize, 0), a2.len);

    var a3 = [_]f64{ -2.1, -5.68, -45.11 };
    wiggleSort(f64, &a3);
    try testing.expectEqualSlices(f64, &[_]f64{ -45.11, -2.1, -5.68 }, &a3);
}

test "wiggle sort: edge cases" {
    var one = [_]i32{7};
    wiggleSort(i32, &one);
    try testing.expectEqualSlices(i32, &[_]i32{7}, &one);

    var dup = [_]i32{ 2, 2, 2, 2 };
    wiggleSort(i32, &dup);
    try expectWiggle(i32, &dup);
}

test "wiggle sort: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 20_000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(42);
    var random = prng.random();
    for (arr) |*v| {
        v.* = random.intRangeAtMost(i32, -10_000, 10_000);
    }

    wiggleSort(i32, arr);
    try expectWiggle(i32, arr);
}
