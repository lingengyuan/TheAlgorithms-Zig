//! Cyclic Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/cyclic_sort.py

const std = @import("std");
const testing = std.testing;

pub const CyclicSortError = error{
    OutOfRangeValue,
    DuplicateValue,
};

/// In-place cyclic sort for integer domain [1..n].
/// Time complexity: O(n), Space complexity: O(1)
pub fn cyclicSort(nums: []usize) CyclicSortError!void {
    var index: usize = 0;
    while (index < nums.len) {
        const value = nums[index];
        if (value == 0 or value > nums.len) return error.OutOfRangeValue;

        const correct_index = value - 1;
        if (index != correct_index) {
            if (nums[correct_index] == value) return error.DuplicateValue;
            std.mem.swap(usize, &nums[index], &nums[correct_index]);
        } else {
            index += 1;
        }
    }
}

test "cyclic sort: python reference examples" {
    var a1 = [_]usize{};
    try cyclicSort(&a1);
    try testing.expectEqual(@as(usize, 0), a1.len);

    var a2 = [_]usize{ 3, 5, 2, 1, 4 };
    try cyclicSort(&a2);
    try testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 3, 4, 5 }, &a2);
}

test "cyclic sort: invalid domain detection" {
    var out_of_range = [_]usize{ 1, 4, 2 }; // 4 > n
    try testing.expectError(error.OutOfRangeValue, cyclicSort(&out_of_range));

    var duplicate = [_]usize{ 2, 2, 1 };
    try testing.expectError(error.DuplicateValue, cyclicSort(&duplicate));
}

test "cyclic sort: extreme reversed input" {
    const alloc = testing.allocator;
    const n: usize = 20_000;
    const arr = try alloc.alloc(usize, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, i| {
        v.* = n - i;
    }

    try cyclicSort(arr);
    for (arr, 0..) |v, i| {
        try testing.expectEqual(i + 1, v);
    }
}
