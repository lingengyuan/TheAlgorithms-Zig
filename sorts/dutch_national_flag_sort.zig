//! Dutch National Flag Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/dutch_national_flag_sort.py

const std = @import("std");
const testing = std.testing;

pub const DnfSortError = error{InvalidValue};

/// In-place DNF sort for values in {0, 1, 2}.
/// Time complexity: O(n), Space complexity: O(1)
pub fn dutchNationalFlagSort(sequence: []i8) DnfSortError!void {
    if (sequence.len <= 1) {
        if (sequence.len == 1 and (sequence[0] < 0 or sequence[0] > 2)) return error.InvalidValue;
        return;
    }

    var low: usize = 0;
    var mid: usize = 0;
    var high: usize = sequence.len - 1;

    while (mid <= high) {
        switch (sequence[mid]) {
            0 => {
                std.mem.swap(i8, &sequence[low], &sequence[mid]);
                low += 1;
                mid += 1;
            },
            1 => mid += 1,
            2 => {
                std.mem.swap(i8, &sequence[mid], &sequence[high]);
                if (high == 0) break;
                high -= 1;
            },
            else => return error.InvalidValue,
        }
    }
}

test "dutch national flag sort: python reference examples" {
    var a1 = [_]i8{};
    try dutchNationalFlagSort(&a1);
    try testing.expectEqual(@as(usize, 0), a1.len);

    var a2 = [_]i8{0};
    try dutchNationalFlagSort(&a2);
    try testing.expectEqualSlices(i8, &[_]i8{0}, &a2);

    var a3 = [_]i8{ 2, 1, 0, 0, 1, 2 };
    try dutchNationalFlagSort(&a3);
    try testing.expectEqualSlices(i8, &[_]i8{ 0, 0, 1, 1, 2, 2 }, &a3);

    var a4 = [_]i8{ 0, 1, 1, 0, 1, 2, 1, 2, 0, 0, 0, 1 };
    try dutchNationalFlagSort(&a4);
    try testing.expectEqualSlices(i8, &[_]i8{ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2 }, &a4);
}

test "dutch national flag sort: invalid value cases" {
    var a1 = [_]i8{ 3, 2, 3, 1, 3, 0, 3 };
    try testing.expectError(error.InvalidValue, dutchNationalFlagSort(&a1));

    var a2 = [_]i8{ -1, 2, -1, 1, -1, 0, -1 };
    try testing.expectError(error.InvalidValue, dutchNationalFlagSort(&a2));
}

test "dutch national flag sort: extreme large input" {
    const alloc = testing.allocator;
    const n: usize = 200_000;
    const arr = try alloc.alloc(i8, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, i| {
        v.* = @as(i8, @intCast(i % 3));
    }
    std.mem.reverse(i8, arr);

    try dutchNationalFlagSort(arr);

    // verify non-decreasing and value domain
    for (arr) |v| try testing.expect(v >= 0 and v <= 2);
    for (1..arr.len) |i| try testing.expect(arr[i - 1] <= arr[i]);
}
