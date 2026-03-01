//! Maximum Product Subarray - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/max_product_subarray.py

const std = @import("std");
const testing = std.testing;

pub const MaxProductError = error{Overflow};

fn mulChecked(a: i64, b: i64) MaxProductError!i64 {
    const prod = @mulWithOverflow(a, b);
    if (prod[1] != 0) return MaxProductError.Overflow;
    return prod[0];
}

/// Returns the maximum product over all non-empty contiguous subarrays.
/// Returns 0 for empty input.
/// Time complexity: O(n), space complexity: O(1)
pub fn maxProductSubarray(numbers: []const i64) MaxProductError!i64 {
    if (numbers.len == 0) return 0;

    var max_till_now = numbers[0];
    var min_till_now = numbers[0];
    var max_prod = numbers[0];

    for (numbers[1..]) |number| {
        if (number < 0) {
            std.mem.swap(i64, &max_till_now, &min_till_now);
        }

        const max_mul = try mulChecked(max_till_now, number);
        const min_mul = try mulChecked(min_till_now, number);

        max_till_now = @max(number, max_mul);
        min_till_now = @min(number, min_mul);
        max_prod = @max(max_prod, max_till_now);
    }

    return max_prod;
}

test "max product subarray: basic cases" {
    try testing.expectEqual(@as(i64, 6), try maxProductSubarray(&[_]i64{ 2, 3, -2, 4 }));
    try testing.expectEqual(@as(i64, 0), try maxProductSubarray(&[_]i64{ -2, 0, -1 }));
    try testing.expectEqual(@as(i64, 48), try maxProductSubarray(&[_]i64{ 2, 3, -2, 4, -1 }));
}

test "max product subarray: edge values" {
    try testing.expectEqual(@as(i64, -1), try maxProductSubarray(&[_]i64{-1}));
    try testing.expectEqual(@as(i64, 0), try maxProductSubarray(&[_]i64{0}));
    try testing.expectEqual(@as(i64, 0), try maxProductSubarray(&[_]i64{}));
}

test "max product subarray: all negatives" {
    try testing.expectEqual(@as(i64, 12), try maxProductSubarray(&[_]i64{ -2, -3, -4 }));
    try testing.expectEqual(@as(i64, 60), try maxProductSubarray(&[_]i64{ -1, -3, -10, 0, 60 }));
}

test "max product subarray: many zeros" {
    var arr: [256]i64 = undefined;
    @memset(&arr, 0);
    arr[10] = -2;
    arr[11] = -3;
    try testing.expectEqual(@as(i64, 6), try maxProductSubarray(&arr));
}

test "max product subarray: overflow is reported" {
    try testing.expectError(MaxProductError.Overflow, maxProductSubarray(&[_]i64{ std.math.maxInt(i64), 2 }));
}
