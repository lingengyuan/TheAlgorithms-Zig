//! Product Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/arrays/product_sum.py

const std = @import("std");
const testing = std.testing;

pub const SpecialArrayElement = union(enum) {
    value: f64,
    list: []const SpecialArrayElement,
};

/// Computes product sum recursively with explicit depth.
/// Equivalent behavior to Python `product_sum(arr, depth)`.
/// Time complexity: O(n), Space complexity: O(d)
pub fn productSum(arr: []const SpecialArrayElement, depth: i64) f64 {
    var total_sum: f64 = 0;
    for (arr) |element| {
        switch (element) {
            .value => |v| total_sum += v,
            .list => |nested| total_sum += productSum(nested, depth + 1),
        }
    }

    return total_sum * @as(f64, @floatFromInt(depth));
}

/// Computes product sum with default depth = 1.
/// Equivalent behavior to Python `product_sum_array(array)`.
/// Time complexity: O(n), Space complexity: O(d)
pub fn productSumArray(array: []const SpecialArrayElement) f64 {
    return productSum(array, 1);
}

test "product sum: python doctest product_sum" {
    const plain = [_]SpecialArrayElement{ .{ .value = 1 }, .{ .value = 2 }, .{ .value = 3 } };
    try testing.expectApproxEqAbs(@as(f64, 6), productSum(plain[0..], 1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -6), productSum(plain[0..], -1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0), productSum(plain[0..], 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 42), productSum(plain[0..], 7), 1e-12);

    const nested_inner = [_]SpecialArrayElement{ .{ .value = -3 }, .{ .value = 4 } };
    const nested = [_]SpecialArrayElement{
        .{ .value = -1 },
        .{ .value = 2 },
        .{ .list = nested_inner[0..] },
    };
    try testing.expectApproxEqAbs(@as(f64, 8), productSum(nested[0..], 2), 1e-12);

    const floats_level2 = [_]SpecialArrayElement{.{ .value = 0.5 }};
    const floats_level1 = [_]SpecialArrayElement{ .{ .value = 1 }, .{ .list = floats_level2[0..] } };
    const floats = [_]SpecialArrayElement{ .{ .value = -3.5 }, .{ .list = floats_level1[0..] } };
    try testing.expectApproxEqAbs(@as(f64, 1.5), productSum(floats[0..], 1), 1e-12);

    const zero_mix = [_]SpecialArrayElement{ .{ .value = 1 }, .{ .value = -1 } };
    try testing.expectApproxEqAbs(@as(f64, 0), productSum(zero_mix[0..], 1), 1e-12);
}

test "product sum: python doctest product_sum_array" {
    const arr1 = [_]SpecialArrayElement{ .{ .value = 1 }, .{ .value = 2 }, .{ .value = 3 } };
    try testing.expectApproxEqAbs(@as(f64, 6), productSumArray(arr1[0..]), 1e-12);

    const arr2_child = [_]SpecialArrayElement{ .{ .value = 2 }, .{ .value = 3 } };
    const arr2 = [_]SpecialArrayElement{ .{ .value = 1 }, .{ .list = arr2_child[0..] } };
    try testing.expectApproxEqAbs(@as(f64, 11), productSumArray(arr2[0..]), 1e-12);

    const arr3_l2 = [_]SpecialArrayElement{ .{ .value = 3 }, .{ .value = 4 } };
    const arr3_l1 = [_]SpecialArrayElement{ .{ .value = 2 }, .{ .list = arr3_l2[0..] } };
    const arr3 = [_]SpecialArrayElement{ .{ .value = 1 }, .{ .list = arr3_l1[0..] } };
    try testing.expectApproxEqAbs(@as(f64, 47), productSumArray(arr3[0..]), 1e-12);

    const arr4 = [_]SpecialArrayElement{.{ .value = 0 }};
    try testing.expectApproxEqAbs(@as(f64, 0), productSumArray(arr4[0..]), 1e-12);

    const float_l2 = [_]SpecialArrayElement{.{ .value = 0.5 }};
    const float_l1 = [_]SpecialArrayElement{ .{ .value = 1 }, .{ .list = float_l2[0..] } };
    const float_arr = [_]SpecialArrayElement{ .{ .value = -3.5 }, .{ .list = float_l1[0..] } };
    try testing.expectApproxEqAbs(@as(f64, 1.5), productSumArray(float_arr[0..]), 1e-12);

    const arr5 = [_]SpecialArrayElement{ .{ .value = 1 }, .{ .value = -2 } };
    try testing.expectApproxEqAbs(@as(f64, -1), productSumArray(arr5[0..]), 1e-12);
}

test "product sum: boundary and extreme" {
    const empty = [_]SpecialArrayElement{};
    try testing.expectApproxEqAbs(@as(f64, 0), productSumArray(empty[0..]), 1e-12);

    const deep_l5 = [_]SpecialArrayElement{.{ .value = 0 }};
    const deep_l4 = [_]SpecialArrayElement{.{ .list = deep_l5[0..] }};
    const deep_l3 = [_]SpecialArrayElement{.{ .list = deep_l4[0..] }};
    const deep_l2 = [_]SpecialArrayElement{.{ .list = deep_l3[0..] }};
    const deep_l1 = [_]SpecialArrayElement{.{ .list = deep_l2[0..] }};
    const deep_root = [_]SpecialArrayElement{.{ .list = deep_l1[0..] }};
    try testing.expectApproxEqAbs(@as(f64, 0), productSumArray(deep_root[0..]), 1e-12);

    const n: usize = 100_000;
    const large = try testing.allocator.alloc(SpecialArrayElement, n);
    defer testing.allocator.free(large);

    for (0..n) |i| {
        large[i] = .{ .value = if (i % 2 == 0) 1 else -1 };
    }

    try testing.expectApproxEqAbs(@as(f64, 0), productSumArray(large), 1e-12);
}
