//! Next Greater Element - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/stacks/next_greater_element.py

const std = @import("std");
const testing = std.testing;

/// Brute-force next greater element.
/// Time complexity: O(n^2), Space complexity: O(n)
pub fn nextGreatestElementSlow(allocator: std.mem.Allocator, arr: []const f64) ![]f64 {
    const result = try allocator.alloc(f64, arr.len);

    for (arr, 0..) |outer, i| {
        var next_item: f64 = -1;
        var j = i + 1;
        while (j < arr.len) : (j += 1) {
            if (outer < arr[j]) {
                next_item = arr[j];
                break;
            }
        }
        result[i] = next_item;
    }

    return result;
}

/// Stack-based next greater element.
/// Time complexity: O(n), Space complexity: O(n)
pub fn nextGreatestElement(allocator: std.mem.Allocator, arr: []const f64) ![]f64 {
    const n = arr.len;
    const result = try allocator.alloc(f64, n);
    @memset(result, -1);

    var stack = std.ArrayListUnmanaged(f64){};
    defer stack.deinit(allocator);

    var idx = n;
    while (idx > 0) {
        idx -= 1;
        while (stack.items.len > 0 and stack.items[stack.items.len - 1] <= arr[idx]) {
            _ = stack.pop();
        }
        if (stack.items.len > 0) {
            result[idx] = stack.items[stack.items.len - 1];
        }
        try stack.append(allocator, arr[idx]);
    }

    return result;
}

fn expectFloatSlicesApprox(expected: []const f64, actual: []const f64, tol: f64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try testing.expectApproxEqAbs(e, a, tol);
    }
}

test "next greater element: python sample" {
    const arr = [_]f64{ -10, -5, 0, 5, 5.1, 11, 13, 21, 3, 4, -21, -10, -5, -1, 0 };
    const expected = [_]f64{ -5, 0, 5, 5.1, 11, 13, 21, -1, 4, -1, -10, -5, -1, 0, -1 };

    const slow = try nextGreatestElementSlow(testing.allocator, &arr);
    defer testing.allocator.free(slow);
    const fast = try nextGreatestElement(testing.allocator, &arr);
    defer testing.allocator.free(fast);

    try expectFloatSlicesApprox(&expected, slow, 1e-9);
    try expectFloatSlicesApprox(&expected, fast, 1e-9);
}

test "next greater element: empty and one item" {
    const empty = try nextGreatestElement(testing.allocator, &[_]f64{});
    defer testing.allocator.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const one = try nextGreatestElement(testing.allocator, &[_]f64{42});
    defer testing.allocator.free(one);
    try expectFloatSlicesApprox(&[_]f64{-1}, one, 1e-9);
}

test "next greater element: extreme monotonic arrays" {
    const n: usize = 50_000;

    var ascending = try testing.allocator.alloc(f64, n);
    defer testing.allocator.free(ascending);
    for (0..n) |i| ascending[i] = @floatFromInt(i);

    const nge_asc = try nextGreatestElement(testing.allocator, ascending);
    defer testing.allocator.free(nge_asc);

    for (0..n - 1) |i| {
        try testing.expectApproxEqAbs(@as(f64, @floatFromInt(i + 1)), nge_asc[i], 1e-9);
    }
    try testing.expectApproxEqAbs(@as(f64, -1), nge_asc[n - 1], 1e-9);

    var descending = try testing.allocator.alloc(f64, n);
    defer testing.allocator.free(descending);
    for (0..n) |i| descending[i] = @floatFromInt(n - i);

    const nge_desc = try nextGreatestElement(testing.allocator, descending);
    defer testing.allocator.free(nge_desc);

    for (nge_desc) |v| {
        try testing.expectApproxEqAbs(@as(f64, -1), v, 1e-9);
    }
}
