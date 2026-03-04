//! Rotate Array - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/arrays/rotate_array.py

const std = @import("std");
const testing = std.testing;

fn reverse(slice: []i64) void {
    if (slice.len == 0) return;
    var i: usize = 0;
    var j: usize = slice.len - 1;
    while (i < j) {
        const t = slice[i];
        slice[i] = slice[j];
        slice[j] = t;
        i += 1;
        j -= 1;
    }
}

/// Rotates array to the right by `steps` (negative means left rotation).
/// Time complexity: O(n), Space complexity: O(n)
pub fn rotateArray(allocator: std.mem.Allocator, arr: []const i64, steps: i64) ![]i64 {
    const n = arr.len;
    const out = try allocator.alloc(i64, n);
    if (n == 0) return out;

    @memcpy(out, arr);

    const n_i64: i64 = @intCast(n);
    var k = @mod(steps, n_i64);
    if (k < 0) k += n_i64;

    const k_usize: usize = @intCast(k);

    reverse(out);
    if (k_usize > 0) reverse(out[0..k_usize]);
    if (k_usize < n) reverse(out[k_usize..n]);

    return out;
}

test "rotate array: python examples" {
    {
        const rotated = try rotateArray(testing.allocator, &[_]i64{ 1, 2, 3, 4, 5 }, 2);
        defer testing.allocator.free(rotated);
        try testing.expectEqualSlices(i64, &[_]i64{ 4, 5, 1, 2, 3 }, rotated);
    }
    {
        const rotated = try rotateArray(testing.allocator, &[_]i64{ 1, 2, 3, 4, 5 }, -2);
        defer testing.allocator.free(rotated);
        try testing.expectEqualSlices(i64, &[_]i64{ 3, 4, 5, 1, 2 }, rotated);
    }
    {
        const rotated = try rotateArray(testing.allocator, &[_]i64{ 1, 2, 3, 4, 5 }, 7);
        defer testing.allocator.free(rotated);
        try testing.expectEqualSlices(i64, &[_]i64{ 4, 5, 1, 2, 3 }, rotated);
    }
}

test "rotate array: empty and extreme" {
    {
        const rotated = try rotateArray(testing.allocator, &[_]i64{}, 3);
        defer testing.allocator.free(rotated);
        try testing.expectEqual(@as(usize, 0), rotated.len);
    }

    const n: usize = 100_000;
    var values = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(values);
    for (0..n) |i| values[i] = @intCast(i + 1);

    const rotated = try rotateArray(testing.allocator, values, 1_234_567);
    defer testing.allocator.free(rotated);

    const k = @as(usize, @intCast(@as(i64, 1_234_567) % @as(i64, @intCast(n))));
    try testing.expectEqual(@as(i64, @intCast(n - k + 1)), rotated[0]);
    try testing.expectEqual(@as(i64, @intCast(n - k + 2)), rotated[1]);
}
