//! Hamming Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/special_numbers/hamming_numbers.py

const std = @import("std");
const testing = std.testing;

pub const HammingError = error{ InvalidInput, Overflow };

/// Returns first `n_element` Hamming numbers.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn hamming(
    allocator: std.mem.Allocator,
    n_element: i64,
) (HammingError || std.mem.Allocator.Error)![]u128 {
    if (n_element < 1) return HammingError.InvalidInput;
    if (@sizeOf(usize) < @sizeOf(i64) and n_element > @as(i64, @intCast(std.math.maxInt(usize)))) {
        return HammingError.Overflow;
    }
    const target: usize = @intCast(n_element);

    var values = std.ArrayListUnmanaged(u128){};
    errdefer values.deinit(allocator);
    try values.append(allocator, 1);

    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;

    while (values.items.len < target) {
        const last = values.items[values.items.len - 1];

        while (true) {
            const mul2 = @mulWithOverflow(values.items[i], @as(u128, 2));
            if (mul2[1] != 0) return HammingError.Overflow;
            if (mul2[0] > last) break;
            i += 1;
        }
        while (true) {
            const mul3 = @mulWithOverflow(values.items[j], @as(u128, 3));
            if (mul3[1] != 0) return HammingError.Overflow;
            if (mul3[0] > last) break;
            j += 1;
        }
        while (true) {
            const mul5 = @mulWithOverflow(values.items[k], @as(u128, 5));
            if (mul5[1] != 0) return HammingError.Overflow;
            if (mul5[0] > last) break;
            k += 1;
        }

        const next2 = @mulWithOverflow(values.items[i], @as(u128, 2));
        const next3 = @mulWithOverflow(values.items[j], @as(u128, 3));
        const next5 = @mulWithOverflow(values.items[k], @as(u128, 5));
        if (next2[1] != 0 or next3[1] != 0 or next5[1] != 0) return HammingError.Overflow;

        const candidate = @min(next2[0], @min(next3[0], next5[0]));
        try values.append(allocator, candidate);
    }

    return try values.toOwnedSlice(allocator);
}

test "hamming numbers: python reference examples" {
    try testing.expectError(HammingError.InvalidInput, hamming(testing.allocator, -5));

    const n5 = try hamming(testing.allocator, 5);
    defer testing.allocator.free(n5);
    try testing.expectEqualSlices(u128, &[_]u128{ 1, 2, 3, 4, 5 }, n5);

    const n10 = try hamming(testing.allocator, 10);
    defer testing.allocator.free(n10);
    try testing.expectEqualSlices(u128, &[_]u128{ 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 }, n10);

    const n15 = try hamming(testing.allocator, 15);
    defer testing.allocator.free(n15);
    try testing.expectEqualSlices(u128, &[_]u128{ 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24 }, n15);
}

test "hamming numbers: edge and extreme cases" {
    try testing.expectError(HammingError.InvalidInput, hamming(testing.allocator, 0));

    const n1 = try hamming(testing.allocator, 1);
    defer testing.allocator.free(n1);
    try testing.expectEqualSlices(u128, &[_]u128{1}, n1);

    const n100 = try hamming(testing.allocator, 100);
    defer testing.allocator.free(n100);
    try testing.expectEqual(@as(usize, 100), n100.len);
    try testing.expectEqual(@as(u128, 1_536), n100[n100.len - 1]);

    const n1000 = try hamming(testing.allocator, 1_000);
    defer testing.allocator.free(n1000);
    try testing.expectEqual(@as(usize, 1_000), n1000.len);
    try testing.expectEqual(@as(u128, 51_200_000), n1000[n1000.len - 1]);
}
