//! Kth Lexicographic Permutation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/kth_lexicographic_permutation.py

const std = @import("std");
const testing = std.testing;

pub const PermutationError = error{ InvalidInput, OutOfBounds, Overflow };

/// Returns k-th lexicographic permutation of [0, 1, ..., n-1].
/// Caller owns returned slice.
/// Time complexity: O(n^2), Space complexity: O(n)
pub fn kthPermutation(
    allocator: std.mem.Allocator,
    k: i64,
    n: i64,
) (PermutationError || std.mem.Allocator.Error)![]u64 {
    if (n < 1 or k < 0) return PermutationError.InvalidInput;
    if (@sizeOf(usize) < @sizeOf(i64) and n > @as(i64, @intCast(std.math.maxInt(usize)))) {
        return PermutationError.Overflow;
    }

    const n_usize: usize = @intCast(n);
    if (n_usize == 1) {
        if (k != 0) return PermutationError.OutOfBounds;
        const out = try allocator.alloc(u64, 1);
        out[0] = 0;
        return out;
    }

    var factorials = std.ArrayListUnmanaged(u128){};
    defer factorials.deinit(allocator);
    try factorials.append(allocator, 1);

    var i: usize = 2;
    while (i < n_usize) : (i += 1) {
        const prev = factorials.items[factorials.items.len - 1];
        const mul = @mulWithOverflow(prev, @as(u128, @intCast(i)));
        if (mul[1] != 0) return PermutationError.Overflow;
        try factorials.append(allocator, mul[0]);
    }

    const total_mul = @mulWithOverflow(factorials.items[factorials.items.len - 1], @as(u128, @intCast(n_usize)));
    if (total_mul[1] != 0) return PermutationError.Overflow;
    if (@as(u128, @intCast(k)) >= total_mul[0]) return PermutationError.OutOfBounds;

    var elements = std.ArrayListUnmanaged(u64){};
    defer elements.deinit(allocator);
    try elements.ensureTotalCapacity(allocator, n_usize);
    for (0..n_usize) |idx| try elements.append(allocator, @intCast(idx));

    var permutation = std.ArrayListUnmanaged(u64){};
    errdefer permutation.deinit(allocator);
    try permutation.ensureTotalCapacity(allocator, n_usize);

    var k_rem: u128 = @intCast(k);
    while (factorials.items.len > 0) {
        const factorial = factorials.pop().?;
        const number = k_rem / factorial;
        k_rem %= factorial;

        if (number >= elements.items.len) return PermutationError.OutOfBounds;
        const idx: usize = @intCast(number);
        try permutation.append(allocator, elements.items[idx]);
        _ = elements.orderedRemove(idx);
    }
    if (elements.items.len != 1) return PermutationError.OutOfBounds;
    try permutation.append(allocator, elements.items[0]);

    return try permutation.toOwnedSlice(allocator);
}

test "kth permutation: python reference examples" {
    const r1 = try kthPermutation(testing.allocator, 0, 5);
    defer testing.allocator.free(r1);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 1, 2, 3, 4 }, r1);

    const r2 = try kthPermutation(testing.allocator, 10, 4);
    defer testing.allocator.free(r2);
    try testing.expectEqualSlices(u64, &[_]u64{ 1, 3, 0, 2 }, r2);
}

test "kth permutation: edge and extreme cases" {
    try testing.expectError(PermutationError.OutOfBounds, kthPermutation(testing.allocator, 24, 4));
    try testing.expectError(PermutationError.InvalidInput, kthPermutation(testing.allocator, -1, 4));
    try testing.expectError(PermutationError.InvalidInput, kthPermutation(testing.allocator, 0, 0));

    const r = try kthPermutation(testing.allocator, 0, 1);
    defer testing.allocator.free(r);
    try testing.expectEqualSlices(u64, &[_]u64{0}, r);
}
