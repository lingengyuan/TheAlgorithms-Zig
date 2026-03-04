//! Iterating Through Submasks - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/iterating_through_submasks.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const SubmaskError = error{
    InvalidInput,
};

/// Returns all non-zero submasks of `mask` in descending iteration order:
/// `mask, (mask-1)&mask, ...`.
/// Time complexity: O(number_of_submasks), Space complexity: O(number_of_submasks)
pub fn listOfSubmasks(
    allocator: Allocator,
    mask: i64,
) (SubmaskError || Allocator.Error)![]u64 {
    if (mask <= 0) return SubmaskError.InvalidInput;

    const mask_u: u64 = @intCast(mask);
    var list = std.ArrayListUnmanaged(u64){};
    errdefer list.deinit(allocator);

    var submask = mask_u;
    while (submask != 0) {
        try list.append(allocator, submask);
        submask = (submask - 1) & mask_u;
    }

    return list.toOwnedSlice(allocator);
}

test "iterating through submasks: python examples" {
    const out1 = try listOfSubmasks(testing.allocator, 15);
    defer testing.allocator.free(out1);
    try testing.expectEqualSlices(u64, &[_]u64{ 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 }, out1);

    const out2 = try listOfSubmasks(testing.allocator, 13);
    defer testing.allocator.free(out2);
    try testing.expectEqualSlices(u64, &[_]u64{ 13, 12, 9, 8, 5, 4, 1 }, out2);
}

test "iterating through submasks: boundary and invalid input" {
    const one = try listOfSubmasks(testing.allocator, 1);
    defer testing.allocator.free(one);
    try testing.expectEqualSlices(u64, &[_]u64{1}, one);

    try testing.expectError(SubmaskError.InvalidInput, listOfSubmasks(testing.allocator, -7));
    try testing.expectError(SubmaskError.InvalidInput, listOfSubmasks(testing.allocator, 0));
}

test "iterating through submasks: extreme count for dense mask" {
    const out = try listOfSubmasks(testing.allocator, 65535); // 2^16 - 1
    defer testing.allocator.free(out);
    try testing.expectEqual(@as(usize, 65535), out.len);
    try testing.expectEqual(@as(u64, 65535), out[0]);
    try testing.expectEqual(@as(u64, 1), out[out.len - 1]);
}
