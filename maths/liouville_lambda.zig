//! Liouville Lambda Function - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/liouville_lambda.py

const std = @import("std");
const testing = std.testing;
const prime_factors_mod = @import("prime_factors.zig");

pub const LiouvilleError = error{InvalidInput};

/// Returns the Liouville lambda value for a positive integer.
/// Time complexity: O(sqrt(n)), Space complexity: O(log n)
pub fn liouvilleLambda(allocator: std.mem.Allocator, number: i64) (LiouvilleError || std.mem.Allocator.Error)!i8 {
    if (number < 1) return error.InvalidInput;
    const factors = try prime_factors_mod.primeFactors(allocator, number);
    defer allocator.free(factors);
    return if (factors.len % 2 == 0) 1 else -1;
}

test "liouville lambda: python reference examples" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(i8, 1), try liouvilleLambda(alloc, 10));
    try testing.expectEqual(@as(i8, -1), try liouvilleLambda(alloc, 11));
}

test "liouville lambda: edge cases" {
    const alloc = testing.allocator;
    try testing.expectError(error.InvalidInput, liouvilleLambda(alloc, 0));
    try testing.expectError(error.InvalidInput, liouvilleLambda(alloc, -1));
    try testing.expectEqual(@as(i8, 1), try liouvilleLambda(alloc, 1));
}
