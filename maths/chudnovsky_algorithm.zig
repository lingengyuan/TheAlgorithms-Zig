//! Chudnovsky Algorithm - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/chudnovsky_algorithm.py

const std = @import("std");
const testing = std.testing;
const pi_generator = @import("pi_generator.zig");

pub const ChudnovskyError = error{
    InvalidPrecision,
    OutOfMemory,
};

/// Returns a string with total length `precision`, matching the Python reference output.
/// This implementation uses the repository's exact pi digit generator to preserve output
/// semantics without introducing a second arbitrary-precision decimal engine.
/// Caller owns the returned slice.
pub fn pi(allocator: std.mem.Allocator, precision: usize) ChudnovskyError![]u8 {
    if (precision == 0) return error.InvalidPrecision;
    if (precision == 1) return try allocator.dupe(u8, "");

    const generated = try pi_generator.calculatePi(allocator, precision - 2);
    errdefer allocator.free(generated);
    const result = try allocator.dupe(u8, generated[0..precision]);
    allocator.free(generated);
    return result;
}

test "chudnovsky algorithm: python reference examples" {
    const alloc = testing.allocator;

    const p10 = try pi(alloc, 10);
    defer alloc.free(p10);
    try testing.expectEqualStrings("3.14159265", p10);

    const p20 = try pi(alloc, 20);
    defer alloc.free(p20);
    try testing.expectEqualStrings("3.141592653589793238", p20);

    const p50 = try pi(alloc, 50);
    defer alloc.free(p50);
    try testing.expectEqualStrings("3.141592653589793238462643383279502884197169399375", p50);
}

test "chudnovsky algorithm: edge precision semantics match python" {
    const alloc = testing.allocator;

    try testing.expectError(error.InvalidPrecision, pi(alloc, 0));

    const p1 = try pi(alloc, 1);
    defer alloc.free(p1);
    try testing.expectEqualStrings("", p1);

    const p2 = try pi(alloc, 2);
    defer alloc.free(p2);
    try testing.expectEqualStrings("3.", p2);

    const p3 = try pi(alloc, 3);
    defer alloc.free(p3);
    try testing.expectEqualStrings("3.1", p3);
}
