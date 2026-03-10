//! Print Multiplication Table - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/print_multiplication_table.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Returns the multiplication table for `number` with `number_of_terms` rows.
/// The output matches the Python reference formatting and has no trailing newline.
/// Time complexity: O(number_of_terms), Space complexity: O(number_of_terms)
pub fn multiplicationTable(
    allocator: Allocator,
    number: i64,
    number_of_terms: usize,
) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(allocator);

    for (1..number_of_terms + 1) |i| {
        if (i > 1) try out.append(allocator, '\n');
        try out.writer(allocator).print("{d} * {d} = {d}", .{ number, i, number * @as(i64, @intCast(i)) });
    }
    return try out.toOwnedSlice(allocator);
}

test "print multiplication table: python reference examples" {
    const alloc = testing.allocator;

    const positive = try multiplicationTable(alloc, 3, 5);
    defer alloc.free(positive);
    try testing.expectEqualStrings(
        \\3 * 1 = 3
        \\3 * 2 = 6
        \\3 * 3 = 9
        \\3 * 4 = 12
        \\3 * 5 = 15
    , positive);

    const negative = try multiplicationTable(alloc, -4, 6);
    defer alloc.free(negative);
    try testing.expectEqualStrings(
        \\-4 * 1 = -4
        \\-4 * 2 = -8
        \\-4 * 3 = -12
        \\-4 * 4 = -16
        \\-4 * 5 = -20
        \\-4 * 6 = -24
    , negative);
}

test "print multiplication table: zero terms and extreme multiplier" {
    const alloc = testing.allocator;

    const empty = try multiplicationTable(alloc, 5, 0);
    defer alloc.free(empty);
    try testing.expectEqualStrings("", empty);

    const extreme = try multiplicationTable(alloc, -999, 2);
    defer alloc.free(extreme);
    try testing.expectEqualStrings(
        \\-999 * 1 = -999
        \\-999 * 2 = -1998
    , extreme);
}
