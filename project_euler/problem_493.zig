//! Project Euler Problem 493: Under The Rainbow - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_493/sol1.py

const std = @import("std");
const testing = std.testing;

const Allocator = std.mem.Allocator;
const balls_per_colour: usize = 10;
const num_colours: usize = 7;
const num_balls: usize = balls_per_colour * num_colours;

fn expectedDistinctColours(num_picks: usize) f64 {
    if (num_picks == 0) return 0.0;
    if (num_picks > num_balls) return 0.0;
    if (num_picks > num_balls - balls_per_colour) return @floatFromInt(num_colours);

    var missing_ratio: f64 = 1.0;
    var i: usize = 0;
    while (i < num_picks) : (i += 1) {
        missing_ratio *= @as(f64, @floatFromInt(num_balls - balls_per_colour - i));
        missing_ratio /= @as(f64, @floatFromInt(num_balls - i));
    }

    return @as(f64, @floatFromInt(num_colours)) * (1.0 - missing_ratio);
}

/// Returns the expected number of distinct colours after `num_picks` draws,
/// formatted with nine digits after the decimal point.
/// Time complexity: O(num_picks)
/// Space complexity: O(1)
pub fn solution(allocator: Allocator, num_picks: usize) ![]u8 {
    const result = expectedDistinctColours(num_picks);
    return std.fmt.allocPrint(allocator, "{d:.9}", .{result});
}

test "problem 493: python reference" {
    const alloc = testing.allocator;

    const value_10 = try solution(alloc, 10);
    defer alloc.free(value_10);
    try testing.expectEqualStrings("5.669644129", value_10);

    const value_20 = try solution(alloc, 20);
    defer alloc.free(value_20);
    try testing.expectEqualStrings("6.818741802", value_20);

    const value_30 = try solution(alloc, 30);
    defer alloc.free(value_30);
    try testing.expectEqualStrings("6.985042712", value_30);
}

test "problem 493: edge draw counts" {
    const alloc = testing.allocator;

    const none = try solution(alloc, 0);
    defer alloc.free(none);
    try testing.expectEqualStrings("0.000000000", none);

    const all = try solution(alloc, 70);
    defer alloc.free(all);
    try testing.expectEqualStrings("7.000000000", all);

    try testing.expectApproxEqAbs(@as(f64, 1.0), expectedDistinctColours(1), 1e-12);
}
