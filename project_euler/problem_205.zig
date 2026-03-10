//! Project Euler Problem 205: Dice Game - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_205/sol1.py

const std = @import("std");
const testing = std.testing;

pub fn totalFrequencyDistribution(allocator: std.mem.Allocator, sides_number: u32, dice_number: u32) ![]u64 {
    const max_total = sides_number * dice_number;
    var current = try allocator.alloc(u64, max_total + 1);
    errdefer allocator.free(current);
    @memset(current, 0);
    current[0] = 1;

    var die: u32 = 0;
    while (die < dice_number) : (die += 1) {
        var next = try allocator.alloc(u64, max_total + 1);
        @memset(next, 0);
        for (0..current.len) |total| {
            const ways = current[total];
            if (ways == 0) continue;
            for (1..sides_number + 1) |face| next[total + face] += ways;
        }
        allocator.free(current);
        current = next;
    }
    return current;
}

/// Returns the probability that Peter beats Colin, rounded to seven decimal places.
/// Time complexity: O(dice_number * sides_number * max_total)
/// Space complexity: O(max_total)
pub fn solution(allocator: std.mem.Allocator) !f64 {
    const peter = try totalFrequencyDistribution(allocator, 4, 9);
    defer allocator.free(peter);
    const colin = try totalFrequencyDistribution(allocator, 6, 6);
    defer allocator.free(colin);

    var peter_wins: u64 = 0;
    var prefix: u64 = 0;
    for (0..colin.len) |total| {
        if (total < peter.len) peter_wins += peter[total] * prefix;
        prefix += colin[total];
    }

    const total_games = std.math.pow(u64, 4, 9) * std.math.pow(u64, 6, 6);
    const probability = @as(f64, @floatFromInt(peter_wins)) / @as(f64, @floatFromInt(total_games));
    return std.math.round(probability * 10_000_000.0) / 10_000_000.0;
}

test "problem 205: frequency distributions" {
    const alloc = testing.allocator;
    const d6 = try totalFrequencyDistribution(alloc, 6, 1);
    defer alloc.free(d6);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 1, 1, 1, 1, 1, 1 }, d6);

    const d4 = try totalFrequencyDistribution(alloc, 4, 2);
    defer alloc.free(d4);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 0, 1, 2, 3, 4, 3, 2, 1 }, d4);
}

test "problem 205: python reference" {
    try testing.expectApproxEqAbs(@as(f64, 0.5731441), try solution(testing.allocator), 0.0000001);
}
