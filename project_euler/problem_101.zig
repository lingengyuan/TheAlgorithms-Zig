//! Project Euler Problem 101: Optimum Polynomial - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_101/sol1.py

const std = @import("std");
const testing = std.testing;

fn questionFunction(n: i128) i128 {
    var total: i128 = 0;
    var power: i128 = 1;
    var sign: i128 = 1;

    var degree: usize = 0;
    while (degree <= 10) : (degree += 1) {
        total += sign * power;
        power *= n;
        sign = -sign;
    }
    return total;
}

fn cubicFunction(n: i128) i128 {
    return n * n * n;
}

fn firstIncorrectTerm(values: []const i128) i128 {
    var workspace: [16]i128 = undefined;
    @memcpy(workspace[0..values.len], values);

    const count: i128 = @intCast(values.len);
    var combination: i128 = 1;
    var total: i128 = 0;

    var order: usize = 0;
    while (order < values.len) : (order += 1) {
        total += combination * workspace[0];

        var index: usize = 0;
        while (index + 1 < values.len - order) : (index += 1) {
            workspace[index] = workspace[index + 1] - workspace[index];
        }

        if (order + 1 < values.len) {
            const next_order: i128 = @intCast(order + 1);
            combination = @divExact(combination * (count - @as(i128, @intCast(order))), next_order);
        }
    }

    return total;
}

fn sumFits(comptime func: fn (i128) i128, order: usize) i128 {
    var values: [16]i128 = undefined;
    var index: usize = 0;
    while (index < order) : (index += 1) {
        values[index] = func(@intCast(index + 1));
    }

    var total: i128 = 0;
    var count: usize = 1;
    while (count <= order) : (count += 1) {
        total += firstIncorrectTerm(values[0..count]);
    }
    return total;
}

/// Returns the sum of the FIT values for the optimum polynomials derived from
/// the first `order` terms of the question sequence.
/// Time complexity: O(order^3)
/// Space complexity: O(order)
pub fn solution(order: usize) i128 {
    return sumFits(questionFunction, order);
}

test "problem 101: python reference" {
    try testing.expectEqual(@as(i128, 74), sumFits(cubicFunction, 3));
    try testing.expectEqual(@as(i128, 37076114526), solution(10));
}

test "problem 101: finite-difference extrapolation edge cases" {
    try testing.expectEqual(@as(i128, 1), firstIncorrectTerm(&[_]i128{1}));
    try testing.expectEqual(@as(i128, 15), firstIncorrectTerm(&[_]i128{ 1, 8 }));
    try testing.expectEqual(@as(i128, 58), firstIncorrectTerm(&[_]i128{ 1, 8, 27 }));
    try testing.expectEqual(@as(i128, 0), solution(0));
}
