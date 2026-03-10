//! Project Euler Problem 69: Totient Maximum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_069/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem069Error = error{InvalidLimit, OutOfMemory};

/// Returns the value `n <= limit` for which `n / phi(n)` is maximal.
/// Time complexity: O(limit log log limit)
/// Space complexity: O(limit)
pub fn solution(allocator: std.mem.Allocator, limit: usize) Problem069Error!usize {
    if (limit == 0) return error.InvalidLimit;

    var phi = try allocator.alloc(usize, limit + 1);
    defer allocator.free(phi);

    for (phi, 0..) |*value, index| value.* = index;

    var number: usize = 2;
    while (number <= limit) : (number += 1) {
        if (phi[number] == number) {
            phi[number] -= 1;
            var multiple = number * 2;
            while (multiple <= limit) : (multiple += number) {
                phi[multiple] = (phi[multiple] / number) * (number - 1);
            }
        }
    }

    var answer: usize = 1;
    number = 1;
    while (number <= limit) : (number += 1) {
        if (answer * phi[number] < number * phi[answer]) answer = number;
    }
    return answer;
}

test "problem 069: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 6), try solution(alloc, 10));
    try testing.expectEqual(@as(usize, 30), try solution(alloc, 100));
    try testing.expectEqual(@as(usize, 2310), try solution(alloc, 9973));
    try testing.expectEqual(@as(usize, 510510), try solution(alloc, 1_000_000));
}

test "problem 069: invalid and minimal limits" {
    try testing.expectError(error.InvalidLimit, solution(testing.allocator, 0));
    try testing.expectEqual(@as(usize, 1), try solution(testing.allocator, 1));
    try testing.expectEqual(@as(usize, 2), try solution(testing.allocator, 2));
}
