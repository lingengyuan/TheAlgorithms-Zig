//! Project Euler Problem 112: Bouncy Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_112/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem112Error = error{InvalidPercent};

/// Returns true when the decimal string representation of `n` is neither nondecreasing nor nonincreasing.
/// Time complexity: O(digits)
/// Space complexity: O(1)
pub fn checkBouncy(n: i64) bool {
    var buffer: [32]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "{d}", .{n}) catch unreachable;

    var nondecreasing = true;
    var nonincreasing = true;
    for (1..text.len) |idx| {
        if (text[idx] < text[idx - 1]) nondecreasing = false;
        if (text[idx] > text[idx - 1]) nonincreasing = false;
    }
    return !(nondecreasing or nonincreasing);
}

/// Returns the least integer for which the proportion of bouncy numbers reaches `percent`.
/// Time complexity: O(answer · digits)
/// Space complexity: O(1)
pub fn solution(percent: f64) Problem112Error!u64 {
    if (!(percent > 0.0 and percent < 100.0)) return error.InvalidPercent;

    var bouncy_count: u64 = 0;
    var number: u64 = 1;
    while (true) : (number += 1) {
        if (checkBouncy(@intCast(number))) bouncy_count += 1;
        const ratio = @as(f64, @floatFromInt(bouncy_count)) * 100.0 / @as(f64, @floatFromInt(number));
        if (ratio >= percent) return number;
    }
}

test "problem 112: bouncy predicate" {
    try testing.expect(!checkBouncy(6789));
    try testing.expect(!checkBouncy(-12345));
    try testing.expect(!checkBouncy(0));
    try testing.expect(checkBouncy(132475));
    try testing.expect(!checkBouncy(34));
    try testing.expect(checkBouncy(341));
    try testing.expect(checkBouncy(-6548));
}

test "problem 112: python reference" {
    try testing.expectEqual(@as(u64, 538), try solution(50));
    try testing.expectEqual(@as(u64, 4770), try solution(80));
    try testing.expectEqual(@as(u64, 21780), try solution(90));
    try testing.expectEqual(@as(u64, 1587000), try solution(99));
}

test "problem 112: invalid percent" {
    try testing.expectError(error.InvalidPercent, solution(0));
    try testing.expectError(error.InvalidPercent, solution(100.011));
}
