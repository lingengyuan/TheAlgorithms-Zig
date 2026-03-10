//! Project Euler Problem 70: Totient Permutation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_070/sol1.py

const std = @import("std");
const testing = std.testing;

pub fn hasSameDigits(num1: u32, num2: u32) bool {
    var counts = [_]i8{0} ** 10;
    var a = num1;
    var b = num2;
    if (a == 0) counts[0] += 1;
    if (b == 0) counts[0] -= 1;
    while (a > 0) : (a /= 10) counts[a % 10] += 1;
    while (b > 0) : (b /= 10) counts[b % 10] -= 1;
    for (counts) |count| if (count != 0) return false;
    return true;
}

pub fn getTotients(allocator: std.mem.Allocator, max_one: u32) ![]u32 {
    const totients = try allocator.alloc(u32, max_one);
    for (0..max_one) |i| totients[i] = @intCast(i);
    var i: u32 = 2;
    while (i < max_one) : (i += 1) {
        if (totients[i] == i) {
            var j = i;
            while (j < max_one) : (j += i) totients[j] -= totients[j] / i;
        }
    }
    return totients;
}

/// Finds the value of n for which n/phi(n) is minimized among digit permutations.
/// Time complexity: O(n log log n) for the totient sieve plus digit checks
/// Space complexity: O(n)
pub fn solution(allocator: std.mem.Allocator, max_n: u32) !u32 {
    const totients = try getTotients(allocator, max_n + 1);
    defer allocator.free(totients);

    var min_numerator: u32 = 1;
    var min_denominator: u32 = 0;
    var i: u32 = 2;
    while (i <= max_n) : (i += 1) {
        const t = totients[i];
        if (@as(u64, i) * min_denominator < @as(u64, min_numerator) * t and hasSameDigits(i, t)) {
            min_numerator = i;
            min_denominator = t;
        }
    }
    return min_numerator;
}

test "problem 070: helpers" {
    try testing.expect(hasSameDigits(123456789, 987654321));
    try testing.expect(!hasSameDigits(123, 23));
    try testing.expect(!hasSameDigits(1234566, 123456));

    const totients = try getTotients(testing.allocator, 10);
    defer testing.allocator.free(totients);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 1, 2, 2, 4, 2, 6, 4, 6 }, totients);
}

test "problem 070: python reference" {
    try testing.expectEqual(@as(u32, 21), try solution(testing.allocator, 100));
    try testing.expectEqual(@as(u32, 4435), try solution(testing.allocator, 10000));
    try testing.expectEqual(@as(u32, 8319823), try solution(testing.allocator, 10_000_000));
}
