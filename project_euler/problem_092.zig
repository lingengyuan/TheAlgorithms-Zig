//! Project Euler Problem 92: Square Digit Chains - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_092/sol1.py

const std = @import("std");
const testing = std.testing;

fn nextNumber(number: u32) u32 {
    if (number == 0) return 0;
    var n = number;
    var sum: u32 = 0;
    while (n > 0) : (n /= 10) {
        const digit = n % 10;
        sum += digit * digit;
    }
    return sum;
}

fn endsAtOne(cache: []?bool, number: u32) bool {
    if (cache[number]) |result| return result;
    const result = endsAtOne(cache, nextNumber(number));
    cache[number] = result;
    return result;
}

/// Returns the count of starting numbers below `number` whose chains arrive at `89`.
/// Time complexity: O(number * digits)
/// Space complexity: O(1) beyond the small chain cache
pub fn solution(number: u32) !u32 {
    if (number == 0) return 0;

    var cache = [_]?bool{null} ** 568;
    cache[0] = true; // number 1
    cache[1] = true; // defensive alias for nextNumber(10)
    cache[89] = false;

    var count: u32 = 0;
    var i: u32 = 1;
    while (i < number) : (i += 1) {
        const terminal = endsAtOne(&cache, nextNumber(i));
        if (!terminal) count += 1;
    }
    return count;
}

test "problem 092: python reference" {
    try testing.expectEqual(@as(u32, 80), try solution(100));
    try testing.expectEqual(@as(u32, 8_581_146), try solution(10_000_000));
}

test "problem 092: helper semantics and edges" {
    try testing.expectEqual(@as(u32, 32), nextNumber(44));
    try testing.expectEqual(@as(u32, 1), nextNumber(10));
    try testing.expectEqual(@as(u32, 13), nextNumber(32));
    try testing.expectEqual(@as(u32, 0), try solution(0));
    try testing.expectEqual(@as(u32, 0), try solution(1));
}
