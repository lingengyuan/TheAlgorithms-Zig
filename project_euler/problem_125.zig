//! Project Euler Problem 125: Palindromic Sums - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_125/sol1.py

const std = @import("std");
const testing = std.testing;

pub fn isPalindrome(n: u64) bool {
    if (n % 10 == 0) return false;
    var value = n;
    var reversed: u64 = 0;
    while (value > 0) : (value /= 10) reversed = reversed * 10 + value % 10;
    return reversed == n;
}

/// Returns the sum of palindromic values below `limit` that are sums of consecutive positive squares.
/// Time complexity: roughly O(sqrt(limit)^2)
/// Space complexity: O(number of palindromic sums)
pub fn solution(allocator: std.mem.Allocator, limit: u64) !u64 {
    var seen = std.AutoHashMapUnmanaged(u64, void){};
    defer seen.deinit(allocator);

    var answer: u64 = 0;
    var first_square: u64 = 1;
    var initial_sum: u64 = 5;
    while (initial_sum < limit) {
        var current_sum = initial_sum;
        var last_square = first_square + 1;
        while (current_sum < limit) {
            if (isPalindrome(current_sum)) {
                const gop = try seen.getOrPut(allocator, current_sum);
                if (!gop.found_existing) answer += current_sum;
            }
            last_square += 1;
            current_sum += last_square * last_square;
        }
        first_square += 1;
        initial_sum = first_square * first_square + (first_square + 1) * (first_square + 1);
    }
    return answer;
}

test "problem 125: python reference" {
    try testing.expectEqual(@as(u64, 4164), try solution(testing.allocator, 1000));
    try testing.expectEqual(@as(u64, 2906969179), try solution(testing.allocator, 100_000_000));
}

test "problem 125: palindrome helper and tiny limits" {
    try testing.expect(isPalindrome(12521));
    try testing.expect(!isPalindrome(12522));
    try testing.expect(!isPalindrome(12210));
    try testing.expectEqual(@as(u64, 0), try solution(testing.allocator, 5));
    try testing.expectEqual(@as(u64, 5), try solution(testing.allocator, 6));
}
