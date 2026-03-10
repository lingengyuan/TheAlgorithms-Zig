//! Project Euler Problem 135: Same Differences - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_135/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the count of n <= limit that have exactly ten distinct solutions.
/// Time complexity: roughly O(limit log limit)
/// Space complexity: O(limit)
pub fn solution(allocator: std.mem.Allocator, limit_input: u32) !u32 {
    const limit = limit_input + 1;
    const frequency = try allocator.alloc(u8, limit);
    defer allocator.free(frequency);
    @memset(frequency, 0);

    var first_term: u32 = 1;
    while (first_term < limit) : (first_term += 1) {
        var n = first_term;
        while (n < limit) : (n += first_term) {
            var common_difference = first_term + n / first_term;
            if (common_difference % 4 != 0) continue;
            common_difference /= 4;
            if (first_term > common_difference and first_term < 4 * common_difference) frequency[n] += 1;
        }
    }

    var count: u32 = 0;
    for (frequency[1..limit]) |value| {
        if (value == 10) count += 1;
    }
    return count;
}

test "problem 135: python reference" {
    try testing.expectEqual(@as(u32, 0), try solution(testing.allocator, 100));
    try testing.expectEqual(@as(u32, 45), try solution(testing.allocator, 10000));
    try testing.expectEqual(@as(u32, 292), try solution(testing.allocator, 50050));
}
