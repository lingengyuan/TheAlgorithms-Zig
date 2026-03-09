//! Project Euler Problem 23: Non-Abundant Sums - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_023/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem023Error = error{
    OutOfMemory,
};

/// Returns the sum of all positive integers <= limit that cannot be written
/// as the sum of two abundant numbers.
///
/// Time complexity: O(limit^2) in worst case (abundant pair marking)
/// Space complexity: O(limit)
pub fn solution(limit: usize, allocator: std.mem.Allocator) Problem023Error!u64 {
    if (limit == 0) return 0;

    var sum_divs = try allocator.alloc(u64, limit + 1);
    defer allocator.free(sum_divs);

    @memset(sum_divs, 1);
    sum_divs[0] = 0;
    if (limit >= 1) sum_divs[1] = 0;

    var i: usize = 2;
    while (i * i <= limit) : (i += 1) {
        const square = i * i;
        sum_divs[square] += i;

        var k: usize = i + 1;
        while (k <= limit / i) : (k += 1) {
            sum_divs[k * i] += k + i;
        }
    }

    var abundants = std.ArrayListUnmanaged(usize){};
    defer abundants.deinit(allocator);

    for (1..limit + 1) |n| {
        if (sum_divs[n] > n) {
            try abundants.append(allocator, n);
        }
    }

    var can_be_written = try allocator.alloc(bool, limit + 1);
    defer allocator.free(can_be_written);
    @memset(can_be_written, false);

    for (abundants.items, 0..) |a, idx| {
        for (abundants.items[idx..]) |b| {
            const sum = a + b;
            if (sum > limit) break;
            can_be_written[sum] = true;
        }
    }

    var total: u64 = 0;
    for (1..limit + 1) |n| {
        if (!can_be_written[n]) {
            total += n;
        }
    }

    return total;
}

test "problem 023: python reference" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 4_179_871), try solution(28_123, allocator));
    try testing.expectEqual(@as(u64, 240_492), try solution(1000, allocator));
    try testing.expectEqual(@as(u64, 2_766), try solution(100, allocator));
}

test "problem 023: boundaries and small limits" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 0), try solution(0, allocator));
    try testing.expectEqual(@as(u64, 1), try solution(1, allocator));
    try testing.expectEqual(@as(u64, 3), try solution(2, allocator));
    try testing.expectEqual(@as(u64, 55), try solution(10, allocator));

    // 24 is the smallest sum of two abundant numbers.
    try testing.expectEqual(@as(u64, 276), try solution(24, allocator));
    try testing.expectEqual(@as(u64, 301), try solution(25, allocator));
}
