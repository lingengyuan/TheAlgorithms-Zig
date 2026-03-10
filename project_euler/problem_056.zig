//! Project Euler Problem 56: Powerful Digit Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_056/sol1.py

const std = @import("std");
const testing = std.testing;

fn multiplyDigitsSmall(allocator: std.mem.Allocator, digits: []const u8, factor: usize) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    if (factor == 0) {
        try out.append(allocator, 0);
        return try out.toOwnedSlice(allocator);
    }

    var carry: usize = 0;
    for (digits) |digit| {
        const product = @as(usize, digit) * factor + carry;
        try out.append(allocator, @intCast(product % 10));
        carry = product / 10;
    }
    while (carry > 0) {
        try out.append(allocator, @intCast(carry % 10));
        carry /= 10;
    }
    return try out.toOwnedSlice(allocator);
}

fn digitSum(digits: []const u8) usize {
    var sum: usize = 0;
    for (digits) |digit| sum += digit;
    return sum;
}

/// Returns the maximum digital sum among all `base^power` where `base < a` and `power < b`.
/// Time complexity: O(a * b * digits), Space complexity: O(digits)
pub fn solution(allocator: std.mem.Allocator, a: usize, b: usize) !usize {
    var best: usize = 0;

    for (0..a) |base| {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const local = arena.allocator();

        var digits = try local.alloc(u8, 1);
        digits[0] = 1;

        for (0..b) |_| {
            best = @max(best, digitSum(digits));
            digits = try multiplyDigitsSmall(local, digits, base);
        }
    }

    return best;
}

test "problem 056: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 45), try solution(alloc, 10, 10));
    try testing.expectEqual(@as(usize, 972), try solution(alloc, 100, 100));
    try testing.expectEqual(@as(usize, 1872), try solution(alloc, 100, 200));
}

test "problem 056: edge and extreme cases" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 0), try solution(alloc, 0, 0));
    try testing.expectEqual(@as(usize, 1), try solution(alloc, 1, 1));
    try testing.expectEqual(@as(usize, 1), try solution(alloc, 2, 1));
}
