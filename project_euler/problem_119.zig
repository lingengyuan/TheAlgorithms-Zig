//! Project Euler Problem 119: Digit Power Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_119/sol1.py

const std = @import("std");
const testing = std.testing;

fn digitSum(n: u64) u32 {
    var value = n;
    var sum: u32 = 0;
    while (value > 0) : (value /= 10) sum += @intCast(value % 10);
    return sum;
}

/// Returns the nth digit-power-sum value.
/// Time complexity: bounded search over bases and powers
/// Space complexity: O(number of candidates)
pub fn solution(allocator: std.mem.Allocator, n: u32) !u64 {
    var values = std.ArrayListUnmanaged(u64){};
    defer values.deinit(allocator);

    var digit: u64 = 2;
    while (digit < 100) : (digit += 1) {
        var number: u128 = digit * digit;
        var power: u32 = 2;
        while (power < 100 and number <= std.math.maxInt(u64)) : (power += 1) {
            const candidate: u64 = @intCast(number);
            if (candidate >= 10 and digitSum(candidate) == digit) try values.append(allocator, candidate);
            number *= digit;
        }
    }

    std.mem.sort(u64, values.items, {}, comptime std.sort.asc(u64));
    var write_idx: usize = 0;
    for (values.items) |value| {
        if (write_idx == 0 or values.items[write_idx - 1] != value) {
            values.items[write_idx] = value;
            write_idx += 1;
        }
    }
    return values.items[n - 1];
}

test "problem 119: python reference" {
    try testing.expectEqual(@as(u64, 512), try solution(testing.allocator, 2));
    try testing.expectEqual(@as(u64, 5832), try solution(testing.allocator, 5));
    try testing.expectEqual(@as(u64, 614656), try solution(testing.allocator, 10));
    try testing.expectEqual(@as(u64, 248155780267521), try solution(testing.allocator, 30));
}
