//! Project Euler Problem 65: Convergents of e - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_065/sol1.py

const std = @import("std");
const testing = std.testing;
const BigInt = std.math.big.int.Managed;

const Allocator = std.mem.Allocator;

fn sumDigits(num: u64) u64 {
    var value = num;
    var digit_sum: u64 = 0;
    while (value > 0) : (value /= 10) digit_sum += value % 10;
    return digit_sum;
}

fn sumDigitsBig(allocator: Allocator, value: *const BigInt) !u64 {
    const text = try value.toString(allocator, 10, .lower);
    defer allocator.free(text);

    var digit_sum: u64 = 0;
    for (text) |ch| digit_sum += ch - '0';
    return digit_sum;
}

/// Returns the sum of digits in the numerator of the `max_n`-th convergent of `e`.
/// Time complexity: O(max_n * bigint_digits)
/// Space complexity: O(bigint_digits)
pub fn solution(allocator: Allocator, max_n: usize) !u64 {
    if (max_n == 0) return 0;
    if (max_n == 1) return 2;

    var pre_numerator = try BigInt.initSet(allocator, 1);
    defer pre_numerator.deinit();

    var cur_numerator = try BigInt.initSet(allocator, 2);
    defer cur_numerator.deinit();

    var next_numerator = try BigInt.initSet(allocator, 0);
    defer next_numerator.deinit();

    var i: usize = 2;
    while (i <= max_n) : (i += 1) {
        const e_cont = if (i % 3 == 0) 2 * i / 3 else 1;
        var coeff = try BigInt.initSet(allocator, e_cont);
        defer coeff.deinit();

        try next_numerator.mul(&cur_numerator, &coeff);
        try next_numerator.add(&next_numerator, &pre_numerator);
        std.mem.swap(BigInt, &pre_numerator, &cur_numerator);
        std.mem.swap(BigInt, &cur_numerator, &next_numerator);
    }

    return sumDigitsBig(allocator, &cur_numerator);
}

test "problem 065: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u64, 13), try solution(alloc, 9));
    try testing.expectEqual(@as(u64, 17), try solution(alloc, 10));
    try testing.expectEqual(@as(u64, 91), try solution(alloc, 50));
    try testing.expectEqual(@as(u64, 272), try solution(alloc, 100));
}

test "problem 065: digit sum and edge convergents" {
    try testing.expectEqual(@as(u64, 1), sumDigits(1));
    try testing.expectEqual(@as(u64, 15), sumDigits(12345));
    try testing.expectEqual(@as(u64, 28), sumDigits(999001));
    try testing.expectEqual(@as(u64, 0), try solution(testing.allocator, 0));
    try testing.expectEqual(@as(u64, 2), try solution(testing.allocator, 1));
}
