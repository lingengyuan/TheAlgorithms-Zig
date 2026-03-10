//! Project Euler Problem 97: Large Non-Mersenne Prime - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_097/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem097Error = error{InvalidInput};

fn powMod(base: u64, exponent: u64, modulus: u64) u64 {
    if (modulus == 1) return 0;
    var result: u64 = 1 % modulus;
    var b = base % modulus;
    var e = exponent;
    while (e > 0) : (e >>= 1) {
        if ((e & 1) == 1) result = @intCast((@as(u128, result) * b) % modulus);
        b = @intCast((@as(u128, b) * b) % modulus);
    }
    return result;
}

/// Returns the last `n` digits of `28433 * 2^7830457 + 1` as a decimal string.
/// Time complexity: O(log exponent)
/// Space complexity: O(n)
pub fn solution(allocator: std.mem.Allocator, n: i32) Problem097Error![]u8 {
    if (n < 0) return error.InvalidInput;
    const digits: u32 = @intCast(n);
    var modulus: u64 = 1;
    var i: u32 = 0;
    while (i < digits) : (i += 1) modulus *= 10;
    const number = if (digits == 0) 0 else (28_433 * powMod(2, 7_830_457, modulus) + 1) % modulus;
    return std.fmt.allocPrint(allocator, "{d}", .{number}) catch error.InvalidInput;
}

test "problem 097: python reference" {
    const alloc = testing.allocator;
    const p10 = try solution(alloc, 10);
    defer alloc.free(p10);
    try testing.expectEqualStrings("8739992577", p10);

    const p8 = try solution(alloc, 8);
    defer alloc.free(p8);
    try testing.expectEqualStrings("39992577", p8);

    const p1 = try solution(alloc, 1);
    defer alloc.free(p1);
    try testing.expectEqualStrings("7", p1);
}

test "problem 097: invalid input" {
    try testing.expectError(error.InvalidInput, solution(testing.allocator, -1));
}
