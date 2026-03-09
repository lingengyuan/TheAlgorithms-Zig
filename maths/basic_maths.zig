//! Basic Maths Utilities - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/basic_maths.py

const std = @import("std");
const testing = std.testing;
const prime_factors_mod = @import("prime_factors.zig");

pub const BasicMathsError = error{InvalidInput};

/// Returns the prime factors of a positive integer.
pub fn primeFactors(allocator: std.mem.Allocator, n: i64) (BasicMathsError || std.mem.Allocator.Error)![]i64 {
    if (n <= 0) return error.InvalidInput;
    return prime_factors_mod.primeFactors(allocator, n);
}

/// Returns the number of positive divisors of `n`.
pub fn numberOfDivisors(n: i64) BasicMathsError!u64 {
    if (n <= 0) return error.InvalidInput;
    var value = n;
    var divisors: u64 = 1;
    var count: u64 = 1;
    while (@rem(value, 2) == 0) {
        count += 1;
        value = @divTrunc(value, 2);
    }
    divisors *= count;
    var i: i64 = 3;
    while (i * i <= value) : (i += 2) {
        count = 1;
        while (@rem(value, i) == 0) {
            count += 1;
            value = @divTrunc(value, i);
        }
        divisors *= count;
    }
    if (value > 1) divisors *= 2;
    return divisors;
}

/// Returns the sum of positive divisors of `n`.
pub fn sumOfDivisors(n: i64) BasicMathsError!u64 {
    if (n <= 0) return error.InvalidInput;
    var value = n;
    var sum: u64 = 1;
    var count: u64 = 1;
    while (@rem(value, 2) == 0) {
        count += 1;
        value = @divTrunc(value, 2);
    }
    if (count > 1) {
        sum *= (@as(u64, 1) << @intCast(count)) - 1;
    }

    var i: i64 = 3;
    while (i * i <= value) : (i += 2) {
        count = 1;
        while (@rem(value, i) == 0) {
            count += 1;
            value = @divTrunc(value, i);
        }
        if (count > 1) {
            const base: u64 = @intCast(i);
            var power_term: u64 = 1;
            var step: u64 = 0;
            while (step < count) : (step += 1) power_term *= base;
            sum *= @divTrunc(power_term - 1, base - 1);
        }
    }
    if (value > 1) {
        const base: u64 = @intCast(value);
        sum *= base + 1;
    }
    return sum;
}

/// Returns Euler's phi function value for `n`.
pub fn eulerPhi(allocator: std.mem.Allocator, n: i64) (BasicMathsError || std.mem.Allocator.Error)!u64 {
    if (n <= 0) return error.InvalidInput;
    const unique = try prime_factors_mod.uniquePrimeFactors(allocator, n);
    defer allocator.free(unique);

    var result = @as(f64, @floatFromInt(n));
    for (unique) |factor| {
        result *= (@as(f64, @floatFromInt(factor - 1)) / @as(f64, @floatFromInt(factor)));
    }
    return @intFromFloat(result);
}

test "basic maths: python reference examples" {
    const alloc = testing.allocator;
    const pf = try primeFactors(alloc, 100);
    defer alloc.free(pf);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 2, 5, 5 }, pf);
    try testing.expectEqual(@as(u64, 9), try numberOfDivisors(100));
    try testing.expectEqual(@as(u64, 217), try sumOfDivisors(100));
    try testing.expectEqual(@as(u64, 40), try eulerPhi(alloc, 100));
}

test "basic maths: edge cases" {
    const alloc = testing.allocator;
    try testing.expectError(error.InvalidInput, primeFactors(alloc, 0));
    try testing.expectError(error.InvalidInput, numberOfDivisors(-10));
    try testing.expectError(error.InvalidInput, sumOfDivisors(0));
    try testing.expectError(error.InvalidInput, eulerPhi(alloc, -10));
}
