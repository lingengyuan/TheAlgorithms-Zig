//! PrimeLib - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/primelib.py

const std = @import("std");
const testing = std.testing;
const gcd_mod = @import("gcd.zig");
const lcm_mod = @import("lcm.zig");
const prime_check_mod = @import("prime_check.zig");
const factorial_mod = @import("factorial.zig");

pub const PrimeLibError = error{
    InvalidInput,
    DivisionByZero,
    Overflow,
    OutOfMemory,
};

pub const SimplifiedFraction = struct {
    numerator: i64,
    denominator: i64,
};

/// Returns whether `number` is prime.
/// Time complexity: O(sqrt(n)), Space complexity: O(1)
pub fn isPrime(number: u64) bool {
    return prime_check_mod.isPrime(number);
}

/// Returns all primes in `[2, n]` using a simple sieve.
/// Time complexity: O(n log log n), Space complexity: O(n)
pub fn sieveEr(allocator: std.mem.Allocator, n: u64) PrimeLibError![]u64 {
    if (n <= 2) return error.InvalidInput;

    const composite = try allocator.alloc(bool, n + 1);
    defer allocator.free(composite);
    @memset(composite, false);

    var primes = std.ArrayListUnmanaged(u64){};
    defer primes.deinit(allocator);

    var value: u64 = 2;
    while (value <= n) : (value += 1) {
        if (composite[value]) continue;
        try primes.append(allocator, value);
        if (value > n / value) continue;
        var multiple = value * value;
        while (multiple <= n) : (multiple += value) {
            composite[multiple] = true;
        }
    }

    return try primes.toOwnedSlice(allocator);
}

/// Returns all primes in `[2, n]`.
/// Time complexity: O(n log log n), Space complexity: O(n)
pub fn getPrimeNumbers(allocator: std.mem.Allocator, n: u64) PrimeLibError![]u64 {
    return sieveEr(allocator, n);
}

/// Returns the prime factorization of `number`.
/// Time complexity: O(sqrt(n)), Space complexity: O(log n)
pub fn primeFactorization(allocator: std.mem.Allocator, number: u64) PrimeLibError![]u64 {
    var factors = std.ArrayListUnmanaged(u64){};
    defer factors.deinit(allocator);

    if (number <= 1) {
        try factors.append(allocator, number);
        return try factors.toOwnedSlice(allocator);
    }

    var quotient = number;
    while (quotient % 2 == 0) : (quotient /= 2) {
        try factors.append(allocator, 2);
    }

    var factor: u64 = 3;
    while (factor <= quotient / factor) : (factor += 2) {
        while (quotient % factor == 0) : (quotient /= factor) {
            try factors.append(allocator, factor);
        }
    }

    if (quotient > 1) try factors.append(allocator, quotient);
    return try factors.toOwnedSlice(allocator);
}

pub fn greatestPrimeFactor(allocator: std.mem.Allocator, number: u64) PrimeLibError!u64 {
    const factors = try primeFactorization(allocator, number);
    defer allocator.free(factors);
    return factors[factors.len - 1];
}

pub fn smallestPrimeFactor(allocator: std.mem.Allocator, number: u64) PrimeLibError!u64 {
    const factors = try primeFactorization(allocator, number);
    defer allocator.free(factors);
    return factors[0];
}

pub fn isEven(number: i64) bool {
    return @mod(number, 2) == 0;
}

pub fn isOdd(number: i64) bool {
    return @mod(number, 2) != 0;
}

/// Returns one Goldbach pair for `number`.
/// Time complexity: O(pi(n)^2), Space complexity: O(pi(n))
pub fn goldbach(allocator: std.mem.Allocator, number: u64) PrimeLibError![]u64 {
    if (number <= 2 or !isEven(@intCast(number))) return error.InvalidInput;

    const primes = try getPrimeNumbers(allocator, number);
    defer allocator.free(primes);

    var result = std.ArrayListUnmanaged(u64){};
    defer result.deinit(allocator);

    for (primes, 0..) |left, i| {
        for (primes[i..]) |right| {
            if (left + right == number) {
                try result.append(allocator, left);
                try result.append(allocator, right);
                return try result.toOwnedSlice(allocator);
            }
        }
    }

    return error.InvalidInput;
}

pub fn kgV(number1: u64, number2: u64) PrimeLibError!u64 {
    if (number1 < 1 or number2 < 1) return error.InvalidInput;
    return lcm_mod.lcm(@intCast(number1), @intCast(number2));
}

pub fn getPrime(n: u64) u64 {
    var index: u64 = 0;
    var answer: u64 = 2;
    while (index < n) : (index += 1) {
        answer += 1;
        while (!isPrime(answer)) : (answer += 1) {}
    }
    return answer;
}

pub fn getPrimesBetween(allocator: std.mem.Allocator, p_number_1: u64, p_number_2: u64) PrimeLibError![]u64 {
    if (!isPrime(p_number_1) or !isPrime(p_number_2) or p_number_1 >= p_number_2) {
        return error.InvalidInput;
    }

    var number = p_number_1 + 1;
    var result = std.ArrayListUnmanaged(u64){};
    defer result.deinit(allocator);

    while (number < p_number_2) : (number += 1) {
        if (isPrime(number)) try result.append(allocator, number);
    }

    return try result.toOwnedSlice(allocator);
}

pub fn getDivisors(allocator: std.mem.Allocator, n: u64) PrimeLibError![]u64 {
    if (n < 1) return error.InvalidInput;

    var result = std.ArrayListUnmanaged(u64){};
    defer result.deinit(allocator);

    var divisor: u64 = 1;
    while (divisor <= n) : (divisor += 1) {
        if (n % divisor == 0) try result.append(allocator, divisor);
    }

    return try result.toOwnedSlice(allocator);
}

pub fn isPerfectNumber(allocator: std.mem.Allocator, number: u64) PrimeLibError!bool {
    if (number <= 1) return error.InvalidInput;
    const divisors = try getDivisors(allocator, number);
    defer allocator.free(divisors);

    var sum: u64 = 0;
    for (divisors[0 .. divisors.len - 1]) |divisor| {
        sum += divisor;
    }
    return sum == number;
}

pub fn simplifyFraction(numerator: i64, denominator: i64) PrimeLibError!SimplifiedFraction {
    if (denominator == 0) return error.DivisionByZero;
    const divisor = gcd_mod.gcd(numerator, denominator);
    return .{
        .numerator = @divTrunc(numerator, @as(i64, @intCast(divisor))),
        .denominator = @divTrunc(denominator, @as(i64, @intCast(divisor))),
    };
}

pub fn factorial(n: u32) PrimeLibError!u64 {
    return factorial_mod.factorial(n) catch return error.Overflow;
}

/// Returns the Python-reference Fibonacci term where `fib(0) == 1`.
/// Time complexity: O(n), Space complexity: O(1)
pub fn fib(n: u32) PrimeLibError!u128 {
    var tmp: u128 = 0;
    var fib1: u128 = 1;
    var ans: u128 = 1;

    var index: u32 = 0;
    while (index < n -| 1) : (index += 1) {
        tmp = ans;
        const with_overflow = @addWithOverflow(ans, fib1);
        if (with_overflow[1] != 0) return error.Overflow;
        ans = with_overflow[0];
        fib1 = tmp;
    }
    return ans;
}

test "primelib: prime helpers match python examples" {
    const alloc = testing.allocator;

    try testing.expect(isPrime(3));
    try testing.expect(!isPrime(10));
    try testing.expectEqual(@as(u64, 0), try greatestPrimeFactor(alloc, 0));
    try testing.expectEqual(@as(u64, 2), try greatestPrimeFactor(alloc, 8));
    try testing.expectEqual(@as(u64, 41), try greatestPrimeFactor(alloc, 287));
    try testing.expectEqual(@as(u64, 0), try smallestPrimeFactor(alloc, 0));
    try testing.expectEqual(@as(u64, 7), try smallestPrimeFactor(alloc, 287));
}

test "primelib: sieve and factorization" {
    const alloc = testing.allocator;

    const primes = try sieveEr(alloc, 8);
    defer alloc.free(primes);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 5, 7 }, primes);

    const factors = try primeFactorization(alloc, 287);
    defer alloc.free(factors);
    try testing.expectEqualSlices(u64, &[_]u64{ 7, 41 }, factors);

    const one = try primeFactorization(alloc, 1);
    defer alloc.free(one);
    try testing.expectEqualSlices(u64, &[_]u64{1}, one);
}

test "primelib: parity, lcm and goldbach" {
    const alloc = testing.allocator;
    try testing.expect(isEven(8));
    try testing.expect(!isEven(-1));
    try testing.expect(isOdd(-1));
    try testing.expectEqual(@as(u64, 40), try kgV(8, 10));

    const pair = try goldbach(alloc, 8);
    defer alloc.free(pair);
    try testing.expectEqualSlices(u64, &[_]u64{ 3, 5 }, pair);

    const pair_four = try goldbach(alloc, 4);
    defer alloc.free(pair_four);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 2 }, pair_four);

    try testing.expectError(error.InvalidInput, goldbach(alloc, 9));
}

test "primelib: prime ranges and divisors" {
    const alloc = testing.allocator;

    try testing.expectEqual(@as(u64, 2), getPrime(0));
    try testing.expectEqual(@as(u64, 23), getPrime(8));

    const between = try getPrimesBetween(alloc, 3, 67);
    defer alloc.free(between);
    try testing.expectEqual(@as(usize, 16), between.len);
    try testing.expectEqual(@as(u64, 5), between[0]);
    try testing.expectEqual(@as(u64, 61), between[between.len - 1]);

    const divisors = try getDivisors(alloc, 8);
    defer alloc.free(divisors);
    try testing.expectEqualSlices(u64, &[_]u64{ 1, 2, 4, 8 }, divisors);
}

test "primelib: perfect number, fraction, factorial and fibonacci" {
    const alloc = testing.allocator;

    try testing.expect(try isPerfectNumber(alloc, 28));
    try testing.expect(!(try isPerfectNumber(alloc, 824)));
    try testing.expectError(error.InvalidInput, isPerfectNumber(alloc, 1));

    try testing.expectEqual(SimplifiedFraction{ .numerator = 1, .denominator = 2 }, try simplifyFraction(10, 20));
    try testing.expectEqual(SimplifiedFraction{ .numerator = 10, .denominator = -1 }, try simplifyFraction(10, -1));
    try testing.expectError(error.DivisionByZero, simplifyFraction(10, 0));

    try testing.expectEqual(@as(u64, 1), try factorial(0));
    try testing.expectEqual(@as(u64, 2_432_902_008_176_640_000), try factorial(20));
    try testing.expectError(error.Overflow, factorial(21));

    try testing.expectEqual(@as(u128, 1), try fib(0));
    try testing.expectEqual(@as(u128, 8), try fib(5));
    try testing.expectEqual(@as(u128, 354_224_848_179_261_915_075), try fib(99));
}

test "primelib: invalid edge cases" {
    const alloc = testing.allocator;
    try testing.expectError(error.InvalidInput, sieveEr(alloc, 2));
    try testing.expectError(error.InvalidInput, kgV(10, 0));
    try testing.expectError(error.InvalidInput, getPrimesBetween(alloc, 4, 7));
    try testing.expectError(error.InvalidInput, getDivisors(alloc, 0));
}
