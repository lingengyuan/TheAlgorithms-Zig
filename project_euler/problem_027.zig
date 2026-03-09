//! Project Euler Problem 27: Quadratic Primes - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_027/sol1.py

const std = @import("std");
const testing = std.testing;

/// Checks primality in O(sqrt(n)) using 6k +/- 1 optimization.
pub fn isPrime(number: i64) bool {
    if (number == 2 or number == 3) return true;
    if (number < 2 or @mod(number, 2) == 0 or @mod(number, 3) == 0) return false;

    var i: i64 = 5;
    while (i * i <= number) : (i += 6) {
        if (@mod(number, i) == 0 or @mod(number, i + 2) == 0) return false;
    }
    return true;
}

/// Returns product a*b for quadratic n^2 + a*n + b yielding longest run of
/// consecutive primes starting at n=0, with bounds matching Python reference.
///
/// Time complexity: O(a_limit * b_limit * run * sqrt(value))
/// Space complexity: O(1)
pub fn solution(a_limit: i64, b_limit: i64) i64 {
    var best_len: i64 = 0;
    var best_a: i64 = 0;
    var best_b: i64 = 0;

    var a = -a_limit + 1;
    while (a < a_limit) : (a += 1) {
        var b: i64 = 2;
        while (b < b_limit) : (b += 1) {
            if (!isPrime(b)) continue;

            var count: i64 = 0;
            var n: i64 = 0;

            while (true) {
                const value = n * n + a * n + b;
                if (!isPrime(value)) break;
                count += 1;
                n += 1;
            }

            if (count > best_len) {
                best_len = count;
                best_a = a;
                best_b = b;
            }
        }
    }

    return best_a * best_b;
}

test "problem 027: python reference" {
    try testing.expectEqual(@as(i64, -59_231), solution(1000, 1000));
    try testing.expectEqual(@as(i64, -59_231), solution(200, 1000));
    try testing.expectEqual(@as(i64, -4925), solution(200, 200));
}

test "problem 027: boundaries and prime helper" {
    try testing.expectEqual(@as(i64, 0), solution(-1000, 1000));
    try testing.expectEqual(@as(i64, 0), solution(-1000, -1000));
    try testing.expectEqual(@as(i64, 0), solution(0, 1000));
    try testing.expectEqual(@as(i64, 0), solution(1000, 2));
    try testing.expectEqual(@as(i64, 0), solution(1, 3));

    try testing.expect(isPrime(2));
    try testing.expect(isPrime(3));
    try testing.expect(!isPrime(0));
    try testing.expect(!isPrime(1));
    try testing.expect(!isPrime(-10));
    try testing.expect(!isPrime(27));
    try testing.expect(isPrime(2999));
}
