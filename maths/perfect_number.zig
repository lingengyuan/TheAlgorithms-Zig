//! Perfect Number Check - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/perfect_number.py

const std = @import("std");
const testing = std.testing;

/// Returns true if `number` is a perfect number.
/// A perfect number equals the sum of its proper divisors.
/// Time complexity: O(n), Space complexity: O(1)
pub fn isPerfectNumber(number: i64) bool {
    if (number <= 1) return false;
    const n: u64 = @intCast(number);

    var divisor: u64 = 1;
    const half = n / 2;
    var sum: u128 = 0;
    while (divisor <= half) : (divisor += 1) {
        if (n % divisor == 0) {
            sum += divisor;
            if (sum > @as(u128, n)) return false;
        }
    }
    return sum == @as(u128, n);
}

test "perfect number: known values" {
    try testing.expect(!isPerfectNumber(27));
    try testing.expect(isPerfectNumber(28));
    try testing.expect(!isPerfectNumber(29));
    try testing.expect(isPerfectNumber(6));
    try testing.expect(!isPerfectNumber(12));
    try testing.expect(isPerfectNumber(496));
    try testing.expect(isPerfectNumber(8128));
}

test "perfect number: edge and extreme cases" {
    try testing.expect(!isPerfectNumber(0));
    try testing.expect(!isPerfectNumber(-1));
    try testing.expect(!isPerfectNumber(1));
    try testing.expect(isPerfectNumber(33_550_336)); // large reference perfect number
    try testing.expect(!isPerfectNumber(33_550_337));
}
