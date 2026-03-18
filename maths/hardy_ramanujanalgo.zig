//! Hardy Ramanujan Algo - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/hardy_ramanujanalgo.py

const std = @import("std");
const testing = std.testing;

/// Counts the number of distinct prime factors of `n`.
/// Time complexity: O(sqrt(n)), Space complexity: O(1)
pub fn exactPrimeFactorCount(n: u64) u32 {
    if (n < 2) return 0;

    var count: u32 = 0;
    var value = n;
    if (value % 2 == 0) {
        count += 1;
        while (value % 2 == 0) value /= 2;
    }

    var i: u64 = 3;
    while (i * i <= value) : (i += 2) {
        if (value % i == 0) {
            count += 1;
            while (value % i == 0) value /= i;
        }
    }

    if (value > 2) count += 1;
    return count;
}

/// Returns the Hardy-Ramanujan approximation `log(log(n))`.
/// Time complexity: O(1), Space complexity: O(1)
pub fn hardyRamanujanApprox(n: u64) !f64 {
    if (n <= 1) return error.InvalidInput;
    return @log(@log(@as(f64, @floatFromInt(n))));
}

test "hardy ramanujan algo: python reference example" {
    try testing.expectEqual(@as(u32, 3), exactPrimeFactorCount(51_242_183));
}

test "hardy ramanujan algo: edge and extreme cases" {
    try testing.expectEqual(@as(u32, 0), exactPrimeFactorCount(0));
    try testing.expectEqual(@as(u32, 0), exactPrimeFactorCount(1));
    try testing.expectEqual(@as(u32, 1), exactPrimeFactorCount(2));
    try testing.expectEqual(@as(u32, 1), exactPrimeFactorCount(1 << 20));
    try testing.expectEqual(@as(u32, 4), exactPrimeFactorCount(2 * 3 * 5 * 7));

    try testing.expectError(error.InvalidInput, hardyRamanujanApprox(1));
    try testing.expectApproxEqAbs(@as(f64, 2.8765), try hardyRamanujanApprox(51_242_183), 1e-4);
}
