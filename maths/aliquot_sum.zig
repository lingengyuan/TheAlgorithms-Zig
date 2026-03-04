//! Aliquot Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/aliquot_sum.py

const std = @import("std");
const testing = std.testing;

pub const AliquotError = error{ InvalidInput, Overflow };

/// Returns the aliquot sum of a positive integer.
/// Aliquot sum is the sum of all proper divisors (< n) that divide n evenly.
/// Time complexity: O(n), Space complexity: O(1)
pub fn aliquotSum(input_num: i64) AliquotError!u64 {
    if (input_num <= 0) return AliquotError.InvalidInput;

    const n: u64 = @intCast(input_num);
    if (n == 1) return 0;

    var sum: u128 = 0;
    var divisor: u64 = 1;
    const half = n / 2;
    while (divisor <= half) : (divisor += 1) {
        if (n % divisor == 0) {
            sum += divisor;
            if (sum > std.math.maxInt(u64)) return AliquotError.Overflow;
        }
    }

    return @intCast(sum);
}

test "aliquot sum: known values" {
    try testing.expectEqual(@as(u64, 9), try aliquotSum(15));
    try testing.expectEqual(@as(u64, 6), try aliquotSum(6));
    try testing.expectEqual(@as(u64, 16), try aliquotSum(12));
    try testing.expectEqual(@as(u64, 0), try aliquotSum(1));
    try testing.expectEqual(@as(u64, 1), try aliquotSum(19));
}

test "aliquot sum: invalid input" {
    try testing.expectError(AliquotError.InvalidInput, aliquotSum(-1));
    try testing.expectError(AliquotError.InvalidInput, aliquotSum(0));
}
