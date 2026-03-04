//! Happy Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/special_numbers/happy_number.py

const std = @import("std");
const testing = std.testing;

pub const HappyNumberError = error{InvalidInput};

/// Returns true when `number` is a happy number.
/// Time complexity: O(k), Space complexity: O(1), where k is cycle detection steps.
pub fn isHappyNumber(number: i64) HappyNumberError!bool {
    if (number <= 0) return HappyNumberError.InvalidInput;

    var current: u64 = @intCast(number);
    // In base-10, every non-happy sequence eventually reaches cycle containing 4.
    while (current != 1 and current != 4) {
        current = sumOfSquaredDigits(current);
    }
    return current == 1;
}

fn sumOfSquaredDigits(number: u64) u64 {
    var n = number;
    var sum: u64 = 0;
    while (n > 0) {
        const digit = n % 10;
        sum += digit * digit;
        n /= 10;
    }
    return sum;
}

test "happy number: python reference examples" {
    try testing.expect(try isHappyNumber(19));
    try testing.expect(!(try isHappyNumber(2)));
    try testing.expect(try isHappyNumber(23));
    try testing.expect(try isHappyNumber(1));
}

test "happy number: invalid inputs" {
    try testing.expectError(HappyNumberError.InvalidInput, isHappyNumber(0));
    try testing.expectError(HappyNumberError.InvalidInput, isHappyNumber(-19));
}

test "happy number: edge and extreme cases" {
    try testing.expect(try isHappyNumber(1_000_000_000_000_000_000));
    try testing.expect(!(try isHappyNumber(std.math.maxInt(i64))));
}
