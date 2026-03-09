//! Project Euler Problem 32: Pandigital Products - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_032/sol32.py

const std = @import("std");
const testing = std.testing;

fn parseDigits(value: []const u8) u32 {
    var result: u32 = 0;
    for (value) |ch| {
        result = result * 10 + (ch - '0');
    }
    return result;
}

/// Python-reference validator for a 9-digit tuple-like combination.
pub fn isCombinationValid(combination: [9]u8) bool {
    const left2 = parseDigits(combination[0..2]);
    const right3 = parseDigits(combination[2..5]);
    const product4 = parseDigits(combination[5..9]);

    if (left2 * right3 == product4) return true;

    const left1 = parseDigits(combination[0..1]);
    const right4 = parseDigits(combination[1..5]);
    return left1 * right4 == product4;
}

fn isPandigitalIdentity(a: u32, b: u32, product: u32) bool {
    var mask: u16 = 0;
    var count: u8 = 0;

    var values = [3]u32{ a, b, product };
    for (&values) |*value_ptr| {
        var value = value_ptr.*;
        while (value > 0) {
            const digit: u16 = @intCast(value % 10);
            if (digit == 0) return false;

            const bit: u16 = @as(u16, 1) << @intCast(digit);
            if ((mask & bit) != 0) return false;

            mask |= bit;
            count += 1;
            value /= 10;
        }
    }

    return count == 9 and mask == 0b11_1111_1110;
}

/// Returns the sum of all unique pandigital products.
///
/// Time complexity: O(search_space)
/// Space complexity: O(1)
pub fn solution() u32 {
    var seen_products = [_]bool{false} ** 10_000;
    var total: u32 = 0;

    // Cases: 1-digit * 4-digit = 4-digit OR 2-digit * 3-digit = 4-digit.
    var a: u32 = 2;
    while (a <= 99) : (a += 1) {
        const b_start: u32 = if (a < 10) 1234 else 123;
        const b_end: u32 = if (a < 10) 9876 else 987;

        var b: u32 = b_start;
        while (b <= b_end) : (b += 1) {
            const product = a * b;
            if (product < 1000 or product > 9999) continue;

            if (isPandigitalIdentity(a, b, product) and !seen_products[product]) {
                seen_products[product] = true;
                total += product;
            }
        }
    }

    return total;
}

test "problem 032: python reference" {
    try testing.expectEqual(@as(u32, 45_228), solution());
}

test "problem 032: combination validator and edge cases" {
    try testing.expect(isCombinationValid([9]u8{ '3', '9', '1', '8', '6', '7', '2', '5', '4' }));
    try testing.expect(!isCombinationValid([9]u8{ '1', '2', '3', '4', '5', '6', '7', '8', '9' }));

    try testing.expect(isPandigitalIdentity(39, 186, 7254));
    try testing.expect(isPandigitalIdentity(4, 1738, 6952));
    try testing.expect(!isPandigitalIdentity(12, 34, 408));
    try testing.expect(!isPandigitalIdentity(1, 2345, 2345));
}
