//! Narcissistic Number Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/narcissistic_number.py

const std = @import("std");
const testing = std.testing;

fn digitPower(cache: *[20][10]u64, digit: u8, power: u8) u64 {
    if (cache[power][digit] == 0 and !(digit == 0 and power > 0)) {
        var value: u64 = 1;
        var i: u8 = 0;
        while (i < power) : (i += 1) value *= digit;
        cache[power][digit] = value;
    }
    return cache[power][digit];
}

fn digitCount(number: u64) u8 {
    if (number == 0) return 1;
    var temp = number;
    var count: u8 = 0;
    while (temp > 0) : (temp /= 10) count += 1;
    return count;
}

/// Finds all narcissistic numbers below `limit`.
/// Time complexity: O(limit * digits), Space complexity: O(results)
pub fn findNarcissisticNumbers(allocator: std.mem.Allocator, limit: u64) ![]u64 {
    if (limit == 0) return try allocator.alloc(u64, 0);

    var cache = [_][10]u64{[_]u64{0} ** 10} ** 20;
    var out = std.ArrayListUnmanaged(u64){};
    errdefer out.deinit(allocator);

    var number: u64 = 0;
    while (number < limit) : (number += 1) {
        const digits = digitCount(number);
        var remaining = number;
        var digit_sum: u64 = 0;
        if (number == 0) {
            digit_sum = 0;
        } else {
            while (remaining > 0) : (remaining /= 10) {
                const digit: u8 = @intCast(remaining % 10);
                digit_sum += digitPower(&cache, digit, digits);
            }
        }
        if (digit_sum == number) try out.append(allocator, number);
    }

    return try out.toOwnedSlice(allocator);
}

test "narcissistic number: python examples" {
    const numbers10 = try findNarcissisticNumbers(testing.allocator, 10);
    defer testing.allocator.free(numbers10);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, numbers10);

    const numbers160 = try findNarcissisticNumbers(testing.allocator, 160);
    defer testing.allocator.free(numbers160);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 153 }, numbers160);
}

test "narcissistic number: known ranges" {
    const numbers1000 = try findNarcissisticNumbers(testing.allocator, 1000);
    defer testing.allocator.free(numbers1000);
    try testing.expectEqualSlices(u64, &[_]u64{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 153, 370, 371, 407 }, numbers1000);

    const numbers10000 = try findNarcissisticNumbers(testing.allocator, 10000);
    defer testing.allocator.free(numbers10000);
    try testing.expectEqual(@as(u64, 9474), numbers10000[numbers10000.len - 1]);
}

test "narcissistic number: boundaries and extreme cache reuse" {
    const numbers0 = try findNarcissisticNumbers(testing.allocator, 0);
    defer testing.allocator.free(numbers0);
    try testing.expectEqual(@as(usize, 0), numbers0.len);

    const numbers1 = try findNarcissisticNumbers(testing.allocator, 1);
    defer testing.allocator.free(numbers1);
    try testing.expectEqualSlices(u64, &[_]u64{0}, numbers1);

    const numbers100000 = try findNarcissisticNumbers(testing.allocator, 100000);
    defer testing.allocator.free(numbers100000);
    try testing.expectEqual(@as(u64, 93084), numbers100000[numbers100000.len - 1]);
}
