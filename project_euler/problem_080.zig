//! Project Euler Problem 80: Square Root Digital Expansion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_080/sol1.py

const std = @import("std");
const testing = std.testing;
const BigInt = std.math.big.int.Managed;

const Allocator = std.mem.Allocator;

fn isPerfectSquare(value: u32) bool {
    const root = std.math.sqrt(value);
    return root * root == value;
}

fn mulScalarBig(allocator: Allocator, value: *const BigInt, scalar: u32) !BigInt {
    var scalar_big = try BigInt.initSet(allocator, scalar);
    defer scalar_big.deinit();

    var result = try BigInt.init(allocator);
    try result.mul(value, &scalar_big);
    return result;
}

fn addScalarBig(allocator: Allocator, value: *const BigInt, scalar: u32) !BigInt {
    var scalar_big = try BigInt.initSet(allocator, scalar);
    defer scalar_big.deinit();

    var result = try BigInt.init(allocator);
    try result.add(value, &scalar_big);
    return result;
}

fn subBig(allocator: Allocator, left: *const BigInt, right: *const BigInt) !BigInt {
    var result = try BigInt.init(allocator);
    try result.sub(left, right);
    return result;
}

fn cmpSmall(left: *const BigInt, right: *const BigInt) std.math.Order {
    return BigInt.order(left.*, right.*);
}

fn digitSumSquareRoot(allocator: Allocator, value: u32, digits: usize) !u32 {
    if (digits == 0 or isPerfectSquare(value)) return 0;

    var remainder = try BigInt.initSet(allocator, value);
    defer remainder.deinit();

    const integer_digit: u32 = blk: {
        var candidate: u32 = 9;
        while (candidate > 0) : (candidate -= 1) {
            if (candidate * candidate <= value) break :blk candidate;
        }
        break :blk 0;
    };

    var root = try BigInt.initSet(allocator, integer_digit);
    defer root.deinit();

    var square = try BigInt.initSet(allocator, integer_digit * integer_digit);
    defer square.deinit();

    var tmp_remainder = try subBig(allocator, &remainder, &square);
    remainder.deinit();
    remainder = tmp_remainder;

    var digit_sum: u32 = integer_digit;
    var produced: usize = 1;
    while (produced < digits) : (produced += 1) {
        const scaled_remainder = try mulScalarBig(allocator, &remainder, 100);
        remainder.deinit();
        remainder = scaled_remainder;

        var base = try mulScalarBig(allocator, &root, 20);
        defer base.deinit();

        var next_digit: u32 = 9;
        while (true) {
            var candidate_base = try addScalarBig(allocator, &base, next_digit);
            defer candidate_base.deinit();

            var product = try mulScalarBig(allocator, &candidate_base, next_digit);
            defer product.deinit();

            if (cmpSmall(&product, &remainder) != .gt) {
                var next_root = try mulScalarBig(allocator, &root, 10);
                defer next_root.deinit();

                const updated_root = try addScalarBig(allocator, &next_root, next_digit);
                root.deinit();
                root = updated_root;

                tmp_remainder = try subBig(allocator, &remainder, &product);
                remainder.deinit();
                remainder = tmp_remainder;

                digit_sum += next_digit;
                break;
            }

            if (next_digit == 0) break;
            next_digit -= 1;
        }
    }

    return digit_sum;
}

/// Returns the total of the first `digits` decimal digits for irrational square roots
/// of all integers in `[2, upper_bound)`.
/// Time complexity: O(upper_bound * digits * bigint_digits)
/// Space complexity: O(bigint_digits)
pub fn solution(allocator: Allocator, upper_bound: u32, digits: usize) !u32 {
    if (upper_bound <= 2 or digits == 0) return 0;

    var total: u32 = 0;
    var value: u32 = 2;
    while (value < upper_bound) : (value += 1) {
        total += try digitSumSquareRoot(allocator, value, digits);
    }
    return total;
}

test "problem 080: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u32, 12), try solution(alloc, 3, 5));
    try testing.expectEqual(@as(u32, 239), try solution(alloc, 10, 10));
    try testing.expectEqual(@as(u32, 40886), try solution(alloc, 100, 100));
}

test "problem 080: per-number digit sums and perfect squares" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u32, 12), try digitSumSquareRoot(alloc, 2, 5));
    try testing.expectEqual(@as(u32, 29), try digitSumSquareRoot(alloc, 2, 10));
    try testing.expectEqual(@as(u32, 49), try digitSumSquareRoot(alloc, 5, 10));
    try testing.expectEqual(@as(u32, 0), try digitSumSquareRoot(alloc, 4, 100));
    try testing.expectEqual(@as(u32, 0), try solution(alloc, 100, 0));
}
