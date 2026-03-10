//! Project Euler Problem 180: Rational Zeros of a Function of Three Variables - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_180/sol1.py

const std = @import("std");
const testing = std.testing;

const Allocator = std.mem.Allocator;

const Fraction = struct {
    num: u64,
    den: u64,
};

fn gcd128(a: u128, b: u128) u128 {
    var x = a;
    var y = b;
    while (y != 0) {
        const r = x % y;
        x = y;
        y = r;
    }
    return x;
}

fn reduceFraction(num: u128, den: u128) Fraction {
    const divisor = gcd128(num, den);
    return .{
        .num = @intCast(num / divisor),
        .den = @intCast(den / divisor),
    };
}

fn isSquare(value: u128) bool {
    const root = std.math.sqrt(value);
    return root * root == value;
}

fn addThree(x_num: u64, x_den: u64, y_num: u64, y_den: u64, z_num: u64, z_den: u64) Fraction {
    const top = @as(u128, x_num) * y_den * z_den + @as(u128, y_num) * x_den * z_den + @as(u128, z_num) * x_den * y_den;
    const bottom = @as(u128, x_den) * y_den * z_den;
    return reduceFraction(top, bottom);
}

fn packFraction(value: Fraction) u128 {
    return (@as(u128, value.num) << 64) | value.den;
}

fn unpackFraction(value: u128) Fraction {
    return .{
        .num = @intCast(value >> 64),
        .den = @intCast(value & std.math.maxInt(u64)),
    };
}

fn maybeAddFraction(
    unique_s: *std.AutoHashMap(u128, void),
    x_num: u64,
    x_den: u64,
    y_num: u64,
    y_den: u64,
    z_num_raw: u128,
    z_den_raw: u128,
    order: u64,
) !void {
    const z = reduceFraction(z_num_raw, z_den_raw);
    if (!(z.num > 0 and z.num < z.den and z.den <= order)) return;

    const sum_fraction = addThree(x_num, x_den, y_num, y_den, z.num, z.den);
    try unique_s.put(packFraction(sum_fraction), {});
}

/// Returns the sum of the numerator and denominator of the reduced total of all
/// distinct `x + y + z` values for golden triples of the given order.
/// Time complexity: O(order^4)
/// Space complexity: O(unique_sums)
pub fn solution(allocator: Allocator, order: u64) !u128 {
    var unique_s = std.AutoHashMap(u128, void).init(allocator);
    defer unique_s.deinit();

    var x_num: u64 = 1;
    while (x_num <= order) : (x_num += 1) {
        var x_den: u64 = x_num + 1;
        while (x_den <= order) : (x_den += 1) {
            var y_num: u64 = 1;
            while (y_num <= order) : (y_num += 1) {
                var y_den: u64 = y_num + 1;
                while (y_den <= order) : (y_den += 1) {
                    try maybeAddFraction(
                        &unique_s,
                        x_num,
                        x_den,
                        y_num,
                        y_den,
                        @as(u128, x_num) * y_den + @as(u128, x_den) * y_num,
                        @as(u128, x_den) * y_den,
                        order,
                    );

                    const square_sum_num = @as(u128, x_num) * x_num * y_den * y_den + @as(u128, x_den) * x_den * y_num * y_num;
                    const square_sum_den = @as(u128, x_den) * x_den * y_den * y_den;
                    if (isSquare(square_sum_num) and isSquare(square_sum_den)) {
                        try maybeAddFraction(
                            &unique_s,
                            x_num,
                            x_den,
                            y_num,
                            y_den,
                            std.math.sqrt(square_sum_num),
                            std.math.sqrt(square_sum_den),
                            order,
                        );
                    }

                    try maybeAddFraction(
                        &unique_s,
                        x_num,
                        x_den,
                        y_num,
                        y_den,
                        @as(u128, x_num) * y_num,
                        @as(u128, x_den) * y_num + @as(u128, x_num) * y_den,
                        order,
                    );

                    const inverse_square_num = @as(u128, x_num) * x_num * y_num * y_num;
                    const inverse_square_den = @as(u128, x_den) * x_den * y_num * y_num + @as(u128, x_num) * x_num * y_den * y_den;
                    if (isSquare(inverse_square_num) and isSquare(inverse_square_den)) {
                        try maybeAddFraction(
                            &unique_s,
                            x_num,
                            x_den,
                            y_num,
                            y_den,
                            std.math.sqrt(inverse_square_num),
                            std.math.sqrt(inverse_square_den),
                            order,
                        );
                    }
                }
            }
        }
    }

    var total = Fraction{ .num = 0, .den = 1 };
    var iterator = unique_s.iterator();
    while (iterator.next()) |entry| {
        const value = unpackFraction(entry.key_ptr.*);
        total = reduceFraction(@as(u128, total.num) * value.den + @as(u128, value.num) * total.den, @as(u128, total.den) * value.den);
    }

    return @as(u128, total.num) + total.den;
}

test "problem 180: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u128, 296), try solution(alloc, 5));
    try testing.expectEqual(@as(u128, 12519), try solution(alloc, 10));
    try testing.expectEqual(@as(u128, 19408891927), try solution(alloc, 20));
    try testing.expectEqual(@as(u128, 285196020571078987), try solution(alloc, 35));
}

test "problem 180: helper functions" {
    try testing.expect(isSquare(1));
    try testing.expect(!isSquare(1_000_001));
    try testing.expect(isSquare(1_000_000));

    const sum1 = addThree(1, 3, 1, 3, 1, 3);
    try testing.expectEqual(Fraction{ .num = 1, .den = 1 }, sum1);

    const sum2 = addThree(2, 5, 4, 11, 12, 3);
    try testing.expectEqual(Fraction{ .num = 262, .den = 55 }, sum2);
}
