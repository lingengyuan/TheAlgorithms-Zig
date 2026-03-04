//! Minimum Squares To Represent A Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/minimum_squares_to_represent_a_number.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const MinimumSquaresError = error{
    NegativeInput,
    NumberTooLarge,
    Overflow,
};

/// Returns the minimum number of perfect squares summing to `number`.
/// Matches Python behavior for zero input by returning 1.
/// Time complexity: O(n * sqrt(n)), Space complexity: O(n)
pub fn minimumSquaresToRepresentANumber(
    allocator: Allocator,
    number: i64,
) (MinimumSquaresError || Allocator.Error)!u32 {
    if (number < 0) return MinimumSquaresError.NegativeInput;
    if (number == 0) return 1;
    if (number > std.math.maxInt(usize)) return MinimumSquaresError.NumberTooLarge;

    const n: usize = @intCast(number);
    const n_plus = @addWithOverflow(n, @as(usize, 1));
    if (n_plus[1] != 0) return MinimumSquaresError.Overflow;

    const answers = try allocator.alloc(u32, n_plus[0]);
    defer allocator.free(answers);

    @memset(answers, 0);
    answers[0] = 0;

    for (1..n_plus[0]) |i| {
        var best: u32 = std.math.maxInt(u32);

        var j: usize = 1;
        while (j <= i / j) : (j += 1) {
            const square = j * j;
            const candidate = @addWithOverflow(answers[i - square], @as(u32, 1));
            if (candidate[1] != 0) return MinimumSquaresError.Overflow;
            best = @min(best, candidate[0]);
        }

        answers[i] = best;
    }

    return answers[n];
}

test "minimum squares: python examples" {
    try testing.expectEqual(@as(u32, 1), try minimumSquaresToRepresentANumber(testing.allocator, 25));
    try testing.expectEqual(@as(u32, 2), try minimumSquaresToRepresentANumber(testing.allocator, 37));
    try testing.expectEqual(@as(u32, 3), try minimumSquaresToRepresentANumber(testing.allocator, 21));
    try testing.expectEqual(@as(u32, 2), try minimumSquaresToRepresentANumber(testing.allocator, 58));
}

test "minimum squares: boundary behavior" {
    try testing.expectEqual(@as(u32, 1), try minimumSquaresToRepresentANumber(testing.allocator, 0));
    try testing.expectEqual(@as(u32, 1), try minimumSquaresToRepresentANumber(testing.allocator, 1));
    try testing.expectEqual(@as(u32, 2), try minimumSquaresToRepresentANumber(testing.allocator, 2));
}

test "minimum squares: invalid input" {
    try testing.expectError(MinimumSquaresError.NegativeInput, minimumSquaresToRepresentANumber(testing.allocator, -1));
}

test "minimum squares: extreme theorem bound check" {
    const answer = try minimumSquaresToRepresentANumber(testing.allocator, 9999);
    // By Lagrange's four-square theorem, every positive integer <= 4 squares.
    try testing.expect(answer >= 1 and answer <= 4);
}
