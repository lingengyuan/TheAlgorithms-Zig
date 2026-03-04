//! Fizz Buzz - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/fizz_buzz.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const FizzBuzzError = error{
    InvalidStartNumber,
    InvalidIterations,
    Overflow,
};

/// Returns the space-separated fizz-buzz sequence string.
/// Behavior matches the Python reference, including loop bound (`number <= iterations`).
pub fn fizzBuzz(
    allocator: Allocator,
    number: i64,
    iterations: i64,
) (FizzBuzzError || Allocator.Error)![]u8 {
    if (number < 1) return FizzBuzzError.InvalidStartNumber;
    if (iterations < 1) return FizzBuzzError.InvalidIterations;

    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    var current = number;
    while (current <= iterations) {
        const by3 = @mod(current, 3) == 0;
        const by5 = @mod(current, 5) == 0;

        if (by3) try out.appendSlice(allocator, "Fizz");
        if (by5) try out.appendSlice(allocator, "Buzz");
        if (!by3 and !by5) {
            var buf: [64]u8 = undefined;
            const text = std.fmt.bufPrint(&buf, "{d}", .{current}) catch unreachable;
            try out.appendSlice(allocator, text);
        }
        try out.append(allocator, ' ');

        if (current == std.math.maxInt(i64)) break;
        current += 1;
    }

    return out.toOwnedSlice(allocator);
}

test "fizz buzz: python examples" {
    const out = try fizzBuzz(testing.allocator, 1, 7);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("1 2 Fizz 4 Buzz Fizz 7 ", out);
}

test "fizz buzz: validation behavior" {
    try testing.expectError(FizzBuzzError.InvalidIterations, fizzBuzz(testing.allocator, 1, 0));
    try testing.expectError(FizzBuzzError.InvalidStartNumber, fizzBuzz(testing.allocator, -5, 5));
    try testing.expectError(FizzBuzzError.InvalidIterations, fizzBuzz(testing.allocator, 10, -5));
}

test "fizz buzz: boundary and extreme output size" {
    const out1 = try fizzBuzz(testing.allocator, 10, 5);
    defer testing.allocator.free(out1);
    try testing.expectEqualStrings("", out1);

    const out2 = try fizzBuzz(testing.allocator, 1, 10000);
    defer testing.allocator.free(out2);
    try testing.expect(out2.len > 0);
    try testing.expect(std.mem.endsWith(u8, out2, "Buzz "));
}
