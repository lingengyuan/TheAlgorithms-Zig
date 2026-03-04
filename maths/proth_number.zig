//! Proth Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/special_numbers/proth_number.py

const std = @import("std");
const testing = std.testing;

pub const ProthError = error{ InvalidInput, Overflow };

/// Returns the nth Proth number (1-indexed).
/// Time complexity: O(n), Space complexity: O(n)
pub fn proth(
    allocator: std.mem.Allocator,
    number: i64,
) (ProthError || std.mem.Allocator.Error)!u128 {
    if (number < 1) return ProthError.InvalidInput;
    if (number == 1) return 3;
    if (number == 2) return 5;

    if (@sizeOf(usize) < @sizeOf(i64) and number > @as(i64, @intCast(std.math.maxInt(usize)))) {
        return ProthError.Overflow;
    }
    const target: usize = @intCast(number);

    var values = std.ArrayListUnmanaged(u128){};
    defer values.deinit(allocator);

    try values.append(allocator, 3);
    try values.append(allocator, 5);

    var proth_index: usize = 2;
    var increment: usize = 3;
    var block: u32 = 1;

    while (values.items.len < target) : (block += 1) {
        if (block == std.math.maxInt(u32)) return ProthError.Overflow;
        const step = try pow2(block + 1);

        var generated: usize = 0;
        while (generated < increment and values.items.len < target) : (generated += 1) {
            const prev_idx = proth_index - 1;
            if (prev_idx >= values.items.len) return ProthError.Overflow;

            const prev = values.items[prev_idx];
            const add = @addWithOverflow(prev, step);
            if (add[1] != 0) return ProthError.Overflow;
            try values.append(allocator, add[0]);
            proth_index += 1;
        }

        const next_increment = @mulWithOverflow(increment, @as(usize, 2));
        if (next_increment[1] != 0) return ProthError.Overflow;
        increment = next_increment[0];
    }

    return values.items[target - 1];
}

/// Returns true when `number` is a Proth number.
/// Time complexity: O(log n), Space complexity: O(1)
pub fn isProthNumber(number: i64) ProthError!bool {
    if (number <= 0) return ProthError.InvalidInput;
    if (number == 1) return false;

    var n: u128 = @intCast(number - 1);
    var exponent: u7 = 0;
    while (n % 2 == 0) {
        exponent += 1;
        n /= 2;
    }
    const bound: u128 = @as(u128, 1) << exponent;
    return n < bound;
}

fn pow2(exp: u32) ProthError!u128 {
    if (exp >= 128) return ProthError.Overflow;
    const shift: u7 = @intCast(exp);
    return @as(u128, 1) << shift;
}

test "proth number: python reference examples" {
    try testing.expectEqual(@as(u128, 25), try proth(testing.allocator, 6));
    try testing.expectError(ProthError.InvalidInput, proth(testing.allocator, 0));
    try testing.expectError(ProthError.InvalidInput, proth(testing.allocator, -1));

    try testing.expect(!(try isProthNumber(1)));
    try testing.expect(!(try isProthNumber(2)));
    try testing.expect(try isProthNumber(3));
    try testing.expect(!(try isProthNumber(4)));
    try testing.expect(try isProthNumber(5));
    try testing.expect(!(try isProthNumber(34)));
    try testing.expectError(ProthError.InvalidInput, isProthNumber(-1));
}

test "proth number: sequence prefix and larger index" {
    const expected = [_]u128{ 3, 5, 9, 13, 17, 25, 33, 41, 49, 57 };
    var i: usize = 0;
    while (i < expected.len) : (i += 1) {
        try testing.expectEqual(expected[i], try proth(testing.allocator, @intCast(i + 1)));
    }

    try testing.expectEqual(@as(u128, 500_737), try proth(testing.allocator, 1_000));
    try testing.expect(!(try isProthNumber(std.math.maxInt(i64))));
}
