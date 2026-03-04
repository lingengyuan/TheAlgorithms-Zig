//! Integer Partition Count - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/integer_partition.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const IntegerPartitionError = error{
    InvalidInput,
    Overflow,
};

fn index(cols: usize, row: usize, col: usize) usize {
    return row * cols + col;
}

/// Returns the partition count p(n), i.e. number of ways to write `n`
/// as a sum of positive integers without regard to order.
/// Equivalent to the Python DP approach in `integer_partition.py`.
/// Time complexity: O(n^2), Space complexity: O(n^2)
pub fn integerPartitionCount(
    allocator: Allocator,
    m: i32,
) (IntegerPartitionError || Allocator.Error)!u128 {
    if (m <= 0) return IntegerPartitionError.InvalidInput;

    const n: usize = @intCast(m);
    const rows = @addWithOverflow(n, @as(usize, 1));
    if (rows[1] != 0) return IntegerPartitionError.Overflow;
    const cols: usize = n;
    const cells = @mulWithOverflow(rows[0], cols);
    if (cells[1] != 0) return IntegerPartitionError.Overflow;

    const memo = try allocator.alloc(u128, cells[0]);
    defer allocator.free(memo);
    @memset(memo, 0);

    for (0..rows[0]) |i| {
        memo[index(cols, i, 0)] = 1;
    }

    for (0..rows[0]) |nn| {
        for (1..cols) |k| {
            var cell = memo[index(cols, nn, k - 1)];

            if (nn > k) {
                const addend = memo[index(cols, nn - k - 1, k)];
                const next = @addWithOverflow(cell, addend);
                if (next[1] != 0) return IntegerPartitionError.Overflow;
                cell = next[0];
            }

            memo[index(cols, nn, k)] = cell;
        }
    }

    return memo[index(cols, n, cols - 1)];
}

test "integer partition: python examples" {
    try testing.expectEqual(@as(u128, 7), try integerPartitionCount(testing.allocator, 5));
    try testing.expectEqual(@as(u128, 15), try integerPartitionCount(testing.allocator, 7));
    try testing.expectEqual(@as(u128, 190569292), try integerPartitionCount(testing.allocator, 100));
    try testing.expectEqual(@as(u128, 24061467864032622473692149727991), try integerPartitionCount(testing.allocator, 1000));
}

test "integer partition: boundary and invalid input" {
    try testing.expectEqual(@as(u128, 1), try integerPartitionCount(testing.allocator, 1));
    try testing.expectError(IntegerPartitionError.InvalidInput, integerPartitionCount(testing.allocator, 0));
    try testing.expectError(IntegerPartitionError.InvalidInput, integerPartitionCount(testing.allocator, -7));
}

test "integer partition: extreme arithmetic overflow is detected" {
    try testing.expectError(IntegerPartitionError.Overflow, integerPartitionCount(testing.allocator, 1500));
}
