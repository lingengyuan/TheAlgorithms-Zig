//! Collatz Sequence - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/collatz_sequence.py

const std = @import("std");
const testing = std.testing;

/// Computes the Collatz sequence starting from n, returning the number of steps to reach 1.
/// Time complexity: O(unknown — Collatz conjecture), Space complexity: O(1)
pub fn collatzSteps(n: u64) u64 {
    if (n == 0) return 0;
    var current = n;
    var steps: u64 = 0;
    while (current != 1) {
        if (current % 2 == 0) {
            current = current / 2;
        } else {
            current = 3 * current + 1;
        }
        steps += 1;
    }
    return steps;
}

/// Generates the full Collatz sequence into a caller-provided buffer.
/// Returns the slice of the buffer that was filled.
pub fn collatzSequence(n: u64, buf: []u64) []u64 {
    if (n == 0) return buf[0..0];
    buf[0] = n;
    var len: usize = 1;
    var current = n;
    while (current != 1) {
        if (current % 2 == 0) {
            current = current / 2;
        } else {
            current = 3 * current + 1;
        }
        buf[len] = current;
        len += 1;
    }
    return buf[0..len];
}

test "collatz steps: known values" {
    try testing.expectEqual(@as(u64, 0), collatzSteps(1));
    try testing.expectEqual(@as(u64, 1), collatzSteps(2));
    try testing.expectEqual(@as(u64, 2), collatzSteps(4));
    try testing.expectEqual(@as(u64, 14), collatzSteps(11));
}

test "collatz sequence: n=4" {
    var buf: [100]u64 = undefined;
    const seq = collatzSequence(4, &buf);
    try testing.expectEqualSlices(u64, &[_]u64{ 4, 2, 1 }, seq);
}

test "collatz sequence: n=11" {
    var buf: [100]u64 = undefined;
    const seq = collatzSequence(11, &buf);
    const expected = [_]u64{ 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1 };
    try testing.expectEqualSlices(u64, &expected, seq);
}

test "collatz sequence: n=1" {
    var buf: [100]u64 = undefined;
    const seq = collatzSequence(1, &buf);
    try testing.expectEqualSlices(u64, &[_]u64{1}, seq);
}
