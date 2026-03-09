//! Project Euler Problem 47: Distinct Prime Factors - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_047/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem047Error = error{InvalidSpan};

/// Returns the unique prime factors of `n` in ascending order.
/// Caller owns the returned slice.
///
/// Time complexity: O(sqrt(n))
/// Space complexity: O(log n)
pub fn uniquePrimeFactors(allocator: std.mem.Allocator, n: u64) ![]u64 {
    var remaining = n;
    var factors = std.ArrayListUnmanaged(u64){};
    defer factors.deinit(allocator);

    var factor: u64 = 2;
    while (factor * factor <= remaining) : (factor += if (factor == 2) 1 else 2) {
        if (remaining % factor != 0) continue;
        try factors.append(allocator, factor);
        while (remaining % factor == 0) remaining /= factor;
    }
    if (remaining > 1) try factors.append(allocator, remaining);
    return factors.toOwnedSlice(allocator);
}

fn uniquePrimeFactorCount(n: u64) u8 {
    var remaining = n;
    var count: u8 = 0;

    var factor: u64 = 2;
    while (factor * factor <= remaining) : (factor += if (factor == 2) 1 else 2) {
        if (remaining % factor != 0) continue;
        count += 1;
        while (remaining % factor == 0) remaining /= factor;
    }
    if (remaining > 1) count += 1;
    return count;
}

/// Returns true if all elements in `values` are equal.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn equality(values: []const usize) bool {
    if (values.len <= 1) return true;
    const first = values[0];
    for (values[1..]) |value| {
        if (value != first) return false;
    }
    return true;
}

/// Returns the first run of `span` consecutive integers having `span`
/// distinct prime factors each.
/// Caller owns the returned slice.
///
/// Time complexity: roughly O(answer * span * sqrt(answer))
/// Space complexity: O(span)
pub fn run(allocator: std.mem.Allocator, span: usize) (Problem047Error || std.mem.Allocator.Error)![]u64 {
    if (span == 0) return error.InvalidSpan;

    var base: u64 = 2;
    while (true) : (base += 1) {
        var matched = true;
        for (0..span) |offset| {
            if (uniquePrimeFactorCount(base + offset) != span) {
                matched = false;
                break;
            }
        }
        if (!matched) continue;

        const out = try allocator.alloc(u64, span);
        for (0..span) |offset| out[offset] = base + offset;
        return out;
    }
}

/// Returns the first integer of the first run matching the Python reference semantics.
pub fn solution(allocator: std.mem.Allocator, span: usize) (Problem047Error || std.mem.Allocator.Error)!u64 {
    const values = try run(allocator, span);
    defer allocator.free(values);
    return values[0];
}

test "problem 047: python reference" {
    const allocator = testing.allocator;

    const span3 = try run(allocator, 3);
    defer allocator.free(span3);
    try testing.expectEqualSlices(u64, &[_]u64{ 644, 645, 646 }, span3);

    try testing.expectEqual(@as(u64, 134_043), try solution(allocator, 4));
}

test "problem 047: helper semantics and extremes" {
    const allocator = testing.allocator;

    const factors14 = try uniquePrimeFactors(allocator, 14);
    defer allocator.free(factors14);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 7 }, factors14);

    const factors644 = try uniquePrimeFactors(allocator, 644);
    defer allocator.free(factors644);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 7, 23 }, factors644);

    const factors646 = try uniquePrimeFactors(allocator, 646);
    defer allocator.free(factors646);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 17, 19 }, factors646);

    const factors1 = try uniquePrimeFactors(allocator, 1);
    defer allocator.free(factors1);
    try testing.expectEqual(@as(usize, 0), factors1.len);

    try testing.expect(!equality(&[_]usize{ 1, 2, 3, 4 }));
    try testing.expect(equality(&[_]usize{ 2, 2, 2, 2 }));
    try testing.expect(!equality(&[_]usize{ 1, 2, 3, 2, 1 }));
    try testing.expect(equality(&[_]usize{}));

    const span1 = try run(allocator, 1);
    defer allocator.free(span1);
    try testing.expectEqualSlices(u64, &[_]u64{2}, span1);

    try testing.expectError(error.InvalidSpan, run(allocator, 0));
}
