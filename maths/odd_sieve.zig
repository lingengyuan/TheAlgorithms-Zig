//! Odd Sieve - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/odd_sieve.py

const std = @import("std");
const testing = std.testing;

pub const OddSieveError = error{Overflow};

/// Returns all prime numbers strictly less than `num` using an odd-only sieve.
/// Caller owns returned slice.
/// Time complexity: O(n log log n), Space complexity: O(n)
pub fn oddSieve(
    allocator: std.mem.Allocator,
    num: i64,
) (OddSieveError || std.mem.Allocator.Error)![]u64 {
    if (num <= 2) return try allocator.alloc(u64, 0);
    if (num == 3) {
        const out = try allocator.alloc(u64, 1);
        out[0] = 2;
        return out;
    }

    if (@sizeOf(usize) < @sizeOf(i64) and num > @as(i64, @intCast(std.math.maxInt(usize)))) {
        return OddSieveError.Overflow;
    }
    const limit: usize = @intCast(num);

    const half = limit >> 1;
    const sieve_len = @subWithOverflow(half, @as(usize, 1));
    if (sieve_len[1] != 0) return OddSieveError.Overflow;
    const sieve = try allocator.alloc(bool, sieve_len[0]);
    defer allocator.free(sieve);
    @memset(sieve, true);

    var i: usize = 3;
    while (i <= (limit - 1) / i) : (i += 2) {
        const i_idx = @subWithOverflow(i >> 1, @as(usize, 1));
        if (i_idx[1] != 0) return OddSieveError.Overflow;
        if (!sieve[i_idx[0]]) continue;

        const i_squared = @mulWithOverflow(i, i);
        if (i_squared[1] != 0) return OddSieveError.Overflow;
        if (i_squared[0] >= limit) continue;

        const step = @mulWithOverflow(i, @as(usize, 2));
        if (step[1] != 0) return OddSieveError.Overflow;

        var j = i_squared[0];
        while (j < limit) {
            const j_idx = @subWithOverflow(j >> 1, @as(usize, 1));
            if (j_idx[1] != 0) return OddSieveError.Overflow;
            sieve[j_idx[0]] = false;

            const next = @addWithOverflow(j, step[0]);
            if (next[1] != 0) break;
            j = next[0];
        }
    }

    var primes = std.ArrayListUnmanaged(u64){};
    errdefer primes.deinit(allocator);
    try primes.append(allocator, 2);

    for (sieve, 0..) |is_prime, idx| {
        if (!is_prime) continue;
        const doubled = @mulWithOverflow(idx, @as(usize, 2));
        if (doubled[1] != 0) return OddSieveError.Overflow;
        const value = @addWithOverflow(doubled[0], @as(usize, 3));
        if (value[1] != 0) return OddSieveError.Overflow;
        try primes.append(allocator, @intCast(value[0]));
    }

    return try primes.toOwnedSlice(allocator);
}

test "odd sieve: examples from python reference" {
    const alloc = testing.allocator;

    const p2 = try oddSieve(alloc, 2);
    defer alloc.free(p2);
    try testing.expectEqual(@as(usize, 0), p2.len);

    const p3 = try oddSieve(alloc, 3);
    defer alloc.free(p3);
    try testing.expectEqualSlices(u64, &[_]u64{2}, p3);

    const p10 = try oddSieve(alloc, 10);
    defer alloc.free(p10);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 5, 7 }, p10);

    const p20 = try oddSieve(alloc, 20);
    defer alloc.free(p20);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 5, 7, 11, 13, 17, 19 }, p20);
}

test "odd sieve: boundary and extreme cases" {
    const p_neg = try oddSieve(testing.allocator, -100);
    defer testing.allocator.free(p_neg);
    try testing.expectEqual(@as(usize, 0), p_neg.len);

    const p4 = try oddSieve(testing.allocator, 4);
    defer testing.allocator.free(p4);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 3 }, p4);

    const p100k = try oddSieve(testing.allocator, 100_000);
    defer testing.allocator.free(p100k);
    try testing.expectEqual(@as(usize, 9_592), p100k.len);
    try testing.expectEqual(@as(u64, 99_991), p100k[p100k.len - 1]);
}

test "odd sieve: overflow on narrow usize targets" {
    if (@sizeOf(usize) >= @sizeOf(i64)) return;
    const too_big_i128 = @as(i128, @intCast(std.math.maxInt(usize))) + 1;
    const too_big: i64 = @intCast(too_big_i128);
    try testing.expectError(OddSieveError.Overflow, oddSieve(testing.allocator, too_big));
}
