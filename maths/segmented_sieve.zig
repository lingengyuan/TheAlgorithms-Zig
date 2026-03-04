//! Segmented Sieve - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/segmented_sieve.py

const std = @import("std");
const testing = std.testing;

pub const SegmentedSieveError = error{ InvalidInput, Overflow };

/// Returns all prime numbers up to and including `n`.
/// For n <= 0, returns `error.InvalidInput`.
/// Caller owns returned slice.
/// Time complexity: O(n log log n), Space complexity: O(sqrt(n))
pub fn segmentedSieve(
    allocator: std.mem.Allocator,
    n: i64,
) (SegmentedSieveError || std.mem.Allocator.Error)![]u64 {
    if (n <= 0) return SegmentedSieveError.InvalidInput;
    if (@sizeOf(usize) < @sizeOf(i64) and n > @as(i64, @intCast(std.math.maxInt(usize)))) {
        return SegmentedSieveError.Overflow;
    }

    const limit: usize = @intCast(n);
    if (limit < 2) return try allocator.alloc(u64, 0);

    const end = floorSqrt(limit);

    const end_plus_one = @addWithOverflow(end, @as(usize, 1));
    if (end_plus_one[1] != 0) return SegmentedSieveError.Overflow;
    const base_marks = try allocator.alloc(bool, end_plus_one[0]);
    defer allocator.free(base_marks);
    @memset(base_marks, true);
    if (base_marks.len > 0) base_marks[0] = false;
    if (base_marks.len > 1) base_marks[1] = false;

    var base_primes = std.ArrayListUnmanaged(usize){};
    defer base_primes.deinit(allocator);

    var p: usize = 2;
    while (p <= end) : (p += 1) {
        if (!base_marks[p]) continue;
        try base_primes.append(allocator, p);

        const sq = @mulWithOverflow(p, p);
        if (sq[1] != 0) return SegmentedSieveError.Overflow;
        if (sq[0] > end) continue;

        var j = sq[0];
        while (j <= end) {
            base_marks[j] = false;
            const next = @addWithOverflow(j, p);
            if (next[1] != 0) break;
            j = next[0];
        }
    }

    var primes = std.ArrayListUnmanaged(u64){};
    errdefer primes.deinit(allocator);
    for (base_primes.items) |bp| {
        try primes.append(allocator, @intCast(bp));
    }

    const low_init = @addWithOverflow(end, @as(usize, 1));
    if (low_init[1] != 0) return SegmentedSieveError.Overflow;
    var low = low_init[0];

    const twice_end = @mulWithOverflow(end, @as(usize, 2));
    if (twice_end[1] != 0) return SegmentedSieveError.Overflow;
    var high = @min(twice_end[0], limit);

    while (low <= limit) {
        const span = @subWithOverflow(high, low);
        if (span[1] != 0) return SegmentedSieveError.Overflow;
        const seg_len = @addWithOverflow(span[0], @as(usize, 1));
        if (seg_len[1] != 0) return SegmentedSieveError.Overflow;

        const segment = try allocator.alloc(bool, seg_len[0]);
        defer allocator.free(segment);
        @memset(segment, true);

        for (base_primes.items) |prime| {
            var first = (low / prime) * prime;
            if (first < low) {
                const bumped = @addWithOverflow(first, prime);
                if (bumped[1] != 0) continue;
                first = bumped[0];
            }

            var j = first;
            while (j <= high) {
                const idx = @subWithOverflow(j, low);
                if (idx[1] != 0) return SegmentedSieveError.Overflow;
                segment[idx[0]] = false;

                const next = @addWithOverflow(j, prime);
                if (next[1] != 0) break;
                j = next[0];
            }
        }

        for (segment, 0..) |is_prime, idx| {
            if (!is_prime) continue;
            const value = @addWithOverflow(low, idx);
            if (value[1] != 0) return SegmentedSieveError.Overflow;
            if (value[0] >= 2) {
                try primes.append(allocator, @intCast(value[0]));
            }
        }

        const next_low = @addWithOverflow(high, @as(usize, 1));
        if (next_low[1] != 0) break;
        low = next_low[0];
        if (low > limit) break;

        const next_high = @addWithOverflow(high, end);
        high = if (next_high[1] != 0) limit else @min(next_high[0], limit);
    }

    return try primes.toOwnedSlice(allocator);
}

fn floorSqrt(n: usize) usize {
    if (n < 2) return n;
    var x = n;
    var y: usize = @intCast((@as(u128, x) + @as(u128, n) / x) / 2);
    while (y < x) {
        x = y;
        y = @intCast((@as(u128, x) + @as(u128, n) / x) / 2);
    }
    return x;
}

test "segmented sieve: examples from python reference" {
    const alloc = testing.allocator;

    const p8 = try segmentedSieve(alloc, 8);
    defer alloc.free(p8);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 5, 7 }, p8);

    const p27 = try segmentedSieve(alloc, 27);
    defer alloc.free(p27);
    try testing.expectEqualSlices(u64, &[_]u64{ 2, 3, 5, 7, 11, 13, 17, 19, 23 }, p27);
}

test "segmented sieve: invalid and small inputs" {
    try testing.expectError(SegmentedSieveError.InvalidInput, segmentedSieve(testing.allocator, 0));
    try testing.expectError(SegmentedSieveError.InvalidInput, segmentedSieve(testing.allocator, -1));

    const p1 = try segmentedSieve(testing.allocator, 1);
    defer testing.allocator.free(p1);
    try testing.expectEqual(@as(usize, 0), p1.len);

    const p2 = try segmentedSieve(testing.allocator, 2);
    defer testing.allocator.free(p2);
    try testing.expectEqualSlices(u64, &[_]u64{2}, p2);
}

test "segmented sieve: larger boundary coverage" {
    const primes = try segmentedSieve(testing.allocator, 100_000);
    defer testing.allocator.free(primes);
    try testing.expectEqual(@as(usize, 9_592), primes.len);
    try testing.expectEqual(@as(u64, 99_991), primes[primes.len - 1]);
}

test "segmented sieve: overflow on narrow usize targets" {
    if (@sizeOf(usize) >= @sizeOf(i64)) return;
    const too_big_i128 = @as(i128, @intCast(std.math.maxInt(usize))) + 1;
    const too_big: i64 = @intCast(too_big_i128);
    try testing.expectError(SegmentedSieveError.Overflow, segmentedSieve(testing.allocator, too_big));
}
