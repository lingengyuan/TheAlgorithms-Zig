//! Project Euler Problem 14: Longest Collatz Sequence - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_014/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem014Error = error{
    Overflow,
    LimitTooLarge,
    SequenceTooLong,
};

/// Returns the next Collatz value.
pub fn collatzNext(number: u64) Problem014Error!u64 {
    if (number % 2 == 0) return number / 2;

    const max_before_mul = @divFloor(std.math.maxInt(u64) - 1, 3);
    if (number > max_before_mul) return Problem014Error.Overflow;
    return number * 3 + 1;
}

fn collatzLength(start: u64, cache: []u32) Problem014Error!u32 {
    var path: [4096]u64 = undefined;
    var path_len: usize = 0;
    var number = start;
    var known_len: u32 = 0;

    while (true) {
        if (number < cache.len) {
            const idx: usize = @intCast(number);
            if (cache[idx] != 0) {
                known_len = cache[idx];
                break;
            }
        }

        if (path_len >= path.len) return Problem014Error.SequenceTooLong;
        path[path_len] = number;
        path_len += 1;

        number = try collatzNext(number);
    }

    var length = known_len;
    while (path_len > 0) {
        path_len -= 1;
        length += 1;

        const value = path[path_len];
        if (value < cache.len) {
            const idx: usize = @intCast(value);
            if (cache[idx] == 0) {
                cache[idx] = length;
            }
        }
    }

    return length;
}

/// Returns the number in [1, limit) with the longest Collatz chain.
///
/// Time complexity: roughly O(limit * average_chain_length)
/// Space complexity: O(limit)
pub fn solution(limit: u64, allocator: std.mem.Allocator) !u64 {
    if (limit <= 2) return 1;
    if (limit > std.math.maxInt(usize) - 1) return Problem014Error.LimitTooLarge;

    const cache_len: usize = @intCast(limit + 1);
    var cache = try allocator.alloc(u32, cache_len);
    defer allocator.free(cache);
    @memset(cache, 0);
    cache[1] = 1;

    var largest_number: u64 = 1;
    var best_len: u32 = 1;

    var candidate: u64 = 2;
    while (candidate < limit) : (candidate += 1) {
        const len = try collatzLength(candidate, cache);
        if (len > best_len) {
            best_len = len;
            largest_number = candidate;
        }
    }

    return largest_number;
}

test "problem 014: python reference" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 837799), try solution(1_000_000, allocator));
    try testing.expectEqual(@as(u64, 171), try solution(200, allocator));
    try testing.expectEqual(@as(u64, 3711), try solution(5000, allocator));
    try testing.expectEqual(@as(u64, 13255), try solution(15000, allocator));
}

test "problem 014: boundaries and overflow guard" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 1), try solution(0, allocator));
    try testing.expectEqual(@as(u64, 1), try solution(1, allocator));
    try testing.expectEqual(@as(u64, 1), try solution(2, allocator));
    try testing.expectEqual(@as(u64, 2), try solution(3, allocator));

    try testing.expectEqual(@as(u64, 1), try collatzNext(2));
    try testing.expectEqual(@as(u64, 40), try collatzNext(13));

    const overflow_input = @divFloor(std.math.maxInt(u64) - 1, 3) + 1;
    try testing.expectError(Problem014Error.Overflow, collatzNext(overflow_input));
}
