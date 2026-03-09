//! Project Euler Problem 50: Consecutive Prime Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_050/sol1.py

const std = @import("std");
const testing = std.testing;

const PrimeTable = struct {
    primes: []u32,
    is_prime: []bool,
};

fn freePrimeTable(allocator: std.mem.Allocator, table: PrimeTable) void {
    allocator.free(table.primes);
    allocator.free(table.is_prime);
}

fn buildPrimeTable(allocator: std.mem.Allocator, limit: usize) !PrimeTable {
    const is_prime = try allocator.alloc(bool, limit);
    @memset(is_prime, true);
    if (limit > 0) is_prime[0] = false;
    if (limit > 1) is_prime[1] = false;

    var i: usize = 2;
    while (i * i < limit) : (i += 1) {
        if (!is_prime[i]) continue;
        var multiple = i * i;
        while (multiple < limit) : (multiple += i) {
            is_prime[multiple] = false;
        }
    }

    var primes = std.ArrayListUnmanaged(u32){};
    errdefer primes.deinit(allocator);
    for (0..limit) |n| {
        if (is_prime[n]) try primes.append(allocator, @intCast(n));
    }

    return .{
        .primes = try primes.toOwnedSlice(allocator),
        .is_prime = is_prime,
    };
}

/// Returns all primes below `limit`.
/// Caller owns the returned slice.
///
/// Time complexity: O(limit log log limit)
/// Space complexity: O(limit)
pub fn primeSieve(allocator: std.mem.Allocator, limit: usize) ![]u32 {
    const table = try buildPrimeTable(allocator, limit);
    defer allocator.free(table.is_prime);
    return table.primes;
}

/// Returns the largest prime below `ceiling` that can be written as the sum
/// of the most consecutive primes.
///
/// Time complexity: O(p^2)
/// Space complexity: O(p + ceiling)
pub fn solution(allocator: std.mem.Allocator, ceiling: usize) !u32 {
    if (ceiling <= 2) return 0;

    const table = try buildPrimeTable(allocator, ceiling);
    defer freePrimeTable(allocator, table);

    const prefix = try allocator.alloc(u64, table.primes.len + 1);
    defer allocator.free(prefix);
    prefix[0] = 0;
    for (table.primes, 0..) |prime, idx| {
        prefix[idx + 1] = prefix[idx] + prime;
    }

    var best_len: usize = 0;
    var best_prime: u32 = 0;
    for (0..table.primes.len) |start| {
        var end = start + best_len + 1;
        while (end <= table.primes.len) : (end += 1) {
            const sum = prefix[end] - prefix[start];
            if (sum >= ceiling) break;
            if (table.is_prime[@intCast(sum)]) {
                best_len = end - start;
                best_prime = @intCast(sum);
            }
        }
    }
    return best_prime;
}

test "problem 050: python reference" {
    const allocator = testing.allocator;
    try testing.expectEqual(@as(u32, 499), try solution(allocator, 500));
    try testing.expectEqual(@as(u32, 953), try solution(allocator, 1_000));
    try testing.expectEqual(@as(u32, 9521), try solution(allocator, 10_000));
}

test "problem 050: helper semantics and extremes" {
    const allocator = testing.allocator;

    const primes3 = try primeSieve(allocator, 3);
    defer allocator.free(primes3);
    try testing.expectEqualSlices(u32, &[_]u32{2}, primes3);

    const primes50 = try primeSieve(allocator, 50);
    defer allocator.free(primes50);
    try testing.expectEqualSlices(u32, &[_]u32{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47 }, primes50);

    try testing.expectEqual(@as(u32, 0), try solution(allocator, 2));
    try testing.expectEqual(@as(u32, 2), try solution(allocator, 3));
}
