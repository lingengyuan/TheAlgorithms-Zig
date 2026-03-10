//! Project Euler Problem 87: Prime Power Triples - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_087/sol1.py

const std = @import("std");
const testing = std.testing;

fn sievePrimes(allocator: std.mem.Allocator, limit: usize) ![]usize {
    var is_prime = try allocator.alloc(bool, limit + 1);
    defer allocator.free(is_prime);
    @memset(is_prime, true);
    if (limit >= 0) is_prime[0] = false;
    if (limit >= 1) is_prime[1] = false;

    var p: usize = 2;
    while (p <= limit / p) : (p += 1) {
        if (!is_prime[p]) continue;
        var multiple = p * p;
        while (multiple <= limit) : (multiple += p) is_prime[multiple] = false;
    }

    var primes = std.ArrayListUnmanaged(usize){};
    errdefer primes.deinit(allocator);
    for (is_prime, 0..) |prime, value| {
        if (prime) try primes.append(allocator, value);
    }
    return primes.toOwnedSlice(allocator);
}

/// Returns the number of integers less than `limit` that are expressible as the sum
/// of a prime square, prime cube, and prime fourth power.
/// Time complexity: roughly O(pi(limit^(1/2)) * pi(limit^(1/3)) * pi(limit^(1/4)))
/// Space complexity: O(limit) bits
pub fn solution(allocator: std.mem.Allocator, limit: usize) !usize {
    if (limit <= 28) return 0;

    const prime_square_limit = @as(usize, @intFromFloat(std.math.sqrt(@as(f64, @floatFromInt(limit - 24)))));
    const primes = try sievePrimes(allocator, prime_square_limit);
    defer allocator.free(primes);

    var seen = try std.DynamicBitSetUnmanaged.initEmpty(allocator, limit);
    defer seen.deinit(allocator);

    var count: usize = 0;
    for (primes) |prime1| {
        const square = prime1 * prime1;
        for (primes) |prime2| {
            const cube = prime2 * prime2 * prime2;
            if (square + cube >= limit - 16) break;
            for (primes) |prime3| {
                const fourth = prime3 * prime3 * prime3 * prime3;
                const total = square + cube + fourth;
                if (total >= limit) break;
                if (!seen.isSet(total)) {
                    seen.set(total);
                    count += 1;
                }
            }
        }
    }
    return count;
}

test "problem 087: python reference" {
    try testing.expectEqual(@as(usize, 4), try solution(testing.allocator, 50));
    try testing.expectEqual(@as(usize, 1_097_343), try solution(testing.allocator, 50_000_000));
}

test "problem 087: tiny and boundary limits" {
    try testing.expectEqual(@as(usize, 0), try solution(testing.allocator, 0));
    try testing.expectEqual(@as(usize, 0), try solution(testing.allocator, 28));
    try testing.expectEqual(@as(usize, 1), try solution(testing.allocator, 29));
}
