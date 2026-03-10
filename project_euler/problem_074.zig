//! Project Euler Problem 74: Digit Factorial Chains - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_074/sol1.py

const std = @import("std");
const testing = std.testing;

const DIGIT_FACTORIALS = [_]u32{ 1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880 };

pub const Problem074Error = error{OutOfMemory};

fn contains(values: []const u32, target: u32) bool {
    for (values) |value| if (value == target) return true;
    return false;
}

/// Returns the sum of the factorials of the decimal digits of `n`.
/// Time complexity: O(digits)
/// Space complexity: O(1)
pub fn sumDigitFactorials(n: u32) u32 {
    if (n == 0) return 1;

    var value = n;
    var total: u32 = 0;
    while (value > 0) : (value /= 10) total += DIGIT_FACTORIALS[value % 10];
    return total;
}

/// Returns the non-repeating chain length using the same semantics as the Python reference.
/// Time complexity: amortized near O(chain_length) with memoization.
/// Space complexity: O(chain_length)
pub fn chainLength(
    allocator: std.mem.Allocator,
    chain_cache: *std.AutoHashMap(u32, u32),
    n: u32,
    previous: *std.ArrayListUnmanaged(u32),
) Problem074Error!u32 {
    if (chain_cache.get(n)) |length| return length;

    const next_number = sumDigitFactorials(n);
    if (contains(previous.items, next_number)) {
        try chain_cache.put(n, 0);
        return 0;
    }

    try previous.append(allocator, n);
    const ret = 1 + try chainLength(allocator, chain_cache, next_number, previous);
    try chain_cache.put(n, ret);
    return ret;
}

/// Returns the number of starting values `< max_start` whose chain length is exactly `num_terms`.
/// Time complexity: O(max_start * average_chain) amortized
/// Space complexity: O(cache_size)
pub fn solution(allocator: std.mem.Allocator, num_terms: u32, max_start: u32) Problem074Error!u32 {
    var chain_cache = std.AutoHashMap(u32, u32).init(allocator);
    defer chain_cache.deinit();

    try chain_cache.put(145, 0);
    try chain_cache.put(169, 3);
    try chain_cache.put(36_301, 3);
    try chain_cache.put(1454, 3);
    try chain_cache.put(871, 2);
    try chain_cache.put(45_361, 2);
    try chain_cache.put(872, 2);
    try chain_cache.put(45_362, 2);

    var count: u32 = 0;
    var start: u32 = 1;
    while (start < max_start) : (start += 1) {
        var previous = std.ArrayListUnmanaged(u32){};
        defer previous.deinit(allocator);
        if (try chainLength(allocator, &chain_cache, start, &previous) == num_terms) count += 1;
    }
    return count;
}

test "problem 074: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u32, 28), try solution(alloc, 10, 1000));
    try testing.expectEqual(@as(u32, 402), try solution(alloc, 60, 1_000_000));
}

test "problem 074: helper semantics and extreme chains" {
    try testing.expectEqual(@as(u32, 145), sumDigitFactorials(145));
    try testing.expectEqual(@as(u32, 871), sumDigitFactorials(45_361));
    try testing.expectEqual(@as(u32, 145), sumDigitFactorials(540));

    var cache = std.AutoHashMap(u32, u32).init(testing.allocator);
    defer cache.deinit();
    try cache.put(145, 0);
    try cache.put(169, 3);
    try cache.put(36_301, 3);
    try cache.put(1454, 3);
    try cache.put(871, 2);
    try cache.put(45_361, 2);
    try cache.put(872, 2);
    try cache.put(45_362, 2);

    var previous = std.ArrayListUnmanaged(u32){};
    defer previous.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 11), try chainLength(testing.allocator, &cache, 10_101, &previous));

    previous.clearRetainingCapacity();
    try testing.expectEqual(@as(u32, 20), try chainLength(testing.allocator, &cache, 555, &previous));

    previous.clearRetainingCapacity();
    try testing.expectEqual(@as(u32, 39), try chainLength(testing.allocator, &cache, 178_924, &previous));
}
