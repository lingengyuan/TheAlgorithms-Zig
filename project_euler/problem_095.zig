//! Project Euler Problem 95: Amicable Chains - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_095/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the smallest member of the longest amicable chain with no element
/// exceeding `max_num`.
/// Time complexity: roughly O(max_num log max_num)
/// Space complexity: O(max_num)
pub fn solution(allocator: std.mem.Allocator, max_num: usize) !usize {
    if (max_num < 2) return 0;

    var sums = try allocator.alloc(usize, max_num + 1);
    defer allocator.free(sums);
    @memset(sums, 1);
    sums[0] = 0;
    if (max_num >= 1) sums[1] = 0;

    var divisor: usize = 2;
    while (divisor <= max_num / 2) : (divisor += 1) {
        var multiple = divisor * 2;
        while (multiple <= max_num) : (multiple += divisor) {
            sums[multiple] += divisor;
        }
    }

    var best_len: usize = 0;
    var best_min: usize = 0;
    var done = try allocator.alloc(bool, max_num + 1);
    defer allocator.free(done);
    @memset(done, false);

    var visit_stamp = try allocator.alloc(u32, max_num + 1);
    defer allocator.free(visit_stamp);
    @memset(visit_stamp, 0);

    var visit_index = try allocator.alloc(u32, max_num + 1);
    defer allocator.free(visit_index);
    @memset(visit_index, 0);

    var stamp: u32 = 0;
    var sequence = std.ArrayListUnmanaged(usize){};
    defer sequence.deinit(allocator);

    var start: usize = 2;
    while (start <= max_num) : (start += 1) {
        if (done[start]) continue;
        stamp += 1;
        sequence.clearRetainingCapacity();

        var current: usize = start;
        while (current > 1 and current <= max_num and !done[current]) {
            if (visit_stamp[current] == stamp) {
                const loop_start = visit_index[current];
                const chain = sequence.items[loop_start..];
                if (chain.len > best_len) {
                    best_len = chain.len;
                    best_min = chain[0];
                    for (chain[1..]) |value| best_min = @min(best_min, value);
                }
                break;
            }

            visit_stamp[current] = stamp;
            visit_index[current] = @intCast(sequence.items.len);
            try sequence.append(allocator, current);
            current = sums[current];
        }

        for (sequence.items) |value| {
            if (value <= max_num) done[value] = true;
        }
    }

    return best_min;
}

test "problem 095: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 6), try solution(alloc, 10));
    try testing.expectEqual(@as(usize, 12_496), try solution(alloc, 200_000));
    try testing.expectEqual(@as(usize, 14_316), try solution(alloc, 1_000_000));
}

test "problem 095: degenerate limits" {
    try testing.expectEqual(@as(usize, 0), try solution(testing.allocator, 0));
    try testing.expectEqual(@as(usize, 0), try solution(testing.allocator, 1));
}
