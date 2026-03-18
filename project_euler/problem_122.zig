//! Project Euler Problem 122: Efficient Exponentiation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_122/sol1.py

const std = @import("std");
const testing = std.testing;

fn canReach(chain: []u16, length: usize, goal: u16, depth: usize) bool {
    if (length > depth) return false;

    var index: usize = 0;
    while (index < length) : (index += 1) {
        const next = chain[index] + chain[length - 1];
        if (next == goal) return true;

        chain[length] = next;
        if (canReach(chain, length + 1, goal, depth)) return true;
    }
    return false;
}

/// Returns the sum of the minimum addition-chain lengths for exponents `1..limit`.
/// Time complexity: Exponential backtracking with small limit-specific pruning
/// Space complexity: O(limit)
pub fn solution(limit: u16) u32 {
    if (limit <= 1) return 0;

    var total: u32 = 0;
    var chain: [256]u16 = undefined;
    chain[0] = 1;

    var value: u16 = 2;
    while (value <= limit) : (value += 1) {
        var depth: usize = 0;
        while (true) : (depth += 1) {
            if (canReach(&chain, 1, value, depth)) {
                total += @intCast(depth);
                break;
            }
        }
    }
    return total;
}

test "problem 122: python reference" {
    try testing.expectEqual(@as(u32, 0), solution(1));
    try testing.expectEqual(@as(u32, 1), solution(2));
    try testing.expectEqual(@as(u32, 45), solution(14));
    try testing.expectEqual(@as(u32, 50), solution(15));
    try testing.expectEqual(@as(u32, 1582), solution(200));
}

test "problem 122: recursive reachability edge cases" {
    var chain: [16]u16 = undefined;
    chain[0] = 1;

    try testing.expect(canReach(&chain, 1, 2, 2));
    try testing.expect(!canReach(&chain, 1, 2, 0));
    try testing.expect(canReach(&chain, 1, 15, 5));
    try testing.expect(!canReach(&chain, 1, 15, 4));
}
