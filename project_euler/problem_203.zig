//! Project Euler Problem 203: Squarefree Binomial Coefficients - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_203/sol1.py

const std = @import("std");
const testing = std.testing;

pub fn isSquarefree(number: u64) bool {
    if (number < 4) return true;
    var divisor: u64 = 2;
    var copy = number;
    while (divisor * divisor <= copy) : (divisor += 1) {
        var multiplicity: u8 = 0;
        while (copy % divisor == 0) {
            copy /= divisor;
            multiplicity += 1;
            if (multiplicity >= 2) return false;
        }
    }
    return true;
}

/// Returns the sum of distinct squarefree coefficients in the first `n` rows of Pascal's triangle.
/// Time complexity: acceptable direct translation for n <= 51
/// Space complexity: O(number of distinct coefficients)
pub fn solution(allocator: std.mem.Allocator, n: u32) !u64 {
    var unique = std.AutoHashMapUnmanaged(u64, void){};
    defer unique.deinit(allocator);
    try unique.put(allocator, 1, {});

    var previous = std.ArrayListUnmanaged(u64){};
    defer previous.deinit(allocator);
    try previous.append(allocator, 1);

    var depth: u32 = 2;
    while (depth <= n) : (depth += 1) {
        var current = std.ArrayListUnmanaged(u64){};
        try current.ensureTotalCapacity(allocator, previous.items.len + 1);

        for (0..previous.items.len + 1) |idx| {
            const left = if (idx < previous.items.len) previous.items[idx] else 0;
            const right = if (idx > 0) previous.items[idx - 1] else 0;
            const value = left + right;
            try current.append(allocator, value);
            try unique.put(allocator, value, {});
        }

        previous.deinit(allocator);
        previous = current;
    }

    var sum: u64 = 0;
    var iterator = unique.iterator();
    while (iterator.next()) |entry| {
        if (isSquarefree(entry.key_ptr.*)) sum += entry.key_ptr.*;
    }
    return sum;
}

test "problem 203: python reference" {
    try testing.expectEqual(@as(u64, 1), try solution(testing.allocator, 1));
    try testing.expectEqual(@as(u64, 105), try solution(testing.allocator, 8));
    try testing.expectEqual(@as(u64, 175), try solution(testing.allocator, 9));
    try testing.expectEqual(@as(u64, 34029210557338), try solution(testing.allocator, 51));
}

test "problem 203: squarefree helper" {
    try testing.expect(isSquarefree(1));
    try testing.expect(isSquarefree(21));
    try testing.expect(!isSquarefree(4));
    try testing.expect(!isSquarefree(20));
}
