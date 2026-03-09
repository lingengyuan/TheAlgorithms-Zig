//! Project Euler Problem 24: Lexicographic Permutations - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_024/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem024Error = error{
    InvalidRank,
    DuplicateDigit,
    Overflow,
    OutOfMemory,
};

fn factorial(value: usize) Problem024Error!u128 {
    var result: u128 = 1;
    var i: usize = 2;
    while (i <= value) : (i += 1) {
        const mul = @mulWithOverflow(result, i);
        if (mul[1] != 0) return Problem024Error.Overflow;
        result = mul[0];
    }
    return result;
}

/// Returns the 1-indexed `rank`th lexicographic permutation of `digits`.
/// Caller owns returned memory and must free it.
///
/// Time complexity: O(n^2)
/// Space complexity: O(n)
pub fn nthLexicographicPermutation(digits: []const u8, rank: u64, allocator: std.mem.Allocator) Problem024Error![]u8 {
    if (rank == 0) return Problem024Error.InvalidRank;

    var seen = [_]bool{false} ** 256;
    for (digits) |digit| {
        if (seen[digit]) return Problem024Error.DuplicateDigit;
        seen[digit] = true;
    }

    const total_perms = try factorial(digits.len);
    if (@as(u128, rank) > total_perms) return Problem024Error.InvalidRank;

    var available = std.ArrayListUnmanaged(u8){};
    defer available.deinit(allocator);
    try available.appendSlice(allocator, digits);
    std.mem.sort(u8, available.items, {}, std.sort.asc(u8));

    var output = try allocator.alloc(u8, digits.len);
    errdefer allocator.free(output);

    var remaining_rank: u128 = rank - 1;

    var out_idx: usize = 0;
    while (out_idx < output.len) : (out_idx += 1) {
        const step = try factorial(available.items.len - 1);
        const choice_u128 = @divFloor(remaining_rank, step);
        const choice: usize = @intCast(choice_u128);
        remaining_rank = @mod(remaining_rank, step);

        output[out_idx] = available.items[choice];

        var i = choice;
        while (i + 1 < available.items.len) : (i += 1) {
            available.items[i] = available.items[i + 1];
        }
        available.items.len -= 1;
    }

    return output;
}

/// Euler problem default solution (millionth permutation of 0..9).
pub fn solution(allocator: std.mem.Allocator) Problem024Error![10]u8 {
    const perm = try nthLexicographicPermutation("0123456789", 1_000_000, allocator);
    defer allocator.free(perm);

    var out: [10]u8 = undefined;
    @memcpy(&out, perm);
    return out;
}

test "problem 024: python reference" {
    const result = try solution(testing.allocator);
    try testing.expectEqualStrings("2783915460", result[0..]);
}

test "problem 024: known permutations and boundaries" {
    const allocator = testing.allocator;

    const p1 = try nthLexicographicPermutation("012", 1, allocator);
    defer allocator.free(p1);
    try testing.expectEqualStrings("012", p1);

    const p2 = try nthLexicographicPermutation("012", 2, allocator);
    defer allocator.free(p2);
    try testing.expectEqualStrings("021", p2);

    const p6 = try nthLexicographicPermutation("012", 6, allocator);
    defer allocator.free(p6);
    try testing.expectEqualStrings("210", p6);

    const p9 = try nthLexicographicPermutation("0123", 9, allocator);
    defer allocator.free(p9);
    try testing.expectEqualStrings("1203", p9);

    try testing.expectError(Problem024Error.InvalidRank, nthLexicographicPermutation("012", 0, allocator));
    try testing.expectError(Problem024Error.InvalidRank, nthLexicographicPermutation("012", 7, allocator));
    try testing.expectError(Problem024Error.DuplicateDigit, nthLexicographicPermutation("0012", 1, allocator));
}
