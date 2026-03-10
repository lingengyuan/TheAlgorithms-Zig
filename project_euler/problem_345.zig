//! Project Euler Problem 345: Matrix Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_345/sol1.py

const std = @import("std");
const testing = std.testing;

const Allocator = std.mem.Allocator;

const matrix_1 = [_][]const u8{
    "7 53 183 439 863",
    "497 383 563 79 973",
    "287 63 343 169 583",
    "627 343 773 959 943",
    "767 473 103 699 303",
};

const matrix_2 = [_][]const u8{
    "7 53 183 439 863 497 383 563 79 973 287 63 343 169 583",
    "627 343 773 959 943 767 473 103 699 303 957 703 583 639 913",
    "447 283 463 29 23 487 463 993 119 883 327 493 423 159 743",
    "217 623 3 399 853 407 103 983 89 463 290 516 212 462 350",
    "960 376 682 962 300 780 486 502 912 800 250 346 172 812 350",
    "870 456 192 162 593 473 915 45 989 873 823 965 425 329 803",
    "973 965 905 919 133 673 665 235 509 613 673 815 165 992 326",
    "322 148 972 962 286 255 941 541 265 323 925 281 601 95 973",
    "445 721 11 525 473 65 511 164 138 672 18 428 154 448 848",
    "414 456 310 312 798 104 566 520 302 248 694 976 430 392 198",
    "184 829 373 181 631 101 969 613 840 740 778 458 284 760 390",
    "821 461 843 513 17 901 711 993 293 157 274 94 192 156 574",
    "34 124 4 878 450 476 712 914 838 669 875 299 823 329 699",
    "815 559 813 459 522 788 168 586 966 232 308 833 251 631 107",
    "813 883 451 509 615 77 281 613 459 205 380 274 302 35 805",
};

fn parseMatrix(allocator: Allocator, lines: []const []const u8) ![]u32 {
    const n = lines.len;
    var matrix = try allocator.alloc(u32, n * n);
    errdefer allocator.free(matrix);

    for (lines, 0..) |line, row| {
        var it = std.mem.tokenizeScalar(u8, line, ' ');
        var col: usize = 0;
        while (it.next()) |token| : (col += 1) {
            matrix[row * n + col] = try std.fmt.parseInt(u32, token, 10);
        }
        if (col != n) return error.InvalidMatrix;
    }

    return matrix;
}

fn bestSum(matrix: []const u32, n: usize, row: usize, mask: u32, memo: []?u32) u32 {
    if (row == n) return 0;
    if (memo[mask]) |cached| return cached;

    var best: u32 = 0;
    var col: usize = 0;
    while (col < n) : (col += 1) {
        const bit: u32 = @as(u32, 1) << @intCast(col);
        if ((mask & bit) == 0) continue;
        const candidate = matrix[row * n + col] + bestSum(matrix, n, row + 1, mask ^ bit, memo);
        best = @max(best, candidate);
    }

    memo[mask] = best;
    return best;
}

/// Returns the maximum sum of matrix entries with no shared row or column.
/// Time complexity: O(n * 2^n)
/// Space complexity: O(2^n)
pub fn solution(allocator: Allocator, lines: []const []const u8) !u32 {
    const n = lines.len;
    if (n == 0) return 0;
    if (n > 31) return error.UnsupportedMatrix;

    const matrix = try parseMatrix(allocator, lines);
    defer allocator.free(matrix);

    const memo = try allocator.alloc(?u32, @as(usize, 1) << @intCast(n));
    defer allocator.free(memo);
    @memset(memo, null);

    const full_mask = (@as(u32, 1) << @intCast(n)) - 1;
    return bestSum(matrix, n, 0, full_mask, memo);
}

test "problem 345: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u32, 5), try solution(alloc, &[_][]const u8{ "1 2", "3 4" }));
    try testing.expectEqual(@as(u32, 3315), try solution(alloc, &matrix_1));
    try testing.expectEqual(@as(u32, 13938), try solution(alloc, &matrix_2));
}

test "problem 345: edge cases" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u32, 0), try solution(alloc, &[_][]const u8{}));
    try testing.expectError(error.InvalidMatrix, solution(alloc, &[_][]const u8{ "1 2", "3" }));
}
