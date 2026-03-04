//! Abbreviation DP - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/abbreviation.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const AbbreviationError = error{
    Overflow,
};

fn index(cols: usize, row: usize, col: usize) usize {
    return row * cols + col;
}

/// Returns true if `a` can be transformed into `b` by:
/// - capitalizing zero or more lowercase letters in `a`
/// - deleting the remaining lowercase letters
/// Matching follows the Python DP implementation.
/// Time complexity: O(n * m), Space complexity: O(n * m)
pub fn canAbbreviate(
    allocator: Allocator,
    a: []const u8,
    b: []const u8,
) (AbbreviationError || Allocator.Error)!bool {
    const n_plus = @addWithOverflow(a.len, @as(usize, 1));
    if (n_plus[1] != 0) return AbbreviationError.Overflow;
    const m_plus = @addWithOverflow(b.len, @as(usize, 1));
    if (m_plus[1] != 0) return AbbreviationError.Overflow;
    const cells = @mulWithOverflow(n_plus[0], m_plus[0]);
    if (cells[1] != 0) return AbbreviationError.Overflow;

    const dp = try allocator.alloc(bool, cells[0]);
    defer allocator.free(dp);
    @memset(dp, false);

    dp[index(m_plus[0], 0, 0)] = true;

    for (0..a.len) |i| {
        for (0..m_plus[0]) |j| {
            if (!dp[index(m_plus[0], i, j)]) continue;

            if (j < b.len and std.ascii.toUpper(a[i]) == b[j]) {
                dp[index(m_plus[0], i + 1, j + 1)] = true;
            }

            if (std.ascii.isLower(a[i])) {
                dp[index(m_plus[0], i + 1, j)] = true;
            }
        }
    }

    return dp[index(m_plus[0], a.len, b.len)];
}

test "abbreviation: python examples" {
    try testing.expect(try canAbbreviate(testing.allocator, "daBcd", "ABC"));
    try testing.expect(!(try canAbbreviate(testing.allocator, "dBcd", "ABC")));
}

test "abbreviation: boundary behavior" {
    try testing.expect(try canAbbreviate(testing.allocator, "", ""));
    try testing.expect(!(try canAbbreviate(testing.allocator, "", "A")));
    try testing.expect(try canAbbreviate(testing.allocator, "abc", ""));
    try testing.expect(!(try canAbbreviate(testing.allocator, "ABC", "")));
}

test "abbreviation: strict uppercase cannot be deleted" {
    try testing.expect(!(try canAbbreviate(testing.allocator, "ABCD", "AC")));
    try testing.expect(try canAbbreviate(testing.allocator, "abCd", "AC"));
}

test "abbreviation: extreme long mixed string" {
    var source: [3000]u8 = undefined;
    var target: [1500]u8 = undefined;

    for (0..1500) |i| {
        source[2 * i] = 'a';
        source[2 * i + 1] = 'b';
        target[i] = 'B';
    }

    try testing.expect(try canAbbreviate(testing.allocator, &source, &target));
}
