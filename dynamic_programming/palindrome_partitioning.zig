//! Palindrome Partitioning (Minimum Cuts) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/palindrome_partitioning.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Returns the minimum number of cuts needed to partition `text`
/// so that every substring is a palindrome.
/// Time complexity: O(n^2), Space complexity: O(n^2)
pub fn minPalindromeCuts(allocator: Allocator, text: []const u8) !usize {
    const n = text.len;
    if (n == 0) return 0;

    const elem_count = @mulWithOverflow(n, n);
    if (elem_count[1] != 0) return error.Overflow;
    const is_pal = try allocator.alloc(bool, elem_count[0]);
    defer allocator.free(is_pal);
    @memset(is_pal, false);

    var i = n;
    while (i > 0) {
        i -= 1;
        for (i..n) |j| {
            if (text[i] == text[j] and (j <= i + 1 or is_pal[(i + 1) * n + (j - 1)])) {
                is_pal[i * n + j] = true;
            }
        }
    }

    const cuts = try allocator.alloc(usize, n);
    defer allocator.free(cuts);

    for (0..n) |end| {
        if (is_pal[end]) {
            cuts[end] = 0;
            continue;
        }

        var best = end;
        for (0..end) |prev| {
            if (is_pal[(prev + 1) * n + end]) {
                const candidate = cuts[prev] + 1;
                if (candidate < best) best = candidate;
            }
        }
        cuts[end] = best;
    }

    return cuts[n - 1];
}

test "palindrome partitioning: aab -> 1" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 1), try minPalindromeCuts(alloc, "aab"));
}

test "palindrome partitioning: single char -> 0" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 0), try minPalindromeCuts(alloc, "a"));
}

test "palindrome partitioning: no palindrome merge -> n-1" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 2), try minPalindromeCuts(alloc, "abc"));
}

test "palindrome partitioning: already palindrome" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 0), try minPalindromeCuts(alloc, "abba"));
}

test "palindrome partitioning: empty string" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(usize, 0), try minPalindromeCuts(alloc, ""));
}

test "palindrome partitioning: oversize input length returns overflow" {
    const fake_ptr: [*]const u8 = @ptrFromInt(@alignOf(u8));
    const fake_text = fake_ptr[0..std.math.maxInt(usize)];
    try testing.expectError(error.Overflow, minPalindromeCuts(testing.allocator, fake_text));
}
