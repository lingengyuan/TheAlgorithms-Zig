//! Suffix Array - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/suffix_array.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const SortCtx = struct {
    rank: []const i64,
    k: usize,
    n: usize,

    fn secondRank(self: @This(), idx: usize) i64 {
        if (idx + self.k < self.n) return self.rank[idx + self.k];
        return -1;
    }

    fn lessThan(self: @This(), a: usize, b: usize) bool {
        const ra = self.rank[a];
        const rb = self.rank[b];
        if (ra != rb) return ra < rb;
        return self.secondRank(a) < self.secondRank(b);
    }
};

/// Constructs suffix array for `text` using doubling algorithm.
/// Returns indices of suffixes in lexicographic order.
/// Caller owns returned slice.
/// Time complexity: O(n log^2 n), space complexity: O(n)
pub fn suffixArray(allocator: Allocator, text: []const u8) ![]usize {
    const n = text.len;
    if (n == 0) return try allocator.alloc(usize, 0);

    const sa = try allocator.alloc(usize, n);
    errdefer allocator.free(sa);
    var rank = try allocator.alloc(i64, n);
    defer allocator.free(rank);
    var tmp = try allocator.alloc(i64, n);
    defer allocator.free(tmp);

    for (0..n) |i| {
        sa[i] = i;
        rank[i] = @intCast(text[i]);
    }

    var k: usize = 1;
    while (k < n) : (k *= 2) {
        const ctx = SortCtx{
            .rank = rank,
            .k = k,
            .n = n,
        };
        std.sort.heap(usize, sa, ctx, SortCtx.lessThan);

        tmp[sa[0]] = 0;
        for (1..n) |i| {
            const prev = sa[i - 1];
            const cur = sa[i];

            const prev_second: i64 = if (prev + k < n) rank[prev + k] else -1;
            const cur_second: i64 = if (cur + k < n) rank[cur + k] else -1;

            const different = rank[prev] != rank[cur] or prev_second != cur_second;
            tmp[cur] = tmp[prev] + if (different) @as(i64, 1) else @as(i64, 0);
        }

        for (0..n) |i| rank[i] = tmp[i];

        if (rank[sa[n - 1]] == @as(i64, @intCast(n - 1))) break;
    }

    return sa;
}

/// Kasai algorithm for LCP array.
/// `lcp[i]` is LCP between suffixes at `sa[i-1]` and `sa[i]`, with `lcp[0]=0`.
/// Caller owns returned slice.
/// Time complexity: O(n)
pub fn lcpArray(allocator: Allocator, text: []const u8, sa: []const usize) ![]usize {
    if (sa.len != text.len) return error.InvalidSuffixArrayLength;
    const n = text.len;
    const lcp = try allocator.alloc(usize, n);
    errdefer allocator.free(lcp);
    @memset(lcp, 0);
    if (n == 0) return lcp;

    const rank = try allocator.alloc(usize, n);
    defer allocator.free(rank);
    for (sa, 0..) |suffix_idx, i| {
        if (suffix_idx >= n) return error.InvalidSuffixArrayIndex;
        rank[suffix_idx] = i;
    }

    var k: usize = 0;
    for (0..n) |i| {
        const r = rank[i];
        if (r == 0) {
            k = 0;
            continue;
        }
        const j = sa[r - 1];
        while (i + k < n and j + k < n and text[i + k] == text[j + k]) {
            k += 1;
        }
        lcp[r] = k;
        if (k > 0) k -= 1;
    }

    return lcp;
}

test "suffix array: banana" {
    const sa = try suffixArray(testing.allocator, "banana");
    defer testing.allocator.free(sa);
    try testing.expectEqualSlices(usize, &[_]usize{ 5, 3, 1, 0, 4, 2 }, sa);
}

test "suffix array: empty and single char" {
    const empty = try suffixArray(testing.allocator, "");
    defer testing.allocator.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const single = try suffixArray(testing.allocator, "x");
    defer testing.allocator.free(single);
    try testing.expectEqualSlices(usize, &[_]usize{0}, single);
}

test "suffix array: all same char" {
    const sa = try suffixArray(testing.allocator, "aaaa");
    defer testing.allocator.free(sa);
    try testing.expectEqualSlices(usize, &[_]usize{ 3, 2, 1, 0 }, sa);
}

test "lcp array: banana" {
    const text = "banana";
    const sa = try suffixArray(testing.allocator, text);
    defer testing.allocator.free(sa);

    const lcp = try lcpArray(testing.allocator, text, sa);
    defer testing.allocator.free(lcp);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 3, 0, 0, 2 }, lcp);
}

test "lcp array: invalid inputs" {
    try testing.expectError(
        error.InvalidSuffixArrayLength,
        lcpArray(testing.allocator, "abc", &[_]usize{ 0, 1 }),
    );
    try testing.expectError(
        error.InvalidSuffixArrayIndex,
        lcpArray(testing.allocator, "abc", &[_]usize{ 0, 1, 3 }),
    );
}

test "suffix array: extreme repeated text" {
    var buf: [2048]u8 = undefined;
    @memset(&buf, 'a');
    const sa = try suffixArray(testing.allocator, &buf);
    defer testing.allocator.free(sa);

    // In identical-character text, lexicographic order is by suffix length.
    try testing.expectEqual(@as(usize, buf.len - 1), sa[0]);
    try testing.expectEqual(@as(usize, 0), sa[sa.len - 1]);
}
