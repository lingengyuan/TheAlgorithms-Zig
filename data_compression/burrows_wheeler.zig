//! Burrows-Wheeler Transform - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_compression/burrows_wheeler.py

const std = @import("std");
const testing = std.testing;

pub const BurrowsWheelerError = error{
    EmptyInput,
    InvalidIndex,
};

pub const BwtTransform = struct {
    bwt_string: []u8,
    idx_original_string: usize,
};

fn rotationLessThan(s: []const u8, lhs: usize, rhs: usize) bool {
    if (lhs == rhs) return false;
    const n = s.len;
    var k: usize = 0;
    while (k < n) : (k += 1) {
        const a = s[(lhs + k) % n];
        const b = s[(rhs + k) % n];
        if (a < b) return true;
        if (a > b) return false;
    }
    return false;
}

/// Computes Burrows-Wheeler transform and original-index pointer.
/// Caller owns `bwt_string`.
///
/// Time complexity: O(n^2 log n)
/// Space complexity: O(n)
pub fn bwtTransform(allocator: std.mem.Allocator, s: []const u8) !BwtTransform {
    if (s.len == 0) {
        return BurrowsWheelerError.EmptyInput;
    }

    const n = s.len;
    const indices = try allocator.alloc(usize, n);
    defer allocator.free(indices);

    for (indices, 0..) |*item, i| {
        item.* = i;
    }

    std.sort.heap(usize, indices, s, struct {
        fn lessThan(context: []const u8, lhs: usize, rhs: usize) bool {
            return rotationLessThan(context, lhs, rhs);
        }
    }.lessThan);

    const bwt = try allocator.alloc(u8, n);
    var original_idx: usize = 0;

    for (indices, 0..) |rot_idx, i| {
        bwt[i] = s[(rot_idx + n - 1) % n];
        if (rot_idx == 0) {
            original_idx = i;
        }
    }

    return BwtTransform{ .bwt_string = bwt, .idx_original_string = original_idx };
}

/// Reverses Burrows-Wheeler transform.
/// Caller owns returned string.
///
/// Time complexity: O(n + sigma)
/// Space complexity: O(n + sigma)
pub fn reverseBwt(
    allocator: std.mem.Allocator,
    bwt_string: []const u8,
    idx_original_string: usize,
) ![]u8 {
    if (bwt_string.len == 0) {
        return BurrowsWheelerError.EmptyInput;
    }
    if (idx_original_string >= bwt_string.len) {
        return BurrowsWheelerError.InvalidIndex;
    }

    const n = bwt_string.len;
    var ranks = try allocator.alloc(usize, n);
    defer allocator.free(ranks);

    var counts: [256]usize = [_]usize{0} ** 256;
    for (bwt_string, 0..) |ch, i| {
        ranks[i] = counts[ch];
        counts[ch] += 1;
    }

    var first_occ: [256]usize = [_]usize{0} ** 256;
    var running: usize = 0;
    for (0..256) |i| {
        first_occ[i] = running;
        running += counts[i];
    }

    const out = try allocator.alloc(u8, n);

    var row = idx_original_string;
    var i = n;
    while (i > 0) {
        i -= 1;
        const ch = bwt_string[row];
        out[i] = ch;
        row = first_occ[ch] + ranks[row];
    }

    return out;
}

test "burrows wheeler: python examples" {
    const alloc = testing.allocator;

    const t1 = try bwtTransform(alloc, "^BANANA");
    defer alloc.free(t1.bwt_string);
    try testing.expectEqualStrings("BNN^AAA", t1.bwt_string);
    try testing.expectEqual(@as(usize, 6), t1.idx_original_string);
    const r1 = try reverseBwt(alloc, t1.bwt_string, t1.idx_original_string);
    defer alloc.free(r1);
    try testing.expectEqualStrings("^BANANA", r1);

    const t2 = try bwtTransform(alloc, "a_asa_da_casa");
    defer alloc.free(t2.bwt_string);
    try testing.expectEqualStrings("aaaadss_c__aa", t2.bwt_string);
    try testing.expectEqual(@as(usize, 3), t2.idx_original_string);
    const r2 = try reverseBwt(alloc, t2.bwt_string, t2.idx_original_string);
    defer alloc.free(r2);
    try testing.expectEqualStrings("a_asa_da_casa", r2);

    const t3 = try bwtTransform(alloc, "panamabanana");
    defer alloc.free(t3.bwt_string);
    try testing.expectEqualStrings("mnpbnnaaaaaa", t3.bwt_string);
    try testing.expectEqual(@as(usize, 11), t3.idx_original_string);
    const r3 = try reverseBwt(alloc, t3.bwt_string, t3.idx_original_string);
    defer alloc.free(r3);
    try testing.expectEqualStrings("panamabanana", r3);
}

test "burrows wheeler: validation and extreme values" {
    const alloc = testing.allocator;

    try testing.expectError(BurrowsWheelerError.EmptyInput, bwtTransform(alloc, ""));
    try testing.expectError(BurrowsWheelerError.EmptyInput, reverseBwt(alloc, "", 0));
    try testing.expectError(BurrowsWheelerError.InvalidIndex, reverseBwt(alloc, "abc", 3));

    const long = "abcd" ** 500;
    const transformed = try bwtTransform(alloc, long);
    defer alloc.free(transformed.bwt_string);
    const restored = try reverseBwt(alloc, transformed.bwt_string, transformed.idx_original_string);
    defer alloc.free(restored);
    try testing.expectEqualStrings(long, restored);
}
