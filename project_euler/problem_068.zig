//! Project Euler Problem 68: Magic 5-gon Ring - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_068/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem068Error = error{ InvalidSide, Impossible };

fn buildSource(gon_side: u8, source: *[10]u8) usize {
    var idx: usize = 0;
    var value: i32 = gon_side + 1;
    while (value >= 1) : (value -= 1) {
        source[idx] = @intCast(value);
        idx += 1;
    }
    value = gon_side + 2;
    while (value <= gon_side * 2) : (value += 1) {
        source[idx] = @intCast(value);
        idx += 1;
    }
    return idx;
}

fn generateGonRing(gon_side: u8, perm: []const u8, result: *[15]u8) usize {
    result[0] = perm[0];
    result[1] = perm[1];
    result[2] = perm[2];
    const extra: usize = if (gon_side < 5) 1 else 2;
    const loop_count: usize = (perm.len + 1) / 3 + extra;
    var i: usize = 1;
    while (i < loop_count) : (i += 1) {
        result[3 * i] = perm[2 * i + 1];
        result[3 * i + 1] = result[3 * i - 1];
        result[3 * i + 2] = if (2 * i + 2 < perm.len) perm[2 * i + 2] else perm[1];
    }
    return @as(usize, gon_side) * 3;
}

fn isMagicGon(numbers: []const u8) Problem068Error!bool {
    if (numbers.len % 3 != 0) return error.InvalidSide;
    var min_outer = numbers[0];
    var idx: usize = 3;
    while (idx < numbers.len) : (idx += 3) min_outer = @min(min_outer, numbers[idx]);
    if (min_outer != numbers[0]) return false;

    const total: u16 = @as(u16, numbers[0]) + numbers[1] + numbers[2];
    idx = 3;
    while (idx < numbers.len) : (idx += 3) {
        const chunk_total: u16 = @as(u16, numbers[idx]) + numbers[idx + 1] + numbers[idx + 2];
        if (chunk_total != total) return false;
    }
    return true;
}

fn concatValue(numbers: []const u8) u64 {
    var value: u64 = 0;
    for (numbers) |num| {
        if (num >= 10) {
            value = value * 100 + num;
        } else {
            value = value * 10 + num;
        }
    }
    return value;
}

fn search(gon_side: u8, source: []const u8, used: *[10]bool, perm: *[10]u8, depth: usize) ?u64 {
    if (depth == source.len) {
        var ring: [15]u8 = undefined;
        const ring_len = generateGonRing(gon_side, perm[0..source.len], &ring);
        if ((isMagicGon(ring[0..ring_len]) catch false)) return concatValue(ring[0..ring_len]);
        return null;
    }

    var i: usize = 0;
    while (i < source.len) : (i += 1) {
        if (used[i]) continue;
        used[i] = true;
        perm[depth] = source[i];
        if (search(gon_side, source, used, perm, depth + 1)) |value| return value;
        used[i] = false;
    }
    return null;
}

/// Returns the maximum concatenated magic n-gon value for `gon_side` in [3, 5].
/// Time complexity: factorial search over 2n labels
/// Space complexity: O(n)
pub fn solution(gon_side: u8) Problem068Error!u64 {
    if (gon_side < 3 or gon_side > 5) return error.InvalidSide;

    var source: [10]u8 = undefined;
    const len = buildSource(gon_side, &source);
    var used: [10]bool = [_]bool{false} ** 10;
    var perm: [10]u8 = undefined;
    return search(gon_side, source[0..len], &used, &perm, 0) orelse error.Impossible;
}

test "problem 068: python reference" {
    try testing.expectEqual(@as(u64, 432621513), try solution(3));
    try testing.expectEqual(@as(u64, 426561813732), try solution(4));
    try testing.expectEqual(@as(u64, 6531031914842725), try solution(5));
}

test "problem 068: invalid side" {
    try testing.expectError(error.InvalidSide, solution(6));
}
