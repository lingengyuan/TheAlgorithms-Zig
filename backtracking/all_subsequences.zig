//! All Subsequences (Backtracking) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/all_subsequences.py

const std = @import("std");
const testing = std.testing;

pub const Subsequence = struct {
    items: []i64,
};

fn generateRecursive(
    allocator: std.mem.Allocator,
    sequence: []const i64,
    current: *std.ArrayListUnmanaged(i64),
    index: usize,
    results: *std.ArrayListUnmanaged(Subsequence),
) std.mem.Allocator.Error!void {
    if (index == sequence.len) {
        const copy = try allocator.dupe(i64, current.items);
        errdefer allocator.free(copy);
        try results.append(allocator, .{ .items = copy });
        return;
    }

    try generateRecursive(allocator, sequence, current, index + 1, results);

    try current.append(allocator, sequence[index]);
    defer _ = current.pop();
    try generateRecursive(allocator, sequence, current, index + 1, results);
}

/// Generates all subsequences in the same DFS order as Python:
/// exclude current element first, then include it.
///
/// Time complexity: O(2^n * n)
/// Space complexity: O(2^n * n)
pub fn generateAllSubsequences(
    allocator: std.mem.Allocator,
    sequence: []const i64,
) std.mem.Allocator.Error![]Subsequence {
    var results = std.ArrayListUnmanaged(Subsequence){};
    errdefer {
        for (results.items) |sub| allocator.free(sub.items);
        results.deinit(allocator);
    }

    var current = std.ArrayListUnmanaged(i64){};
    defer current.deinit(allocator);

    try generateRecursive(allocator, sequence, &current, 0, &results);
    return results.toOwnedSlice(allocator);
}

pub fn freeSubsequences(allocator: std.mem.Allocator, subsequences: []Subsequence) void {
    for (subsequences) |sub| allocator.free(sub.items);
    allocator.free(subsequences);
}

test "all subsequences: python order examples" {
    const alloc = testing.allocator;

    const seq = [_]i64{ 3, 2, 1 };
    const subsequences = try generateAllSubsequences(alloc, &seq);
    defer freeSubsequences(alloc, subsequences);

    const expected = [_][]const i64{
        &[_]i64{},
        &[_]i64{1},
        &[_]i64{2},
        &[_]i64{ 2, 1 },
        &[_]i64{3},
        &[_]i64{ 3, 1 },
        &[_]i64{ 3, 2 },
        &[_]i64{ 3, 2, 1 },
    };

    try testing.expectEqual(expected.len, subsequences.len);
    for (expected, subsequences) |exp, got| {
        try testing.expectEqualSlices(i64, exp, got.items);
    }
}

test "all subsequences: empty input and single element" {
    const alloc = testing.allocator;

    const empty = try generateAllSubsequences(alloc, &[_]i64{});
    defer freeSubsequences(alloc, empty);
    try testing.expectEqual(@as(usize, 1), empty.len);
    try testing.expectEqual(@as(usize, 0), empty[0].items.len);

    const single = try generateAllSubsequences(alloc, &[_]i64{42});
    defer freeSubsequences(alloc, single);
    try testing.expectEqual(@as(usize, 2), single.len);
    try testing.expectEqualSlices(i64, &[_]i64{}, single[0].items);
    try testing.expectEqualSlices(i64, &[_]i64{42}, single[1].items);
}

test "all subsequences: extreme count growth" {
    const alloc = testing.allocator;

    var seq: [12]i64 = undefined;
    for (0..seq.len) |i| seq[i] = @intCast(i);

    const subsequences = try generateAllSubsequences(alloc, seq[0..]);
    defer freeSubsequences(alloc, subsequences);

    try testing.expectEqual(@as(usize, 4096), subsequences.len);
    try testing.expectEqual(@as(usize, 0), subsequences[0].items.len);
    try testing.expectEqualSlices(i64, seq[0..], subsequences[subsequences.len - 1].items);
}
