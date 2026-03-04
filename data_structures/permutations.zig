//! Permutations - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/arrays/permutations.py

const std = @import("std");
const testing = std.testing;

/// Frees nested permutation output allocated by `permuteRecursive` or `permuteBacktrack`.
pub fn freePermutations(allocator: std.mem.Allocator, permutations: [][]i64) void {
    for (permutations) |perm| {
        allocator.free(perm);
    }
    allocator.free(permutations);
}

fn permuteRecursiveInternal(allocator: std.mem.Allocator, nums: []const i64) ![][]i64 {
    if (nums.len == 0) {
        const outer = try allocator.alloc([]i64, 1);
        outer[0] = try allocator.alloc(i64, 0);
        return outer;
    }

    var result = std.ArrayListUnmanaged([]i64){};
    errdefer {
        for (result.items) |perm| allocator.free(perm);
        result.deinit(allocator);
    }

    var shift: usize = 0;
    while (shift < nums.len) : (shift += 1) {
        const fixed = nums[shift];

        const tail = try allocator.alloc(i64, nums.len - 1);
        defer allocator.free(tail);

        var j: usize = 1;
        while (j < nums.len) : (j += 1) {
            tail[j - 1] = nums[(shift + j) % nums.len];
        }

        const sub_permutations = try permuteRecursiveInternal(allocator, tail);
        defer freePermutations(allocator, sub_permutations);

        for (sub_permutations) |perm| {
            const extended = try allocator.alloc(i64, perm.len + 1);
            @memcpy(extended[0..perm.len], perm);
            extended[perm.len] = fixed;
            try result.append(allocator, extended);
        }
    }

    return result.toOwnedSlice(allocator);
}

/// Returns all permutations using recursive-rotation strategy.
/// Equivalent behavior to Python `permute_recursive`.
/// Time complexity: O(n! * n), Space complexity: O(n! * n)
pub fn permuteRecursive(allocator: std.mem.Allocator, nums: []const i64) ![][]i64 {
    return permuteRecursiveInternal(allocator, nums);
}

fn backtrack(allocator: std.mem.Allocator, nums: []i64, start: usize, output: *std.ArrayListUnmanaged([]i64)) !void {
    if (nums.len == 0) return;

    if (start == nums.len - 1) {
        const snapshot = try allocator.alloc(i64, nums.len);
        @memcpy(snapshot, nums);
        try output.append(allocator, snapshot);
        return;
    }

    var i: usize = start;
    while (i < nums.len) : (i += 1) {
        std.mem.swap(i64, &nums[start], &nums[i]);
        try backtrack(allocator, nums, start + 1, output);
        std.mem.swap(i64, &nums[start], &nums[i]);
    }
}

/// Returns all permutations using in-place backtracking.
/// Equivalent behavior to Python `permute_backtrack`.
/// Time complexity: O(n! * n), Space complexity: O(n! * n)
pub fn permuteBacktrack(allocator: std.mem.Allocator, nums: []const i64) ![][]i64 {
    var output = std.ArrayListUnmanaged([]i64){};
    errdefer {
        for (output.items) |perm| allocator.free(perm);
        output.deinit(allocator);
    }

    if (nums.len == 0) return output.toOwnedSlice(allocator);

    const working = try allocator.alloc(i64, nums.len);
    defer allocator.free(working);
    @memcpy(working, nums);

    try backtrack(allocator, working, 0, &output);
    return output.toOwnedSlice(allocator);
}

fn expectPermutationListEqual(expected: []const []const i64, actual: [][]i64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, 0..) |exp, i| {
        try testing.expectEqualSlices(i64, exp, actual[i]);
    }
}

fn encodePermutation(perm: []const i64) u64 {
    var code: u64 = 0;
    for (perm) |v| {
        code = code * 16 + @as(u64, @intCast(v));
    }
    return code;
}

test "permutations: python recursive doctest order" {
    const out = try permuteRecursive(testing.allocator, &[_]i64{ 1, 2, 3 });
    defer freePermutations(testing.allocator, out);

    const expected = [_][]const i64{
        &[_]i64{ 3, 2, 1 },
        &[_]i64{ 2, 3, 1 },
        &[_]i64{ 1, 3, 2 },
        &[_]i64{ 3, 1, 2 },
        &[_]i64{ 2, 1, 3 },
        &[_]i64{ 1, 2, 3 },
    };
    try expectPermutationListEqual(expected[0..], out);
}

test "permutations: python backtrack doctest order" {
    const out = try permuteBacktrack(testing.allocator, &[_]i64{ 1, 2, 3 });
    defer freePermutations(testing.allocator, out);

    const expected = [_][]const i64{
        &[_]i64{ 1, 2, 3 },
        &[_]i64{ 1, 3, 2 },
        &[_]i64{ 2, 1, 3 },
        &[_]i64{ 2, 3, 1 },
        &[_]i64{ 3, 2, 1 },
        &[_]i64{ 3, 1, 2 },
    };
    try expectPermutationListEqual(expected[0..], out);
}

test "permutations: boundary behavior" {
    const recursive_empty = try permuteRecursive(testing.allocator, &[_]i64{});
    defer freePermutations(testing.allocator, recursive_empty);
    try testing.expectEqual(@as(usize, 1), recursive_empty.len);
    try testing.expectEqual(@as(usize, 0), recursive_empty[0].len);

    const backtrack_empty = try permuteBacktrack(testing.allocator, &[_]i64{});
    defer freePermutations(testing.allocator, backtrack_empty);
    try testing.expectEqual(@as(usize, 0), backtrack_empty.len);

    const single = try permuteBacktrack(testing.allocator, &[_]i64{42});
    defer freePermutations(testing.allocator, single);
    try testing.expectEqual(@as(usize, 1), single.len);
    try testing.expectEqualSlices(i64, &[_]i64{42}, single[0]);
}

test "permutations: extreme count and uniqueness" {
    const out = try permuteBacktrack(testing.allocator, &[_]i64{ 1, 2, 3, 4, 5, 6, 7 });
    defer freePermutations(testing.allocator, out);

    try testing.expectEqual(@as(usize, 5040), out.len);

    var seen = std.AutoHashMap(u64, void).init(testing.allocator);
    defer seen.deinit();

    for (out) |perm| {
        try testing.expectEqual(@as(usize, 7), perm.len);

        var flags = [_]bool{false} ** 7;
        for (perm) |v| {
            try testing.expect(v >= 1 and v <= 7);
            const idx: usize = @intCast(v - 1);
            flags[idx] = true;
        }
        for (flags) |f| try testing.expect(f);

        const key = encodePermutation(perm);
        try testing.expect(!seen.contains(key));
        try seen.put(key, {});
    }
}
