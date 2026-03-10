//! Subset Generation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/subset_generation.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const SubsetGenerationError = Allocator.Error;
pub const IntCombination = []const i64;

/// Returns all `n`-element combinations from `elements` in lexicographic-by-index order.
/// Time complexity: O(C(r, n) * n), Space complexity: O(C(r, n) * n)
pub fn subsetCombinations(
    allocator: Allocator,
    elements: []const i64,
    n: usize,
) SubsetGenerationError![]IntCombination {
    if (n > elements.len) return allocator.alloc(IntCombination, 0);

    var current = std.ArrayListUnmanaged(i64){};
    defer current.deinit(allocator);

    var results = std.ArrayListUnmanaged(IntCombination){};
    errdefer freeSubsetCombinations(allocator, results.items);

    try dfsSubsetCombinations(allocator, elements, n, 0, &current, &results);
    return results.toOwnedSlice(allocator);
}

fn dfsSubsetCombinations(
    allocator: Allocator,
    elements: []const i64,
    n: usize,
    start: usize,
    current: *std.ArrayListUnmanaged(i64),
    results: *std.ArrayListUnmanaged(IntCombination),
) SubsetGenerationError!void {
    if (current.items.len == n) {
        const combination = try allocator.alloc(i64, current.items.len);
        @memcpy(combination, current.items);
        try results.append(allocator, combination);
        return;
    }

    var index = start;
    while (index < elements.len) : (index += 1) {
        try current.append(allocator, elements[index]);
        defer _ = current.pop();
        try dfsSubsetCombinations(allocator, elements, n, index + 1, current, results);
    }
}

pub fn freeSubsetCombinations(allocator: Allocator, combinations: []const IntCombination) void {
    for (combinations) |combination| allocator.free(combination);
    allocator.free(combinations);
}

test "subset generation: python samples" {
    const elements = [_]i64{ 10, 20, 30, 40 };
    const combinations = try subsetCombinations(testing.allocator, &elements, 2);
    defer freeSubsetCombinations(testing.allocator, combinations);

    try testing.expectEqual(@as(usize, 6), combinations.len);
    try testing.expectEqualSlices(i64, &[_]i64{ 10, 20 }, combinations[0]);
    try testing.expectEqualSlices(i64, &[_]i64{ 30, 40 }, combinations[5]);
}

test "subset generation: zero-sized subset" {
    const elements = [_]i64{ 10, 20, 30 };
    const combinations = try subsetCombinations(testing.allocator, &elements, 0);
    defer freeSubsetCombinations(testing.allocator, combinations);

    try testing.expectEqual(@as(usize, 1), combinations.len);
    try testing.expectEqual(@as(usize, 0), combinations[0].len);
}

test "subset generation: out of range request" {
    const elements = [_]i64{ 1, 2, 3 };
    const combinations = try subsetCombinations(testing.allocator, &elements, 5);
    defer freeSubsetCombinations(testing.allocator, combinations);

    try testing.expectEqual(@as(usize, 0), combinations.len);
}

test "subset generation: extreme choose all" {
    const elements = [_]i64{ 6, 7, 8, 9 };
    const combinations = try subsetCombinations(testing.allocator, &elements, 4);
    defer freeSubsetCombinations(testing.allocator, combinations);

    try testing.expectEqual(@as(usize, 1), combinations.len);
    try testing.expectEqualSlices(i64, &elements, combinations[0]);
}
