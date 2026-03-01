//! Subset Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/sum_of_subset.py

const std = @import("std");
const testing = std.testing;

pub const SubsetSumError = error{ NegativeTarget, NegativeElement };

/// Returns whether any subset of `numbers` sums to `target`.
/// This implementation accepts only non-negative inputs.
/// Time complexity: O(n * target), space complexity: O(target)
pub fn isSubsetSum(
    allocator: std.mem.Allocator,
    numbers: []const i64,
    target: i64,
) (SubsetSumError || std.mem.Allocator.Error)!bool {
    if (target < 0) return SubsetSumError.NegativeTarget;
    for (numbers) |value| {
        if (value < 0) return SubsetSumError.NegativeElement;
    }

    const target_u: usize = @intCast(target);
    const dp = try allocator.alloc(bool, target_u + 1);
    defer allocator.free(dp);
    @memset(dp, false);
    dp[0] = true;

    for (numbers) |value_i64| {
        const value: usize = @intCast(value_i64);
        if (value > target_u) continue;

        var s = target_u;
        while (true) {
            if (s >= value and dp[s - value]) dp[s] = true;
            if (s == 0) break;
            s -= 1;
        }

        if (dp[target_u]) return true;
    }

    return dp[target_u];
}

test "subset sum: python examples" {
    const alloc = testing.allocator;
    try testing.expect(!(try isSubsetSum(alloc, &[_]i64{ 2, 4, 6, 8 }, 5)));
    try testing.expect(try isSubsetSum(alloc, &[_]i64{ 2, 4, 6, 8 }, 14));
}

test "subset sum: empty and zero target" {
    const alloc = testing.allocator;
    try testing.expect(try isSubsetSum(alloc, &[_]i64{}, 0));
    try testing.expect(!(try isSubsetSum(alloc, &[_]i64{}, 1)));
}

test "subset sum: repeated values" {
    const alloc = testing.allocator;
    try testing.expect(try isSubsetSum(alloc, &[_]i64{ 3, 3, 3, 3 }, 6));
    try testing.expect(!(try isSubsetSum(alloc, &[_]i64{ 3, 3, 3, 3 }, 5)));
}

test "subset sum: target too large" {
    const alloc = testing.allocator;
    try testing.expect(!(try isSubsetSum(alloc, &[_]i64{ 1, 2, 3, 4, 5 }, 1000)));
}

test "subset sum: reject negative input" {
    const alloc = testing.allocator;
    try testing.expectError(SubsetSumError.NegativeTarget, isSubsetSum(alloc, &[_]i64{ 1, 2, 3 }, -1));
    try testing.expectError(SubsetSumError.NegativeElement, isSubsetSum(alloc, &[_]i64{ 1, -2, 3 }, 3));
}

test "subset sum: extreme input size" {
    const alloc = testing.allocator;

    var values: [256]i64 = undefined;
    for (0..values.len) |i| values[i] = @intCast((i % 9) + 1);

    try testing.expect(try isSubsetSum(alloc, &values, 500));
}

test "subset sum: fuzz matches brute force for small arrays" {
    return testing.fuzz({}, fuzzSubsetSumMatchesBruteForce, .{});
}

fn fuzzSubsetSumMatchesBruteForce(context: void, input: []const u8) anyerror!void {
    _ = context;

    var values: [12]i64 = [_]i64{0} ** 12;
    const n: usize = if (input.len == 0) 0 else @as(usize, input[0] % values.len);

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const fallback: u8 = @intCast((i * 31 + 7) % 251);
        const b: u8 = if (i + 1 < input.len) input[i + 1] else fallback;
        values[i] = @as(i64, @intCast(b % 16));
    }

    const target_seed: u8 = if (n + 1 < input.len) input[n + 1] else @intCast((n * 13 + 5) % 251);
    const target: i64 = @intCast(target_seed % 96);

    const got = try isSubsetSum(testing.allocator, values[0..n], target);
    const want = subsetSumBruteForce(values[0..n], target);
    try testing.expectEqual(want, got);
}

fn subsetSumBruteForce(numbers: []const i64, target: i64) bool {
    if (target < 0) return false;

    const n = numbers.len;
    const total_masks: usize = @as(usize, 1) << @intCast(n);

    var mask: usize = 0;
    while (mask < total_masks) : (mask += 1) {
        var sum: i64 = 0;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            if (((mask >> @intCast(i)) & 1) == 1) {
                sum += numbers[i];
            }
        }
        if (sum == target) return true;
    }

    return false;
}
