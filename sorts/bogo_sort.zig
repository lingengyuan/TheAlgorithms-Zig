//! Bogo Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/bogo_sort.py

const std = @import("std");
const testing = std.testing;

pub const BogoSortError = error{
    InputTooLarge,
    MaxShufflesExceeded,
};

fn isSorted(comptime T: type, arr: []const T) bool {
    if (arr.len <= 1) return true;
    for (1..arr.len) |i| {
        if (arr[i - 1] > arr[i]) return false;
    }
    return true;
}

fn shuffle(comptime T: type, random: *std.Random, arr: []T) void {
    if (arr.len <= 1) return;

    var i = arr.len;
    while (i > 1) {
        i -= 1;
        const j = random.uintLessThan(usize, i + 1);
        std.mem.swap(T, &arr[i], &arr[j]);
    }
}

/// In-place bogo sort using deterministic PRNG seed.
/// A size guard is applied to avoid runaway runtime.
/// Time complexity: unbounded expected factorial; Space complexity: O(1)
pub fn bogoSort(comptime T: type, arr: []T, seed: u64, maxShuffles: usize) BogoSortError!void {
    if (arr.len <= 1) return;
    if (arr.len > 8) return error.InputTooLarge;

    var prng = std.Random.DefaultPrng.init(seed);
    var random = prng.random();

    var shuffle_count: usize = 0;
    while (!isSorted(T, arr)) {
        if (shuffle_count >= maxShuffles) return error.MaxShufflesExceeded;
        shuffle(T, &random, arr);
        shuffle_count += 1;
    }
}

test "bogo sort: python reference examples" {
    var a1 = [_]i32{ 0, 5, 3, 2, 2 };
    try bogoSort(i32, &a1, 7, 200_000);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 2, 3, 5 }, &a1);

    var a2 = [_]i32{};
    try bogoSort(i32, &a2, 7, 10);
    try testing.expectEqual(@as(usize, 0), a2.len);

    var a3 = [_]i32{ -2, -5, -45 };
    try bogoSort(i32, &a3, 11, 50_000);
    try testing.expectEqualSlices(i32, &[_]i32{ -45, -5, -2 }, &a3);
}

test "bogo sort: boundary and failure behavior" {
    var sorted = [_]i32{ 1, 2, 3, 4 };
    try bogoSort(i32, &sorted, 1, 0);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4 }, &sorted);

    var unsorted_small = [_]i32{ 2, 1 };
    try testing.expectError(error.MaxShufflesExceeded, bogoSort(i32, &unsorted_small, 1, 0));

    var too_large = [_]i32{ 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    try testing.expectError(error.InputTooLarge, bogoSort(i32, &too_large, 1, 1_000));
}

test "bogo sort: extreme allowed-size case" {
    var arr = [_]i32{ 6, 5, 4, 3, 2, 1 };
    try bogoSort(i32, &arr, 1234, 1_000_000);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6 }, &arr);
}
