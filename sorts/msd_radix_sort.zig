//! MSD Radix Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/msd_radix_sort.py

const std = @import("std");
const testing = std.testing;

pub const MsdRadixSortError = error{NegativeValue};

fn bitLength(x: u64) usize {
    if (x == 0) return 1;
    return 64 - @clz(x);
}

fn msdRadixSortInplaceRange(list: []u64, bit_position: usize, begin_index: usize, end_index: usize) void {
    if (bit_position == 0 or end_index <= begin_index + 1) return;

    const next_bit = bit_position - 1;
    var i = begin_index;
    var j = end_index - 1;

    while (i <= j) {
        var changed = false;
        if (((list[i] >> @as(u6, @intCast(next_bit))) & 1) == 0) {
            i += 1;
            changed = true;
        }
        if (((list[j] >> @as(u6, @intCast(next_bit))) & 1) == 1) {
            if (j == 0) break;
            j -= 1;
            changed = true;
        }
        if (changed) continue;

        std.mem.swap(u64, &list[i], &list[j]);
        if (j == 0) break;
        j -= 1;
        if (j != i) i += 1;
    }

    msdRadixSortInplaceRange(list, next_bit, begin_index, i);
    msdRadixSortInplaceRange(list, next_bit, i, end_index);
}

/// In-place MSD radix sort for non-negative integers.
/// Time complexity: O(w * n), Space complexity: O(1) extra (in-place recursion stack excluded)
pub fn msdRadixSortInplace(list_of_ints: []i64) MsdRadixSortError!void {
    if (list_of_ints.len <= 1) return;
    for (list_of_ints) |v| if (v < 0) return error.NegativeValue;

    var max_val: u64 = 0;
    for (list_of_ints) |v| {
        const u: u64 = @intCast(v);
        if (u > max_val) max_val = u;
    }
    const most_bits = bitLength(max_val);

    // reinterpret via temporary u64 buffer to avoid signed-shift issues
    var temp = [_]u64{0} ** 0;
    _ = &temp;
    const as_u64 = @as([*]u64, @ptrCast(list_of_ints.ptr))[0..list_of_ints.len];
    msdRadixSortInplaceRange(as_u64, most_bits, 0, as_u64.len);
}

/// Returns sorted copy using MSD radix sort.
/// Caller owns returned slice.
pub fn msdRadixSort(allocator: std.mem.Allocator, list_of_ints: []const i64) (MsdRadixSortError || std.mem.Allocator.Error)![]i64 {
    if (list_of_ints.len == 0) return try allocator.alloc(i64, 0);
    for (list_of_ints) |v| if (v < 0) return error.NegativeValue;

    const out = try allocator.alloc(i64, list_of_ints.len);
    @memcpy(out, list_of_ints);
    try msdRadixSortInplace(out);
    return out;
}

test "msd radix sort: python reference examples" {
    const alloc = testing.allocator;

    const r1 = try msdRadixSort(alloc, &[_]i64{ 40, 12, 1, 100, 4 });
    defer alloc.free(r1);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 4, 12, 40, 100 }, r1);

    const r2 = try msdRadixSort(alloc, &[_]i64{});
    defer alloc.free(r2);
    try testing.expectEqual(@as(usize, 0), r2.len);

    const r3 = try msdRadixSort(alloc, &[_]i64{ 123, 345, 123, 80 });
    defer alloc.free(r3);
    try testing.expectEqualSlices(i64, &[_]i64{ 80, 123, 123, 345 }, r3);

    const r4 = try msdRadixSort(alloc, &[_]i64{ 1209, 834598, 1, 540402, 45 });
    defer alloc.free(r4);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 45, 1209, 540402, 834598 }, r4);
}

test "msd radix sort: inplace and negative error cases" {
    var lst1 = [_]i64{ 1, 345, 23, 89, 0, 3 };
    try msdRadixSortInplace(&lst1);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 1, 3, 23, 89, 345 }, &lst1);

    var lst2 = [_]i64{ 1, 43, 0, 0, 0, 24, 3, 3 };
    try msdRadixSortInplace(&lst2);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 0, 0, 1, 3, 3, 24, 43 }, &lst2);

    var bad = [_]i64{ -1, 34, 23, 4, -42 };
    try testing.expectError(error.NegativeValue, msdRadixSortInplace(&bad));
}

test "msd radix sort: extreme random non-negative input" {
    const alloc = testing.allocator;
    const n: usize = 100_000;
    const arr = try alloc.alloc(i64, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(777);
    var random = prng.random();
    for (arr) |*v| v.* = random.intRangeAtMost(i64, 0, 10_000_000);

    try msdRadixSortInplace(arr);
    for (1..arr.len) |i| try testing.expect(arr[i - 1] <= arr[i]);
}
