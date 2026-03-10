//! Project Euler Problem 62: Cubic Permutations - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_062/sol1.py

const std = @import("std");
const testing = std.testing;

const Signature = struct {
    counts: [10]u8,
};

const Group = struct {
    first_base: u64,
    count: u32,
};

/// Computes the sorted decimal digits of `num^3`.
/// Caller owns the returned slice.
/// Time complexity: O(d log d)
/// Space complexity: O(d)
pub fn getDigits(allocator: std.mem.Allocator, num: u64) ![]u8 {
    var cube = num * num * num;
    var digits = std.ArrayListUnmanaged(u8){};
    errdefer digits.deinit(allocator);

    if (cube == 0) {
        try digits.append(allocator, '0');
    } else {
        while (cube > 0) : (cube /= 10) {
            try digits.append(allocator, @as(u8, @intCast(cube % 10)) + '0');
        }
        std.mem.sort(u8, digits.items, {}, comptime std.sort.asc(u8));
    }

    return digits.toOwnedSlice(allocator);
}

fn signatureOf(num: u64) Signature {
    var cube = num * num * num;
    var counts = [_]u8{0} ** 10;
    if (cube == 0) {
        counts[0] = 1;
    } else {
        while (cube > 0) : (cube /= 10) {
            counts[cube % 10] += 1;
        }
    }
    return .{ .counts = counts };
}

/// Returns the smallest cube for which exactly `max_base` cube permutations share the same digits.
/// Time complexity: unbounded search; practical runtime follows the Python reference.
/// Space complexity: O(groups)
pub fn solution(max_base: u32) u64 {
    if (max_base == 1) return 0;

    var freqs = std.AutoHashMap(Signature, Group).init(std.heap.page_allocator);
    defer freqs.deinit();

    var num: u64 = 0;
    while (true) : (num += 1) {
        const signature = signatureOf(num);
        const entry = freqs.getOrPut(signature) catch unreachable;
        if (!entry.found_existing) {
            entry.value_ptr.* = .{ .first_base = num, .count = 1 };
        } else {
            entry.value_ptr.count += 1;
        }

        if (entry.value_ptr.count == max_base) {
            const base = entry.value_ptr.first_base;
            return base * base * base;
        }
    }
}

test "problem 062: python reference" {
    try testing.expectEqual(@as(u64, 125), solution(2));
    try testing.expectEqual(@as(u64, 41063625), solution(3));
    try testing.expectEqual(@as(u64, 127035954683), solution(5));
}

test "problem 062: digit helper and extremes" {
    const alloc = testing.allocator;

    const d3 = try getDigits(alloc, 3);
    defer alloc.free(d3);
    try testing.expectEqualStrings("27", d3);

    const d99 = try getDigits(alloc, 99);
    defer alloc.free(d99);
    try testing.expectEqualStrings("027999", d99);

    const d123 = try getDigits(alloc, 123);
    defer alloc.free(d123);
    try testing.expectEqualStrings("0166788", d123);

    try testing.expectEqual(@as(u64, 0), solution(1));
}
