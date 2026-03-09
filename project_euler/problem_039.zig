//! Project Euler Problem 39: Integer Right Triangles - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_039/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem039Error = error{
    OutOfMemory,
};

/// Returns a perimeter-to-count table for integral right triangles up to `max_perimeter`.
///
/// Time complexity: O(p^2)
/// Space complexity: O(p)
pub fn pythagoreanTripleCounts(max_perimeter: u32, allocator: std.mem.Allocator) Problem039Error![]u16 {
    const counts = try allocator.alloc(u16, max_perimeter + 1);
    @memset(counts, 0);

    var base: u32 = 1;
    while (base <= max_perimeter) : (base += 1) {
        var perpendicular: u32 = base;
        while (perpendicular <= max_perimeter) : (perpendicular += 1) {
            const hyp_sq = @as(u64, base) * base + @as(u64, perpendicular) * perpendicular;
            const root = std.math.sqrt(@as(f64, @floatFromInt(hyp_sq)));
            const hypotenuse: u32 = @intFromFloat(root);
            if (@as(u64, hypotenuse) * hypotenuse != hyp_sq) continue;

            const perimeter = base + perpendicular + hypotenuse;
            if (perimeter > max_perimeter) break;
            counts[perimeter] += 1;
        }
    }

    return counts;
}

/// Returns the perimeter with the largest number of integer right-triangle solutions.
pub fn solution(limit: u32, allocator: std.mem.Allocator) Problem039Error!u32 {
    const counts = try allocator.alloc(u16, limit + 1);
    defer allocator.free(counts);
    @memset(counts, 0);

    var order = std.ArrayListUnmanaged(u32){};
    defer order.deinit(allocator);

    var base: u32 = 1;
    while (base <= limit) : (base += 1) {
        var perpendicular: u32 = base;
        while (perpendicular <= limit) : (perpendicular += 1) {
            const hyp_sq = @as(u64, base) * base + @as(u64, perpendicular) * perpendicular;
            const root = std.math.sqrt(@as(f64, @floatFromInt(hyp_sq)));
            const hypotenuse: u32 = @intFromFloat(root);
            if (@as(u64, hypotenuse) * hypotenuse != hyp_sq) continue;

            const perimeter = base + perpendicular + hypotenuse;
            if (perimeter > limit) break;
            if (counts[perimeter] == 0) try order.append(allocator, perimeter);
            counts[perimeter] += 1;
        }
    }

    var best_perimeter: u32 = 0;
    var best_count: u16 = 0;
    for (order.items) |perimeter| {
        if (counts[perimeter] > best_count) {
            best_count = counts[perimeter];
            best_perimeter = perimeter;
        }
    }
    return best_perimeter;
}

test "problem 039: python reference" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u32, 90), try solution(100, allocator));
    try testing.expectEqual(@as(u32, 180), try solution(200, allocator));
    try testing.expectEqual(@as(u32, 840), try solution(1000, allocator));
}

test "problem 039: helper counts and extremes" {
    const allocator = testing.allocator;
    const counts_50 = try pythagoreanTripleCounts(50, allocator);
    defer allocator.free(counts_50);

    try testing.expectEqual(@as(u16, 1), counts_50[12]);
    try testing.expectEqual(@as(u16, 1), counts_50[24]);
    try testing.expectEqual(@as(u16, 1), counts_50[30]);
    try testing.expectEqual(@as(u16, 1), counts_50[36]);
    try testing.expectEqual(@as(u16, 1), counts_50[40]);
    try testing.expectEqual(@as(u16, 1), counts_50[48]);
    try testing.expectEqual(@as(u32, 0), try solution(11, allocator));
}
