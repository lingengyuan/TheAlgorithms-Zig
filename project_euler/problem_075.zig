//! Project Euler Problem 75: Singular Integer Right Triangles - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_075/sol1.py

const std = @import("std");
const testing = std.testing;

fn gcd(a: u32, b: u32) u32 {
    var x = a;
    var y = b;
    while (y != 0) {
        const remainder = x % y;
        x = y;
        y = remainder;
    }
    return x;
}

/// Returns the number of perimeters `<= limit` that form exactly one integer right triangle.
/// Time complexity: roughly O(limit log limit)
/// Space complexity: O(limit)
pub fn solution(allocator: std.mem.Allocator, limit: u32) !u32 {
    var frequencies = try allocator.alloc(u32, limit + 1);
    defer allocator.free(frequencies);
    @memset(frequencies, 0);

    var euclid_m: u32 = 2;
    while (2 * euclid_m * (euclid_m + 1) <= limit) : (euclid_m += 1) {
        var euclid_n: u32 = (euclid_m % 2) + 1;
        while (euclid_n < euclid_m) : (euclid_n += 2) {
            if (gcd(euclid_m, euclid_n) > 1) continue;
            const primitive_perimeter = 2 * euclid_m * (euclid_m + euclid_n);
            var perimeter = primitive_perimeter;
            while (perimeter <= limit) : (perimeter += primitive_perimeter) {
                frequencies[perimeter] += 1;
            }
        }
    }

    var count: u32 = 0;
    for (frequencies) |frequency| {
        if (frequency == 1) count += 1;
    }
    return count;
}

test "problem 075: python reference" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(u32, 6), try solution(alloc, 50));
    try testing.expectEqual(@as(u32, 112), try solution(alloc, 1000));
    try testing.expectEqual(@as(u32, 5502), try solution(alloc, 50_000));
    try testing.expectEqual(@as(u32, 161667), try solution(alloc, 1_500_000));
}

test "problem 075: degenerate and tiny limits" {
    try testing.expectEqual(@as(u32, 0), try solution(testing.allocator, 0));
    try testing.expectEqual(@as(u32, 0), try solution(testing.allocator, 11));
    try testing.expectEqual(@as(u32, 1), try solution(testing.allocator, 12));
}
