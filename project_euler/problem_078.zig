//! Project Euler Problem 78: Coin Partitions - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_078/sol1.py

const std = @import("std");
const testing = std.testing;

/// Returns the least n for which the partition number p(n) is divisible by `number`.
/// Time complexity: O(answer * sqrt(answer))
/// Space complexity: O(answer)
pub fn solution(allocator: std.mem.Allocator, number: u32) !u32 {
    var partitions = std.ArrayListUnmanaged(u32){};
    defer partitions.deinit(allocator);
    try partitions.append(allocator, 1);

    var i: u32 = 1;
    while (true) : (i += 1) {
        var item: i64 = 0;
        var j: u32 = 1;
        while (true) : (j += 1) {
            const sign: i64 = if ((j & 1) == 0) -1 else 1;
            var index = (j * j * 3 - j) / 2;
            if (index > i) break;
            item += @as(i64, partitions.items[i - index]) * sign;
            item = @mod(item, number);
            index += j;
            if (index > i) break;
            item += @as(i64, partitions.items[i - index]) * sign;
            item = @mod(item, number);
        }
        if (item == 0) return i;
        try partitions.append(allocator, @intCast(item));
    }
}

test "problem 078: python reference" {
    try testing.expectEqual(@as(u32, 1), try solution(testing.allocator, 1));
    try testing.expectEqual(@as(u32, 14), try solution(testing.allocator, 9));
    try testing.expectEqual(@as(u32, 74), try solution(testing.allocator, 100));
    try testing.expectEqual(@as(u32, 55374), try solution(testing.allocator, 1_000_000));
}
