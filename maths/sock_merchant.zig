//! Sock Merchant - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/sock_merchant.py

const std = @import("std");
const testing = std.testing;

/// Returns number of matching sock pairs by color.
/// Time complexity: O(n), Space complexity: O(n)
pub fn sockMerchant(allocator: std.mem.Allocator, colors: []const i64) std.mem.Allocator.Error!usize {
    var counts = std.AutoHashMap(i64, usize).init(allocator);
    defer counts.deinit();

    for (colors) |color| {
        const entry = try counts.getOrPut(color);
        if (entry.found_existing) {
            entry.value_ptr.* += 1;
        } else {
            entry.value_ptr.* = 1;
        }
    }

    var pairs: usize = 0;
    var it = counts.iterator();
    while (it.next()) |entry| {
        pairs += entry.value_ptr.* / 2;
    }
    return pairs;
}

test "sock merchant: python reference examples" {
    try testing.expectEqual(@as(usize, 3), try sockMerchant(testing.allocator, &[_]i64{ 10, 20, 20, 10, 10, 30, 50, 10, 20 }));
    try testing.expectEqual(@as(usize, 2), try sockMerchant(testing.allocator, &[_]i64{ 1, 1, 3, 3 }));
}

test "sock merchant: edge and extreme cases" {
    try testing.expectEqual(@as(usize, 0), try sockMerchant(testing.allocator, &[_]i64{}));
    try testing.expectEqual(@as(usize, 0), try sockMerchant(testing.allocator, &[_]i64{1}));

    var big: [100_000]i64 = undefined;
    for (&big, 0..) |*slot, idx| slot.* = @intCast(idx % 50);
    try testing.expectEqual(@as(usize, 50_000), try sockMerchant(testing.allocator, &big));
}
