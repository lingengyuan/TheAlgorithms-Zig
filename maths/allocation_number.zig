//! Allocation Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/allocation_number.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const AllocationList = struct {
    items: [][]u8,

    pub fn deinit(self: AllocationList, allocator: Allocator) void {
        for (self.items) |item| allocator.free(item);
        allocator.free(self.items);
    }
};

/// Splits a byte range into `partitions` non-overlapping 1-based string ranges.
/// Time complexity: O(partitions), Space complexity: O(partitions)
pub fn allocationNum(
    allocator: Allocator,
    number_of_bytes: usize,
    partitions: isize,
) !AllocationList {
    if (partitions <= 0) return error.InvalidPartitions;
    const parts: usize = @intCast(partitions);
    if (parts > number_of_bytes) return error.PartitionsExceedBytes;

    const bytes_per_partition = number_of_bytes / parts;
    const items = try allocator.alloc([]u8, parts);
    errdefer allocator.free(items);

    for (0..parts) |i| {
        const start_bytes = i * bytes_per_partition + 1;
        const end_bytes = if (i == parts - 1) number_of_bytes else (i + 1) * bytes_per_partition;
        items[i] = try std.fmt.allocPrint(allocator, "{d}-{d}", .{ start_bytes, end_bytes });
    }

    return .{ .items = items };
}

test "allocation number: python reference cases" {
    const alloc = testing.allocator;

    var result = try allocationNum(alloc, 16_647, 4);
    defer result.deinit(alloc);
    try testing.expectEqual(@as(usize, 4), result.items.len);
    try testing.expectEqualStrings("1-4161", result.items[0]);
    try testing.expectEqualStrings("4162-8322", result.items[1]);
    try testing.expectEqualStrings("8323-12483", result.items[2]);
    try testing.expectEqualStrings("12484-16647", result.items[3]);

    var result2 = try allocationNum(alloc, 50_000, 5);
    defer result2.deinit(alloc);
    try testing.expectEqualStrings("1-10000", result2.items[0]);
    try testing.expectEqualStrings("40001-50000", result2.items[4]);
}

test "allocation number: edge and extreme cases" {
    const alloc = testing.allocator;
    try testing.expectError(error.PartitionsExceedBytes, allocationNum(alloc, 888, 999));
    try testing.expectError(error.InvalidPartitions, allocationNum(alloc, 888, -4));
    try testing.expectError(error.InvalidPartitions, allocationNum(alloc, 888, 0));

    var single = try allocationNum(alloc, 1, 1);
    defer single.deinit(alloc);
    try testing.expectEqual(@as(usize, 1), single.items.len);
    try testing.expectEqualStrings("1-1", single.items[0]);
}
