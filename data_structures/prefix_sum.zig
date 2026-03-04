//! Prefix Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/arrays/prefix_sum.py

const std = @import("std");
const testing = std.testing;

pub const PrefixSumError = error{ EmptyArray, InvalidRange, Overflow };

pub const PrefixSum = struct {
    allocator: std.mem.Allocator,
    prefix_sum: []i64,

    pub fn init(allocator: std.mem.Allocator, array: []const i64) !PrefixSum {
        const pref = try allocator.alloc(i64, array.len);

        if (array.len > 0) {
            pref[0] = array[0];
            var i: usize = 1;
            while (i < array.len) : (i += 1) {
                const add = @addWithOverflow(pref[i - 1], array[i]);
                if (add[1] != 0) return PrefixSumError.Overflow;
                pref[i] = add[0];
            }
        }

        return .{ .allocator = allocator, .prefix_sum = pref };
    }

    pub fn deinit(self: *PrefixSum) void {
        self.allocator.free(self.prefix_sum);
        self.* = undefined;
    }

    /// Returns sum of range [start, end].
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn getSum(self: *const PrefixSum, start: usize, end: usize) !i64 {
        if (self.prefix_sum.len == 0) return PrefixSumError.EmptyArray;
        if (start > end or end >= self.prefix_sum.len) return PrefixSumError.InvalidRange;

        if (start == 0) return self.prefix_sum[end];

        const sub = @subWithOverflow(self.prefix_sum[end], self.prefix_sum[start - 1]);
        if (sub[1] != 0) return PrefixSumError.Overflow;
        return sub[0];
    }

    /// Returns true if some contiguous subarray sums to target_sum.
    /// Time complexity: O(n), Space complexity: O(n)
    pub fn containsSum(self: *const PrefixSum, allocator: std.mem.Allocator, target_sum: i64) !bool {
        var sums = std.AutoHashMap(i64, void).init(allocator);
        defer sums.deinit();

        try sums.put(0, {});

        for (self.prefix_sum) |sum_item| {
            if (sums.contains(sum_item - target_sum)) return true;
            try sums.put(sum_item, {});
        }

        return false;
    }
};

test "prefix sum: python get_sum examples" {
    var ps = try PrefixSum.init(testing.allocator, &[_]i64{ 1, 2, 3 });
    defer ps.deinit();

    try testing.expectEqual(@as(i64, 6), try ps.getSum(0, 2));
    try testing.expectEqual(@as(i64, 5), try ps.getSum(1, 2));
    try testing.expectEqual(@as(i64, 3), try ps.getSum(2, 2));
}

test "prefix sum: invalid ranges and empty" {
    var empty = try PrefixSum.init(testing.allocator, &[_]i64{});
    defer empty.deinit();
    try testing.expectError(PrefixSumError.EmptyArray, empty.getSum(0, 0));

    var ps = try PrefixSum.init(testing.allocator, &[_]i64{ 1, 2, 3 });
    defer ps.deinit();

    try testing.expectError(PrefixSumError.InvalidRange, ps.getSum(2, 3));
    try testing.expectError(PrefixSumError.InvalidRange, ps.getSum(2, 1));
}

test "prefix sum: contains_sum and extreme" {
    var ps = try PrefixSum.init(testing.allocator, &[_]i64{ 1, 2, 3 });
    defer ps.deinit();

    try testing.expect(try ps.containsSum(testing.allocator, 6));
    try testing.expect(try ps.containsSum(testing.allocator, 5));
    try testing.expect(try ps.containsSum(testing.allocator, 3));
    try testing.expect(!(try ps.containsSum(testing.allocator, 4)));
    try testing.expect(!(try ps.containsSum(testing.allocator, 7)));

    var ps2 = try PrefixSum.init(testing.allocator, &[_]i64{ 1, -2, 3 });
    defer ps2.deinit();
    try testing.expect(try ps2.containsSum(testing.allocator, 2));

    const n: usize = 100_000;
    var values = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(values);
    for (0..n) |i| values[i] = @intCast(i + 1);

    var big = try PrefixSum.init(testing.allocator, values);
    defer big.deinit();

    try testing.expectEqual(@as(i64, @intCast((n * (n + 1)) / 2)), try big.getSum(0, n - 1));
}
