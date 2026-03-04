//! Find Triplets With 0 Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/arrays/find_triplets_with_0_sum.py

const std = @import("std");
const testing = std.testing;

pub const Triplet = struct {
    a: i64,
    b: i64,
    c: i64,
};

fn tripletEqual(lhs: Triplet, rhs: Triplet) bool {
    return lhs.a == rhs.a and lhs.b == rhs.b and lhs.c == rhs.c;
}

fn tripletLessThan(_: void, lhs: Triplet, rhs: Triplet) bool {
    if (lhs.a != rhs.a) return lhs.a < rhs.a;
    if (lhs.b != rhs.b) return lhs.b < rhs.b;
    return lhs.c < rhs.c;
}

fn sortThree(a: i64, b: i64, c: i64) Triplet {
    var x = a;
    var y = b;
    var z = c;
    if (x > y) std.mem.swap(i64, &x, &y);
    if (y > z) std.mem.swap(i64, &y, &z);
    if (x > y) std.mem.swap(i64, &x, &y);
    return .{ .a = x, .b = y, .c = z };
}

fn containsTriplet(items: []const Triplet, target: Triplet) bool {
    for (items) |item| {
        if (tripletEqual(item, target)) return true;
    }
    return false;
}

/// Returns all unique sorted triplets whose sum is zero.
/// Equivalent behavior to Python `find_triplets_with_0_sum`.
/// Time complexity: O(n^3), Space complexity: O(k)
pub fn findTripletsWith0Sum(allocator: std.mem.Allocator, nums: []const i64) ![]Triplet {
    if (nums.len < 3) return allocator.alloc(Triplet, 0);

    const sorted_nums = try allocator.alloc(i64, nums.len);
    defer allocator.free(sorted_nums);
    @memcpy(sorted_nums, nums);
    std.mem.sort(i64, sorted_nums, {}, std.sort.asc(i64));

    var unique = std.AutoHashMap(Triplet, void).init(allocator);
    defer unique.deinit();

    var i: usize = 0;
    while (i + 2 < sorted_nums.len) : (i += 1) {
        var j: usize = i + 1;
        while (j + 1 < sorted_nums.len) : (j += 1) {
            var k: usize = j + 1;
            while (k < sorted_nums.len) : (k += 1) {
                const sum = @as(i128, sorted_nums[i]) + @as(i128, sorted_nums[j]) + @as(i128, sorted_nums[k]);
                if (sum == 0) {
                    try unique.put(.{
                        .a = sorted_nums[i],
                        .b = sorted_nums[j],
                        .c = sorted_nums[k],
                    }, {});
                }
            }
        }
    }

    const out = try allocator.alloc(Triplet, unique.count());
    var idx: usize = 0;
    var it = unique.keyIterator();
    while (it.next()) |key_ptr| : (idx += 1) {
        out[idx] = key_ptr.*;
    }
    std.mem.sort(Triplet, out, {}, tripletLessThan);
    return out;
}

/// Returns all unique triplets by hashing strategy.
/// Equivalent behavior to Python `find_triplets_with_0_sum_hashing`.
/// Time complexity: O(n^2 * k), Space complexity: O(n + k)
pub fn findTripletsWith0SumHashing(allocator: std.mem.Allocator, arr: []const i64) ![]Triplet {
    var output = std.ArrayListUnmanaged(Triplet){};
    errdefer output.deinit(allocator);

    if (arr.len < 3) return output.toOwnedSlice(allocator);

    var index: usize = 0;
    while (index + 2 < arr.len) : (index += 1) {
        const item = arr[index];
        const current_sum = -@as(i128, item);

        var seen = std.AutoHashMap(i64, void).init(allocator);
        defer seen.deinit();

        for (arr[index + 1 ..]) |other_item| {
            const required_i128 = current_sum - @as(i128, other_item);
            if (required_i128 >= std.math.minInt(i64) and required_i128 <= std.math.maxInt(i64)) {
                const required_value: i64 = @intCast(required_i128);
                if (seen.contains(required_value)) {
                    const triplet = sortThree(item, other_item, required_value);
                    if (!containsTriplet(output.items, triplet)) {
                        try output.append(allocator, triplet);
                    }
                }
            }
            try seen.put(other_item, {});
        }
    }

    return output.toOwnedSlice(allocator);
}

fn expectTripletsEqual(expected: []const Triplet, actual: []const Triplet) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, 0..) |item, i| {
        try testing.expect(tripletEqual(item, actual[i]));
    }
}

test "find triplets with 0 sum: python examples combinations" {
    {
        const out = try findTripletsWith0Sum(testing.allocator, &[_]i64{ -1, 0, 1, 2, -1, -4 });
        defer testing.allocator.free(out);
        const expected = [_]Triplet{
            .{ .a = -1, .b = -1, .c = 2 },
            .{ .a = -1, .b = 0, .c = 1 },
        };
        try expectTripletsEqual(expected[0..], out);
    }
    {
        const out = try findTripletsWith0Sum(testing.allocator, &[_]i64{});
        defer testing.allocator.free(out);
        try testing.expectEqual(@as(usize, 0), out.len);
    }
    {
        const out = try findTripletsWith0Sum(testing.allocator, &[_]i64{ 0, 0, 0 });
        defer testing.allocator.free(out);
        const expected_zero = [_]Triplet{.{ .a = 0, .b = 0, .c = 0 }};
        try expectTripletsEqual(expected_zero[0..], out);
    }
    {
        const out = try findTripletsWith0Sum(testing.allocator, &[_]i64{ 1, 2, 3, 0, -1, -2, -3 });
        defer testing.allocator.free(out);
        const expected2 = [_]Triplet{
            .{ .a = -3, .b = 0, .c = 3 },
            .{ .a = -3, .b = 1, .c = 2 },
            .{ .a = -2, .b = -1, .c = 3 },
            .{ .a = -2, .b = 0, .c = 2 },
            .{ .a = -1, .b = 0, .c = 1 },
        };
        try expectTripletsEqual(expected2[0..], out);
    }
}

test "find triplets with 0 sum: python examples hashing" {
    {
        const out = try findTripletsWith0SumHashing(testing.allocator, &[_]i64{ -1, 0, 1, 2, -1, -4 });
        defer testing.allocator.free(out);
        const expected = [_]Triplet{
            .{ .a = -1, .b = 0, .c = 1 },
            .{ .a = -1, .b = -1, .c = 2 },
        };
        try expectTripletsEqual(expected[0..], out);
    }
    {
        const out = try findTripletsWith0SumHashing(testing.allocator, &[_]i64{});
        defer testing.allocator.free(out);
        try testing.expectEqual(@as(usize, 0), out.len);
    }
    {
        const out = try findTripletsWith0SumHashing(testing.allocator, &[_]i64{ 0, 0, 0 });
        defer testing.allocator.free(out);
        const expected_zero = [_]Triplet{.{ .a = 0, .b = 0, .c = 0 }};
        try expectTripletsEqual(expected_zero[0..], out);
    }
    {
        const out = try findTripletsWith0SumHashing(testing.allocator, &[_]i64{ 1, 2, 3, 0, -1, -2, -3 });
        defer testing.allocator.free(out);
        const expected2 = [_]Triplet{
            .{ .a = -1, .b = 0, .c = 1 },
            .{ .a = -3, .b = 1, .c = 2 },
            .{ .a = -2, .b = 0, .c = 2 },
            .{ .a = -2, .b = -1, .c = 3 },
            .{ .a = -3, .b = 0, .c = 3 },
        };
        try expectTripletsEqual(expected2[0..], out);
    }
}

test "find triplets with 0 sum: extreme and uniqueness" {
    const n: usize = 150;
    const values = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(values);

    for (0..n) |i| {
        values[i] = @as(i64, @intCast((i % 50))) - 25;
    }

    const combo = try findTripletsWith0Sum(testing.allocator, values);
    defer testing.allocator.free(combo);

    for (combo, 0..) |triplet, idx| {
        try testing.expectEqual(@as(i128, 0), @as(i128, triplet.a) + @as(i128, triplet.b) + @as(i128, triplet.c));
        try testing.expect(triplet.a <= triplet.b and triplet.b <= triplet.c);
        if (idx > 0) {
            const prev = combo[idx - 1];
            try testing.expect(!tripletEqual(prev, triplet));
            try testing.expect(!tripletLessThan({}, triplet, prev));
        }
    }

    const hash_out = try findTripletsWith0SumHashing(testing.allocator, values);
    defer testing.allocator.free(hash_out);

    for (hash_out, 0..) |triplet, i| {
        try testing.expectEqual(@as(i128, 0), @as(i128, triplet.a) + @as(i128, triplet.b) + @as(i128, triplet.c));
        for (hash_out[i + 1 ..]) |other| {
            try testing.expect(!tripletEqual(triplet, other));
        }
    }
}
