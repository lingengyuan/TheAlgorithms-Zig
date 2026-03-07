//! Coordinate Compression - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_compression/coordinate_compression.py

const std = @import("std");
const testing = std.testing;

pub const CoordinateCompressor = struct {
    allocator: std.mem.Allocator,
    coordinate_map: std.AutoHashMap(i64, usize),
    reverse_map: []i64,

    /// Builds compression and reverse lookup maps from input values.
    ///
    /// Time complexity: O(n log n)
    /// Space complexity: O(n)
    pub fn init(allocator: std.mem.Allocator, arr: []const i64) !CoordinateCompressor {
        var coordinate_map = std.AutoHashMap(i64, usize).init(allocator);
        errdefer coordinate_map.deinit();

        const reverse_map = try allocator.alloc(i64, arr.len);
        errdefer allocator.free(reverse_map);
        @memset(reverse_map, -1);

        const sorted = try allocator.alloc(i64, arr.len);
        defer allocator.free(sorted);
        std.mem.copyForwards(i64, sorted, arr);
        std.sort.heap(i64, sorted, {}, struct {
            fn lessThan(_: void, a: i64, b: i64) bool {
                return a < b;
            }
        }.lessThan);

        var key: usize = 0;
        for (sorted) |value| {
            const gop = try coordinate_map.getOrPut(value);
            if (!gop.found_existing) {
                gop.value_ptr.* = key;
                reverse_map[key] = value;
                key += 1;
            }
        }

        return CoordinateCompressor{
            .allocator = allocator,
            .coordinate_map = coordinate_map,
            .reverse_map = reverse_map,
        };
    }

    pub fn deinit(self: *CoordinateCompressor) void {
        self.coordinate_map.deinit();
        self.allocator.free(self.reverse_map);
    }

    /// Compresses original value to coordinate index, returns -1 if missing.
    pub fn compress(self: *const CoordinateCompressor, original: i64) isize {
        if (self.coordinate_map.get(original)) |idx| {
            return @as(isize, @intCast(idx));
        }
        return -1;
    }

    /// Decompresses coordinate index to original value, returns -1 if out-of-range.
    pub fn decompress(self: *const CoordinateCompressor, num: isize) i64 {
        if (num < 0) return -1;
        const idx: usize = @intCast(num);
        if (idx >= self.reverse_map.len) return -1;
        return self.reverse_map[idx];
    }
};

test "coordinate compression: python examples" {
    const arr = [_]i64{ 100, 10, 52, 83 };
    var cc = try CoordinateCompressor.init(testing.allocator, &arr);
    defer cc.deinit();

    try testing.expectEqual(@as(isize, 3), cc.compress(100));
    try testing.expectEqual(@as(isize, 1), cc.compress(52));
    try testing.expectEqual(@as(i64, 52), cc.decompress(1));

    try testing.expectEqual(@as(isize, -1), cc.compress(7));
    try testing.expectEqual(@as(i64, -1), cc.decompress(5));
}

test "coordinate compression: duplicates and extreme values" {
    const arr = [_]i64{ 5, 5, 5, 10, -2, 10, -2 };
    var cc = try CoordinateCompressor.init(testing.allocator, &arr);
    defer cc.deinit();

    try testing.expectEqual(@as(isize, 0), cc.compress(-2));
    try testing.expectEqual(@as(isize, 1), cc.compress(5));
    try testing.expectEqual(@as(isize, 2), cc.compress(10));

    try testing.expectEqual(@as(i64, -1), cc.decompress(6));

    const large = try testing.allocator.alloc(i64, 10_000);
    defer testing.allocator.free(large);
    for (0..large.len) |i| {
        large[i] = @as(i64, @intCast(large.len - i));
    }
    var cc_large = try CoordinateCompressor.init(testing.allocator, large);
    defer cc_large.deinit();

    try testing.expectEqual(@as(isize, 0), cc_large.compress(1));
    try testing.expectEqual(@as(isize, 9_999), cc_large.compress(10_000));
}
