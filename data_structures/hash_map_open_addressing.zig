//! Hash Map (Open Addressing) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/hashing/hash_map.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const OpenAddressHashMap = struct {
    const Self = @This();

    const BucketState = enum {
        empty,
        occupied,
        deleted,
    };

    const Bucket = struct {
        state: BucketState = .empty,
        key: i64 = 0,
        value: i64 = 0,
    };

    allocator: Allocator,
    buckets: []Bucket,
    len: usize,
    deleted: usize,

    pub fn init(allocator: Allocator, initial_capacity: usize) !Self {
        const cap = try normalizeCapacity(initial_capacity);
        const buckets = try allocator.alloc(Bucket, cap);
        @memset(buckets, Bucket{});
        return .{
            .allocator = allocator,
            .buckets = buckets,
            .len = 0,
            .deleted = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.buckets);
        self.* = undefined;
    }

    pub fn count(self: *const Self) usize {
        return self.len;
    }

    pub fn contains(self: *const Self, key: i64) bool {
        return self.get(key) != null;
    }

    pub fn put(self: *Self, key: i64, value: i64) !void {
        if (self.shouldGrow()) {
            const doubled = @mulWithOverflow(self.buckets.len, @as(usize, 2));
            if (doubled[1] != 0) return error.Overflow;
            try self.resize(doubled[0]);
        }

        const index = self.findInsertIndex(key);
        const bucket = &self.buckets[index];
        if (bucket.state == .occupied) {
            bucket.value = value;
            return;
        }

        if (bucket.state == .deleted) {
            self.deleted -= 1;
        }
        bucket.state = .occupied;
        bucket.key = key;
        bucket.value = value;
        self.len += 1;
    }

    pub fn get(self: *const Self, key: i64) ?i64 {
        const index = self.findExistingIndex(key) orelse return null;
        return self.buckets[index].value;
    }

    pub fn remove(self: *Self, key: i64) bool {
        const index = self.findExistingIndex(key) orelse return false;
        self.buckets[index].state = .deleted;
        self.len -= 1;
        self.deleted += 1;
        return true;
    }

    fn normalizeCapacity(capacity: usize) !usize {
        var cap = if (capacity < 8) @as(usize, 8) else capacity;
        if ((cap & (cap - 1)) != 0) {
            var p: usize = 1;
            while (p < cap) {
                const doubled = @mulWithOverflow(p, @as(usize, 2));
                if (doubled[1] != 0) return error.Overflow;
                p = doubled[0];
            }
            cap = p;
        }
        return cap;
    }

    fn shouldGrow(self: *const Self) bool {
        // Grow by occupied + tombstones to keep probe chains short.
        const used_plus_deleted = @addWithOverflow(self.len, self.deleted);
        if (used_plus_deleted[1] != 0) return true;
        const used = @addWithOverflow(used_plus_deleted[0], @as(usize, 1));
        if (used[1] != 0) return true;

        const lhs = @mulWithOverflow(used[0], @as(usize, 100));
        if (lhs[1] != 0) return true;
        const rhs = @mulWithOverflow(self.buckets.len, @as(usize, 70));
        if (rhs[1] != 0) return true;
        return lhs[0] >= rhs[0];
    }

    fn findInsertIndex(self: *const Self, key: i64) usize {
        const cap = self.buckets.len;
        var first_deleted: ?usize = null;
        var idx = bucketIndex(cap, key);

        var probes: usize = 0;
        while (probes < cap) : (probes += 1) {
            const bucket = self.buckets[idx];
            switch (bucket.state) {
                .empty => return first_deleted orelse idx,
                .deleted => {
                    if (first_deleted == null) first_deleted = idx;
                },
                .occupied => {
                    if (bucket.key == key) return idx;
                },
            }
            idx = (idx + 1) & (cap - 1);
        }

        return first_deleted orelse 0;
    }

    fn findExistingIndex(self: *const Self, key: i64) ?usize {
        const cap = self.buckets.len;
        var idx = bucketIndex(cap, key);
        var probes: usize = 0;

        while (probes < cap) : (probes += 1) {
            const bucket = self.buckets[idx];
            switch (bucket.state) {
                .empty => return null,
                .occupied => if (bucket.key == key) return idx,
                .deleted => {},
            }
            idx = (idx + 1) & (cap - 1);
        }
        return null;
    }

    fn resize(self: *Self, new_capacity: usize) !void {
        const new_cap = try normalizeCapacity(new_capacity);
        const old_buckets = self.buckets;

        self.buckets = try self.allocator.alloc(Bucket, new_cap);
        @memset(self.buckets, Bucket{});
        self.len = 0;
        self.deleted = 0;

        for (old_buckets) |bucket| {
            if (bucket.state == .occupied) {
                const idx = self.findInsertIndex(bucket.key);
                self.buckets[idx] = .{
                    .state = .occupied,
                    .key = bucket.key,
                    .value = bucket.value,
                };
                self.len += 1;
            }
        }
        self.allocator.free(old_buckets);
    }

    fn bucketIndex(capacity: usize, key: i64) usize {
        const h = hashKey(key);
        return @as(usize, @intCast(h & @as(u64, @intCast(capacity - 1))));
    }

    fn hashKey(key: i64) u64 {
        var x: u64 = @bitCast(key);
        x ^= x >> 33;
        x *%= 0xff51afd7ed558ccd;
        x ^= x >> 33;
        x *%= 0xc4ceb9fe1a85ec53;
        x ^= x >> 33;
        return x;
    }
};

test "hash map open addressing: basic put/get/update" {
    var map = try OpenAddressHashMap.init(testing.allocator, 8);
    defer map.deinit();

    try map.put(1, 10);
    try map.put(2, 20);
    try map.put(1, 30);

    try testing.expectEqual(@as(?i64, 30), map.get(1));
    try testing.expectEqual(@as(?i64, 20), map.get(2));
    try testing.expectEqual(@as(usize, 2), map.count());
}

test "hash map open addressing: remove and tombstone reuse" {
    var map = try OpenAddressHashMap.init(testing.allocator, 8);
    defer map.deinit();

    try map.put(11, 1);
    try map.put(19, 2);
    try map.put(27, 3);
    try testing.expect(map.remove(19));
    try testing.expectEqual(@as(?i64, null), map.get(19));
    try testing.expectEqual(@as(usize, 2), map.count());

    try map.put(35, 4);
    try testing.expectEqual(@as(?i64, 4), map.get(35));
    try testing.expectEqual(@as(usize, 3), map.count());
}

test "hash map open addressing: missing remove and contains" {
    var map = try OpenAddressHashMap.init(testing.allocator, 8);
    defer map.deinit();

    try map.put(5, 50);
    try testing.expect(!map.remove(6));
    try testing.expect(map.contains(5));
    try testing.expect(!map.contains(6));
}

test "hash map open addressing: grows under heavy insertions" {
    var map = try OpenAddressHashMap.init(testing.allocator, 2);
    defer map.deinit();

    for (0..2_000) |i| {
        const k: i64 = @intCast(i);
        const v: i64 = @intCast(i * 3);
        try map.put(k, v);
    }

    try testing.expectEqual(@as(usize, 2_000), map.count());
    try testing.expectEqual(@as(?i64, 0), map.get(0));
    try testing.expectEqual(@as(?i64, 3_000), map.get(1_000));
    try testing.expectEqual(@as(?i64, 5_997), map.get(1_999));
}

test "hash map open addressing: extreme keys" {
    var map = try OpenAddressHashMap.init(testing.allocator, 8);
    defer map.deinit();

    const lo = std.math.minInt(i64);
    const hi = std.math.maxInt(i64);
    try map.put(lo, -1);
    try map.put(hi, 1);

    try testing.expectEqual(@as(?i64, -1), map.get(lo));
    try testing.expectEqual(@as(?i64, 1), map.get(hi));
}

test "hash map open addressing: empty map lookups" {
    var map = try OpenAddressHashMap.init(testing.allocator, 8);
    defer map.deinit();

    try testing.expectEqual(@as(?i64, null), map.get(42));
    try testing.expect(!map.remove(42));
    try testing.expectEqual(@as(usize, 0), map.count());
}

test "hash map open addressing: oversize capacity returns overflow" {
    try testing.expectError(error.Overflow, OpenAddressHashMap.init(testing.allocator, std.math.maxInt(usize)));
}
