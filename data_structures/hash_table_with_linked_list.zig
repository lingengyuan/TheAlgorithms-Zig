//! Hash Table With Linked List Buckets - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/hashing/hash_table_with_linked_list.py

const std = @import("std");
const testing = std.testing;

pub const KeyValue = struct {
    key: usize,
    value: i64,
};

fn lessKeyValue(_: void, a: KeyValue, b: KeyValue) bool {
    if (a.key != b.key) return a.key < b.key;
    return a.value < b.value;
}

fn isPrime(n: usize) bool {
    if (n < 2) return false;
    if (n <= 3) return true;
    if (n % 2 == 0) return false;

    var i: usize = 3;
    while (i * i <= n) : (i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

fn nextPrime(value: usize, factor: usize) usize {
    var v = value * factor;
    const first = v;

    while (!isPrime(v)) : (v += 1) {}
    if (v == first) return nextPrime(v + 1, 1);
    return v;
}

pub const HashTableWithLinkedList = struct {
    const Bucket = std.ArrayListUnmanaged(i64);

    size_table: usize,
    buckets: []Bucket,
    occupied: []bool,
    lim_charge: f64,
    charge_factor: usize,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        size_table: usize,
        charge_factor_opt: ?usize,
        lim_charge_opt: ?f64,
    ) !HashTableWithLinkedList {
        if (size_table == 0) return error.InvalidSize;

        const buckets = try allocator.alloc(Bucket, size_table);
        errdefer allocator.free(buckets);
        for (buckets) |*b| b.* = .{};

        const occupied = try allocator.alloc(bool, size_table);
        @memset(occupied, false);

        return .{
            .size_table = size_table,
            .buckets = buckets,
            .occupied = occupied,
            .lim_charge = lim_charge_opt orelse 0.75,
            .charge_factor = charge_factor_opt orelse 1,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HashTableWithLinkedList) void {
        for (self.buckets) |*b| b.deinit(self.allocator);
        self.allocator.free(self.buckets);
        self.allocator.free(self.occupied);
    }

    pub fn hashFunction(self: *const HashTableWithLinkedList, key: i64) usize {
        return @intCast(@mod(key, @as(i64, @intCast(self.size_table))));
    }

    fn bucketContains(self: *const HashTableWithLinkedList, key: usize, data: i64) bool {
        if (!self.occupied[key]) return false;
        for (self.buckets[key].items) |v| {
            if (v == data) return true;
        }
        return false;
    }

    fn countNone(self: *const HashTableWithLinkedList) usize {
        var c: usize = 0;
        for (self.occupied) |ok| {
            if (!ok) c += 1;
        }
        return c;
    }

    pub fn balancedFactor(self: *const HashTableWithLinkedList) f64 {
        var total: isize = 0;
        for (self.buckets, self.occupied) |b, ok| {
            const len = if (ok) b.items.len else 0;
            total += @as(isize, @intCast(self.charge_factor)) - @as(isize, @intCast(len));
        }

        return (@as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(self.size_table))) * @as(f64, @floatFromInt(self.charge_factor));
    }

    fn setValue(self: *HashTableWithLinkedList, key: usize, data: i64) !void {
        self.occupied[key] = true;

        var b = &self.buckets[key];
        try b.append(self.allocator, data);

        var i = b.items.len - 1;
        while (i > 0) : (i -= 1) {
            b.items[i] = b.items[i - 1];
        }
        b.items[0] = data;
    }

    fn collisionResolutionLinear(self: *HashTableWithLinkedList, key: usize) ?usize {
        var new_key = self.hashFunction(@as(i64, @intCast(key + 1)));

        while (self.occupied[new_key]) {
            if (self.countNone() > 0) {
                new_key = self.hashFunction(@as(i64, @intCast(new_key + 1)));
            } else {
                return null;
            }
        }

        return new_key;
    }

    fn collisionResolution(self: *HashTableWithLinkedList, key: usize, data: i64) ?usize {
        _ = data;

        const bucket_len = if (self.occupied[key]) self.buckets[key].items.len else 0;
        if (!(bucket_len == self.charge_factor and self.countNone() == 0)) {
            return key;
        }

        return self.collisionResolutionLinear(key);
    }

    fn rehashing(self: *HashTableWithLinkedList) anyerror!void {
        var survivors = std.ArrayListUnmanaged(i64){};
        defer survivors.deinit(self.allocator);

        for (self.buckets, self.occupied) |b, ok| {
            if (!ok) continue;
            try survivors.appendSlice(self.allocator, b.items);
        }

        for (self.buckets) |*b| b.deinit(self.allocator);
        self.allocator.free(self.buckets);
        self.allocator.free(self.occupied);

        self.size_table = nextPrime(self.size_table, 2);
        self.buckets = try self.allocator.alloc(Bucket, self.size_table);
        for (self.buckets) |*b| b.* = .{};
        self.occupied = try self.allocator.alloc(bool, self.size_table);
        @memset(self.occupied, false);

        for (survivors.items) |v| {
            try self.insertData(v);
        }
    }

    /// Inserts one value into table.
    /// Time complexity: average O(1), worst O(n)
    pub fn insertData(self: *HashTableWithLinkedList, data: i64) anyerror!void {
        const key = self.hashFunction(data);

        if (!self.occupied[key]) {
            try self.setValue(key, data);
        } else if (self.bucketContains(key, data)) {
            return;
        } else {
            const resolved = self.collisionResolution(key, data);
            if (resolved) |idx| {
                try self.setValue(idx, data);
            } else {
                try self.rehashing();
                try self.insertData(data);
            }
        }
    }

    pub fn contains(self: *const HashTableWithLinkedList, value: i64) bool {
        for (self.buckets, self.occupied) |b, ok| {
            if (!ok) continue;
            for (b.items) |v| if (v == value) return true;
        }
        return false;
    }

    pub fn keys(self: *const HashTableWithLinkedList, allocator: std.mem.Allocator) ![]KeyValue {
        var out = std.ArrayListUnmanaged(KeyValue){};
        errdefer out.deinit(allocator);

        for (self.buckets, self.occupied, 0..) |b, ok, i| {
            if (!ok) continue;
            for (b.items) |v| {
                try out.append(allocator, .{ .key = i, .value = v });
            }
        }

        std.mem.sort(KeyValue, out.items, {}, lessKeyValue);
        return out.toOwnedSlice(allocator);
    }
};

test "hash table with linked list: basic insert and contains" {
    var ht = try HashTableWithLinkedList.init(testing.allocator, 5, null, null);
    defer ht.deinit();

    try ht.insertData(10);
    try ht.insertData(15);
    try ht.insertData(20);
    try ht.insertData(25);
    try ht.insertData(30);

    try testing.expect(ht.contains(10));
    try testing.expect(ht.contains(25));
    try testing.expect(!ht.contains(99));
}

test "hash table with linked list: bucket behavior and duplicate" {
    var ht = try HashTableWithLinkedList.init(testing.allocator, 2, 2, null);
    defer ht.deinit();

    try ht.insertData(0);
    try ht.insertData(2);
    try ht.insertData(4);
    try ht.insertData(2); // duplicate

    const kv = try ht.keys(testing.allocator);
    defer testing.allocator.free(kv);

    // Items in same bucket are append-left, then flattened and sorted for stable checks.
    try testing.expectEqual(@as(usize, 3), kv.len);
    try testing.expect(ht.contains(0));
    try testing.expect(ht.contains(2));
    try testing.expect(ht.contains(4));
}

test "hash table with linked list: rehashing and extreme" {
    var ht = try HashTableWithLinkedList.init(testing.allocator, 3, 1, null);
    defer ht.deinit();

    const n: usize = 15_000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const value: i64 = @intCast(3_000_000 + i * 19);
        try ht.insertData(value);
    }

    i = 0;
    while (i < n) : (i += 997) {
        const value: i64 = @intCast(3_000_000 + i * 19);
        try testing.expect(ht.contains(value));
    }

    try testing.expect(ht.size_table > 3);
    try testing.expect(ht.balancedFactor() <= 1.0);
}

test "hash table with linked list: boundary" {
    try testing.expectError(error.InvalidSize, HashTableWithLinkedList.init(testing.allocator, 0, null, null));
}
