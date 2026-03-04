//! Hash Table (Open Addressing + Linear Probing) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/hashing/hash_table.py

const std = @import("std");
const testing = std.testing;

pub const KeyValue = struct {
    key: usize,
    value: i64,
};

fn lessKeyValue(_: void, a: KeyValue, b: KeyValue) bool {
    return a.key < b.key;
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

pub const HashTable = struct {
    size_table: usize,
    values: []?i64,
    lim_charge: f64,
    charge_factor: usize,
    keys_map: std.AutoHashMap(usize, i64),
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        size_table: usize,
        charge_factor_opt: ?usize,
        lim_charge_opt: ?f64,
    ) !HashTable {
        if (size_table == 0) return error.InvalidSize;

        const values = try allocator.alloc(?i64, size_table);
        @memset(values, null);

        return .{
            .size_table = size_table,
            .values = values,
            .lim_charge = lim_charge_opt orelse 0.75,
            .charge_factor = charge_factor_opt orelse 1,
            .keys_map = std.AutoHashMap(usize, i64).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HashTable) void {
        self.allocator.free(self.values);
        self.keys_map.deinit();
    }

    pub fn hashFunction(self: *const HashTable, key: i64) usize {
        return @intCast(@mod(key, @as(i64, @intCast(self.size_table))));
    }

    pub fn balancedFactor(self: *const HashTable) f64 {
        var occupied: usize = 0;
        for (self.values) |slot| {
            if (slot != null) occupied += 1;
        }

        const denom = @as(f64, @floatFromInt(self.size_table * self.charge_factor));
        return @as(f64, @floatFromInt(occupied)) / denom;
    }

    fn countNone(self: *const HashTable) usize {
        var count: usize = 0;
        for (self.values) |slot| {
            if (slot == null) count += 1;
        }
        return count;
    }

    fn setValue(self: *HashTable, key: usize, data: i64) !void {
        self.values[key] = data;
        try self.keys_map.put(key, data);
    }

    /// Linear-probing collision resolution.
    /// Intentionally follows Python reference behavior.
    fn collisionResolution(self: *HashTable, key: usize, data: i64) ?usize {
        _ = data;

        var new_key = self.hashFunction(@as(i64, @intCast(key + 1)));

        while (self.values[new_key] != null and self.values[new_key].? != @as(i64, @intCast(key))) {
            if (self.countNone() > 0) {
                new_key = self.hashFunction(@as(i64, @intCast(new_key + 1)));
            } else {
                return null;
            }
        }

        return new_key;
    }

    fn rehashing(self: *HashTable) anyerror!void {
        var survivor_values = std.ArrayListUnmanaged(i64){};
        defer survivor_values.deinit(self.allocator);

        for (self.values) |slot| {
            if (slot) |v| try survivor_values.append(self.allocator, v);
        }

        self.size_table = nextPrime(self.size_table, 2);
        self.keys_map.clearRetainingCapacity();

        self.allocator.free(self.values);
        self.values = try self.allocator.alloc(?i64, self.size_table);
        @memset(self.values, null);

        for (survivor_values.items) |v| {
            try self.insertData(v);
        }
    }

    /// Inserts one value into hash table.
    /// Time complexity: average O(1), worst O(n)
    pub fn insertData(self: *HashTable, data: i64) anyerror!void {
        const key = self.hashFunction(data);

        if (self.values[key] == null) {
            try self.setValue(key, data);
        } else if (self.values[key].? == data) {
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

    pub fn bulkInsert(self: *HashTable, values: []const i64) anyerror!void {
        for (values) |v| try self.insertData(v);
    }

    pub fn contains(self: *const HashTable, data: i64) bool {
        const start = self.hashFunction(data);
        var idx = start;

        var steps: usize = 0;
        while (steps < self.size_table) : (steps += 1) {
            const slot = self.values[idx];
            if (slot == null) return false;
            if (slot.? == data) return true;
            idx = (idx + 1) % self.size_table;
        }

        return false;
    }

    pub fn keys(self: *const HashTable, allocator: std.mem.Allocator) ![]KeyValue {
        var out = std.ArrayListUnmanaged(KeyValue){};
        errdefer out.deinit(allocator);

        var it = self.keys_map.iterator();
        while (it.next()) |entry| {
            try out.append(allocator, .{ .key = entry.key_ptr.*, .value = entry.value_ptr.* });
        }

        std.mem.sort(KeyValue, out.items, {}, lessKeyValue);
        return out.toOwnedSlice(allocator);
    }
};

test "hash table: hash function examples" {
    var ht = try HashTable.init(testing.allocator, 5, null, null);
    defer ht.deinit();

    try testing.expectEqual(@as(usize, 0), ht.hashFunction(10));
    try testing.expectEqual(@as(usize, 0), ht.hashFunction(20));
    try testing.expectEqual(@as(usize, 4), ht.hashFunction(4));
    try testing.expectEqual(@as(usize, 3), ht.hashFunction(18));
    try testing.expectEqual(@as(usize, 2), ht.hashFunction(-18));
    try testing.expectEqual(@as(usize, 0), ht.hashFunction(0));
}

test "hash table: insert_data and keys examples" {
    {
        var ht = try HashTable.init(testing.allocator, 10, null, null);
        defer ht.deinit();
        try ht.insertData(10);
        try ht.insertData(20);
        try ht.insertData(30);

        const kv = try ht.keys(testing.allocator);
        defer testing.allocator.free(kv);
        try testing.expectEqualSlices(KeyValue, &[_]KeyValue{
            .{ .key = 0, .value = 10 },
            .{ .key = 1, .value = 20 },
            .{ .key = 2, .value = 30 },
        }, kv);
    }

    {
        var ht = try HashTable.init(testing.allocator, 5, null, null);
        defer ht.deinit();
        try ht.bulkInsert(&[_]i64{ 5, 4, 3, 2, 1 });

        const kv = try ht.keys(testing.allocator);
        defer testing.allocator.free(kv);
        try testing.expectEqualSlices(KeyValue, &[_]KeyValue{
            .{ .key = 0, .value = 5 },
            .{ .key = 1, .value = 1 },
            .{ .key = 2, .value = 2 },
            .{ .key = 3, .value = 3 },
            .{ .key = 4, .value = 4 },
        }, kv);
    }
}

test "hash table: rehashing examples and boundary" {
    {
        var ht = try HashTable.init(testing.allocator, 2, null, null);
        defer ht.deinit();

        try ht.insertData(17);
        try ht.insertData(18);
        try ht.insertData(99);

        const kv = try ht.keys(testing.allocator);
        defer testing.allocator.free(kv);
        try testing.expectEqualSlices(KeyValue, &[_]KeyValue{
            .{ .key = 2, .value = 17 },
            .{ .key = 3, .value = 18 },
            .{ .key = 4, .value = 99 },
        }, kv);
    }

    {
        var ht = try HashTable.init(testing.allocator, 1, null, null);
        defer ht.deinit();

        try ht.insertData(17);
        try ht.insertData(18);
        try ht.insertData(99);

        const kv = try ht.keys(testing.allocator);
        defer testing.allocator.free(kv);
        try testing.expectEqualSlices(KeyValue, &[_]KeyValue{
            .{ .key = 2, .value = 17 },
            .{ .key = 3, .value = 18 },
            .{ .key = 4, .value = 99 },
        }, kv);
    }

    try testing.expectError(error.InvalidSize, HashTable.init(testing.allocator, 0, null, null));
}

test "hash table: extreme large insert and contains" {
    const n: usize = 20_000;

    var ht = try HashTable.init(testing.allocator, 7, null, null);
    defer ht.deinit();

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const value: i64 = @intCast(1_000_000 + i * 13);
        try ht.insertData(value);
    }

    i = 0;
    while (i < n) : (i += 997) {
        const value: i64 = @intCast(1_000_000 + i * 13);
        try testing.expect(ht.contains(value));
    }

    try testing.expect(!ht.contains(-123456789));
    try testing.expect(ht.size_table > 7);
}
