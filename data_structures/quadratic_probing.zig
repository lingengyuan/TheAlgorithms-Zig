//! Quadratic Probing Hash Table - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/hashing/quadratic_probing.py

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

pub const QuadraticProbing = struct {
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
    ) !QuadraticProbing {
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

    pub fn deinit(self: *QuadraticProbing) void {
        self.allocator.free(self.values);
        self.keys_map.deinit();
    }

    pub fn hashFunction(self: *const QuadraticProbing, key: i64) usize {
        return @intCast(@mod(key, @as(i64, @intCast(self.size_table))));
    }

    pub fn balancedFactor(self: *const QuadraticProbing) f64 {
        var occupied: usize = 0;
        for (self.values) |slot| {
            if (slot != null) occupied += 1;
        }
        const denom = @as(f64, @floatFromInt(self.size_table * self.charge_factor));
        return @as(f64, @floatFromInt(occupied)) / denom;
    }

    fn setValue(self: *QuadraticProbing, key: usize, data: i64) !void {
        self.values[key] = data;
        try self.keys_map.put(key, data);
    }

    /// Quadratic probing collision resolution.
    fn collisionResolution(self: *QuadraticProbing, key: usize, data: i64) ?usize {
        var i: usize = 1;
        var new_key = self.hashFunction(@as(i64, @intCast(key + i * i)));

        while (self.values[new_key] != null and self.values[new_key].? != data) {
            i += 1;
            if (!(self.balancedFactor() >= self.lim_charge)) {
                new_key = self.hashFunction(@as(i64, @intCast(key + i * i)));
            } else {
                return null;
            }
        }

        return new_key;
    }

    fn rehashing(self: *QuadraticProbing) anyerror!void {
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

    /// Inserts one value.
    /// Time complexity: average O(1), worst O(n)
    pub fn insertData(self: *QuadraticProbing, data: i64) anyerror!void {
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

    pub fn keys(self: *const QuadraticProbing, allocator: std.mem.Allocator) ![]KeyValue {
        var out = std.ArrayListUnmanaged(KeyValue){};
        errdefer out.deinit(allocator);

        var it = self.keys_map.iterator();
        while (it.next()) |entry| {
            try out.append(allocator, .{ .key = entry.key_ptr.*, .value = entry.value_ptr.* });
        }

        std.mem.sort(KeyValue, out.items, {}, lessKeyValue);
        return out.toOwnedSlice(allocator);
    }

    pub fn contains(self: *const QuadraticProbing, value: i64) bool {
        for (self.values) |slot| {
            if (slot != null and slot.? == value) return true;
        }
        return false;
    }
};

test "quadratic probing: python doctest examples" {
    {
        var qp = try QuadraticProbing.init(testing.allocator, 7, null, null);
        defer qp.deinit();

        try qp.insertData(90);
        try qp.insertData(340);
        try qp.insertData(24);
        try qp.insertData(45);
        try qp.insertData(99);
        try qp.insertData(73);
        try qp.insertData(7);

        const kv = try qp.keys(testing.allocator);
        defer testing.allocator.free(kv);

        try testing.expectEqual(@as(usize, 17), qp.size_table);
        try testing.expectEqualSlices(KeyValue, &[_]KeyValue{
            .{ .key = 0, .value = 340 },
            .{ .key = 5, .value = 73 },
            .{ .key = 6, .value = 90 },
            .{ .key = 7, .value = 24 },
            .{ .key = 8, .value = 7 },
            .{ .key = 11, .value = 45 },
            .{ .key = 14, .value = 99 },
        }, kv);
    }

    {
        var qp = try QuadraticProbing.init(testing.allocator, 8, null, null);
        defer qp.deinit();

        try qp.insertData(0);
        try qp.insertData(999);
        try qp.insertData(111);

        const kv = try qp.keys(testing.allocator);
        defer testing.allocator.free(kv);
        try testing.expectEqualSlices(KeyValue, &[_]KeyValue{
            .{ .key = 0, .value = 0 },
            .{ .key = 3, .value = 111 },
            .{ .key = 7, .value = 999 },
        }, kv);
    }

    {
        var qp = try QuadraticProbing.init(testing.allocator, 2, null, null);
        defer qp.deinit();

        try qp.insertData(0);
        try qp.insertData(999);
        try qp.insertData(111);

        const kv = try qp.keys(testing.allocator);
        defer testing.allocator.free(kv);
        try testing.expectEqualSlices(KeyValue, &[_]KeyValue{
            .{ .key = 0, .value = 0 },
            .{ .key = 1, .value = 111 },
            .{ .key = 4, .value = 999 },
        }, kv);
    }

    {
        var qp = try QuadraticProbing.init(testing.allocator, 1, null, null);
        defer qp.deinit();

        try qp.insertData(0);
        try qp.insertData(999);
        try qp.insertData(111);

        const kv = try qp.keys(testing.allocator);
        defer testing.allocator.free(kv);
        try testing.expectEqualSlices(KeyValue, &[_]KeyValue{
            .{ .key = 0, .value = 0 },
            .{ .key = 1, .value = 111 },
            .{ .key = 4, .value = 999 },
        }, kv);
    }
}

test "quadratic probing: boundary" {
    try testing.expectError(error.InvalidSize, QuadraticProbing.init(testing.allocator, 0, null, null));
}

test "quadratic probing: extreme large insert" {
    const n: usize = 20_000;

    var qp = try QuadraticProbing.init(testing.allocator, 5, null, null);
    defer qp.deinit();

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const value: i64 = @intCast(2_000_000 + i * 17);
        try qp.insertData(value);
    }

    i = 0;
    while (i < n) : (i += 997) {
        const value: i64 = @intCast(2_000_000 + i * 17);
        try testing.expect(qp.contains(value));
    }

    try testing.expect(qp.size_table > 5);
}

test "quadratic probing: collision resolution preserves displaced value equal to bucket index" {
    var qp = try QuadraticProbing.init(testing.allocator, 10, null, null);
    defer qp.deinit();

    try qp.insertData(16);
    try qp.insertData(6);
    try qp.insertData(26);

    const kv = try qp.keys(testing.allocator);
    defer testing.allocator.free(kv);
    try testing.expectEqualSlices(KeyValue, &[_]KeyValue{
        .{ .key = 0, .value = 26 },
        .{ .key = 6, .value = 16 },
        .{ .key = 7, .value = 6 },
    }, kv);
}
