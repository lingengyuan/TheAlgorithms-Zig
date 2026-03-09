//! Project Euler Problem 29: Distinct Powers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_029/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem029Error = error{
    OutOfMemory,
};

const Factor = struct {
    prime: u32,
    exponent: u32,
};

fn appendUInt(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: u32) Problem029Error!void {
    var buf: [32]u8 = undefined;
    const rendered = std.fmt.bufPrint(&buf, "{}", .{value}) catch unreachable;
    try list.appendSlice(allocator, rendered);
}

fn factorize(value: u32, factors: *std.ArrayListUnmanaged(Factor), allocator: std.mem.Allocator) Problem029Error!void {
    factors.clearRetainingCapacity();

    var n = value;
    var p: u32 = 2;

    while (p * p <= n) : (p += 1) {
        if (n % p != 0) continue;

        var exponent: u32 = 0;
        while (n % p == 0) {
            n /= p;
            exponent += 1;
        }
        try factors.append(allocator, .{ .prime = p, .exponent = exponent });
    }

    if (n > 1) {
        try factors.append(allocator, .{ .prime = n, .exponent = 1 });
    }
}

fn canonicalPowerKey(
    factors: []const Factor,
    power: u32,
    key_buf: *std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,
) Problem029Error![]u8 {
    key_buf.clearRetainingCapacity();

    for (factors) |factor| {
        try appendUInt(key_buf, allocator, factor.prime);
        try key_buf.append(allocator, '^');
        try appendUInt(key_buf, allocator, factor.exponent * power);
        try key_buf.append(allocator, ';');
    }

    return try allocator.dupe(u8, key_buf.items);
}

/// Returns number of distinct terms in a^b for 2 <= a <= n and 2 <= b <= n.
///
/// Time complexity: roughly O(n^2 * log n)
/// Space complexity: O(n^2)
pub fn solution(n: u32, allocator: std.mem.Allocator) Problem029Error!u64 {
    if (n <= 1) return 0;

    var seen = std.StringHashMap(void).init(allocator);
    defer seen.deinit();

    var owned_keys = std.ArrayListUnmanaged([]u8){};
    defer {
        for (owned_keys.items) |key| allocator.free(key);
        owned_keys.deinit(allocator);
    }

    var factors = std.ArrayListUnmanaged(Factor){};
    defer factors.deinit(allocator);

    var key_builder = std.ArrayListUnmanaged(u8){};
    defer key_builder.deinit(allocator);

    var a: u32 = 2;
    while (a <= n) : (a += 1) {
        try factorize(a, &factors, allocator);

        var b: u32 = 2;
        while (b <= n) : (b += 1) {
            const key = try canonicalPowerKey(factors.items, b, &key_builder, allocator);

            if (seen.contains(key)) {
                allocator.free(key);
            } else {
                try seen.put(key, {});
                try owned_keys.append(allocator, key);
            }
        }
    }

    return seen.count();
}

test "problem 029: python reference" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 9183), try solution(100, allocator));
    try testing.expectEqual(@as(u64, 2184), try solution(50, allocator));
    try testing.expectEqual(@as(u64, 324), try solution(20, allocator));
    try testing.expectEqual(@as(u64, 15), try solution(5, allocator));
    try testing.expectEqual(@as(u64, 1), try solution(2, allocator));
    try testing.expectEqual(@as(u64, 0), try solution(1, allocator));
}

test "problem 029: small boundaries" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u64, 0), try solution(0, allocator));
    try testing.expectEqual(@as(u64, 0), try solution(1, allocator));
    try testing.expectEqual(@as(u64, 4), try solution(3, allocator));
}
