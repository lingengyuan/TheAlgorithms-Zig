//! Diffie Primitive Root Finder - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/diffie.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

fn powMod(base_init: u64, exp_init: u64, modulus: u64) u64 {
    if (modulus == 1) return 0;

    var base = base_init % modulus;
    var exp = exp_init;
    var result: u64 = 1 % modulus;

    while (exp > 0) {
        if ((exp & 1) == 1) {
            result = @as(u64, @intCast((@as(u128, result) * @as(u128, base)) % modulus));
        }
        base = @as(u64, @intCast((@as(u128, base) * @as(u128, base)) % modulus));
        exp >>= 1;
    }

    return result;
}

/// Finds a primitive root modulo `modulus`, if it exists.
/// Returns `null` when no primitive root exists for the given modulus.
/// Time complexity: O(m^2 log m), Space complexity: O(m)
pub fn findPrimitive(allocator: Allocator, modulus: u64) !?u64 {
    if (modulus <= 1) return null;

    for (1..modulus) |r| {
        const seen = try allocator.alloc(bool, modulus);
        defer allocator.free(seen);
        @memset(seen, false);

        var has_repeat = false;
        for (0..(modulus - 1)) |x| {
            const val = powMod(@intCast(r), @intCast(x), modulus);
            if (seen[val]) {
                has_repeat = true;
                break;
            }
            seen[val] = true;
        }

        if (!has_repeat) return @intCast(r);
    }

    return null;
}

test "diffie: python samples" {
    const alloc = testing.allocator;

    try testing.expectEqual(@as(?u64, 3), try findPrimitive(alloc, 7));
    try testing.expectEqual(@as(?u64, 2), try findPrimitive(alloc, 11));
    try testing.expectEqual(@as(?u64, null), try findPrimitive(alloc, 8));
}

test "diffie: edge modulus" {
    const alloc = testing.allocator;
    try testing.expectEqual(@as(?u64, null), try findPrimitive(alloc, 0));
    try testing.expectEqual(@as(?u64, null), try findPrimitive(alloc, 1));
}

test "diffie: extreme-ish prime modulus coverage" {
    const alloc = testing.allocator;

    const primitive = (try findPrimitive(alloc, 97)).?;
    try testing.expect(primitive >= 1 and primitive < 97);

    var seen = [_]bool{false} ** 97;
    for (0..96) |x| {
        const v = powMod(primitive, @intCast(x), 97);
        seen[v] = true;
    }

    for (1..97) |i| {
        try testing.expect(seen[i]);
    }
}
