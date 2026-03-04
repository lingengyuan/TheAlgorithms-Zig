//! Diffie-Hellman Key Exchange - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/diffie_hellman.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const DiffieHellmanError = error{ UnsupportedGroup, InvalidPublicKey, InvalidHexKey };

const Group = struct {
    prime: u128,
    generator: u128,
};

fn groupParams(group: u8) ?Group {
    // Zig implementation uses toy-safe prime groups for practical u128 arithmetic.
    return switch (group) {
        5 => .{ .prime = 23, .generator = 4 },
        14 => .{ .prime = 47, .generator = 4 },
        15 => .{ .prime = 59, .generator = 4 },
        16 => .{ .prime = 83, .generator = 4 },
        17 => .{ .prime = 107, .generator = 4 },
        18 => .{ .prime = 167, .generator = 4 },
        else => null,
    };
}

fn powMod(base_init: u128, exp_init: u128, modulus: u128) u128 {
    if (modulus == 1) return 0;

    var base = base_init % modulus;
    var exp = exp_init;
    var result: u128 = 1;

    while (exp > 0) {
        if ((exp & 1) == 1) result = (result * base) % modulus;
        base = (base * base) % modulus;
        exp >>= 1;
    }

    return result;
}

fn parseHexU128(hex: []const u8) !u128 {
    return std.fmt.parseInt(u128, hex, 16) catch DiffieHellmanError.InvalidHexKey;
}

fn toHex(allocator: Allocator, value: u128) ![]u8 {
    return try std.fmt.allocPrint(allocator, "{x}", .{value});
}

fn sha256Hex(allocator: Allocator, payload: []const u8) ![]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(payload, &digest, .{});

    const out = try allocator.alloc(u8, 64);
    errdefer allocator.free(out);
    const hex = "0123456789abcdef";
    for (digest, 0..) |b, i| {
        out[2 * i] = hex[b >> 4];
        out[2 * i + 1] = hex[b & 0x0F];
    }
    return out;
}

pub const DiffieHellman = struct {
    prime: u128,
    generator: u128,
    private_key: u128,

    pub fn init(random: std.Random, group: u8) !DiffieHellman {
        const params = groupParams(group) orelse return DiffieHellmanError.UnsupportedGroup;

        var priv: u128 = 0;
        while (true) {
            priv = random.intRangeAtMost(u128, 2, params.prime - 2);
            const pub_candidate = powMod(params.generator, priv, params.prime);
            if (isValidPublicKeyStatic(pub_candidate, params.prime)) break;
        }

        return DiffieHellman{ .prime = params.prime, .generator = params.generator, .private_key = priv };
    }

    pub fn getPrivateKey(self: DiffieHellman, allocator: Allocator) ![]u8 {
        return toHex(allocator, self.private_key);
    }

    pub fn generatePublicKey(self: DiffieHellman, allocator: Allocator) ![]u8 {
        const public_key = powMod(self.generator, self.private_key, self.prime);
        return toHex(allocator, public_key);
    }

    pub fn isValidPublicKey(self: DiffieHellman, key: u128) bool {
        return 2 <= key and key <= self.prime - 2 and powMod(key, (self.prime - 1) / 2, self.prime) == 1;
    }

    pub fn generateSharedKey(self: DiffieHellman, allocator: Allocator, other_key_hex: []const u8) ![]u8 {
        const other_key = try parseHexU128(other_key_hex);
        if (!self.isValidPublicKey(other_key)) return DiffieHellmanError.InvalidPublicKey;

        const shared = powMod(other_key, self.private_key, self.prime);
        const shared_dec = try std.fmt.allocPrint(allocator, "{d}", .{shared});
        defer allocator.free(shared_dec);

        return sha256Hex(allocator, shared_dec);
    }

    pub fn isValidPublicKeyStatic(remote_public_key: u128, prime: u128) bool {
        return 2 <= remote_public_key and remote_public_key <= prime - 2 and powMod(remote_public_key, (prime - 1) / 2, prime) == 1;
    }

    pub fn generateSharedKeyStatic(
        allocator: Allocator,
        local_private_hex: []const u8,
        remote_public_hex: []const u8,
        group: u8,
    ) ![]u8 {
        const params = groupParams(group) orelse return DiffieHellmanError.UnsupportedGroup;
        const local_private = try parseHexU128(local_private_hex);
        const remote_public = try parseHexU128(remote_public_hex);

        if (!isValidPublicKeyStatic(remote_public, params.prime)) return DiffieHellmanError.InvalidPublicKey;

        const shared = powMod(remote_public, local_private, params.prime);
        const shared_dec = try std.fmt.allocPrint(allocator, "{d}", .{shared});
        defer allocator.free(shared_dec);

        return sha256Hex(allocator, shared_dec);
    }
};

test "diffie hellman: object and static shared key agreement" {
    const alloc = testing.allocator;

    var prng = std.Random.DefaultPrng.init(777);
    const rng = prng.random();

    const alice = try DiffieHellman.init(rng, 14);
    const bob = try DiffieHellman.init(rng, 14);

    const alice_private = try alice.getPrivateKey(alloc);
    defer alloc.free(alice_private);
    const alice_public = try alice.generatePublicKey(alloc);
    defer alloc.free(alice_public);

    const bob_private = try bob.getPrivateKey(alloc);
    defer alloc.free(bob_private);
    const bob_public = try bob.generatePublicKey(alloc);
    defer alloc.free(bob_public);

    const alice_shared = try alice.generateSharedKey(alloc, bob_public);
    defer alloc.free(alice_shared);
    const bob_shared = try bob.generateSharedKey(alloc, alice_public);
    defer alloc.free(bob_shared);

    try testing.expectEqualStrings(alice_shared, bob_shared);

    const alice_static = try DiffieHellman.generateSharedKeyStatic(alloc, alice_private, bob_public, 14);
    defer alloc.free(alice_static);
    const bob_static = try DiffieHellman.generateSharedKeyStatic(alloc, bob_private, alice_public, 14);
    defer alloc.free(bob_static);

    try testing.expectEqualStrings(alice_static, bob_static);
    try testing.expectEqualStrings(alice_static, alice_shared);
}

test "diffie hellman: invalid cases" {
    const alloc = testing.allocator;

    var prng = std.Random.DefaultPrng.init(778);
    const rng = prng.random();

    try testing.expectError(DiffieHellmanError.UnsupportedGroup, DiffieHellman.init(rng, 99));

    const dh = try DiffieHellman.init(rng, 14);
    try testing.expect(!dh.isValidPublicKey(1));

    try testing.expectError(DiffieHellmanError.InvalidPublicKey, dh.generateSharedKey(alloc, "1"));
    try testing.expectError(DiffieHellmanError.InvalidHexKey, dh.generateSharedKey(alloc, "zzzz"));
}

test "diffie hellman: extreme repeated handshakes" {
    const alloc = testing.allocator;

    var prng = std.Random.DefaultPrng.init(779);
    const rng = prng.random();

    var i: usize = 0;
    while (i < 2000) : (i += 1) {
        const a = try DiffieHellman.init(rng, 17);
        const b = try DiffieHellman.init(rng, 17);

        const ap = try a.generatePublicKey(alloc);
        defer alloc.free(ap);
        const bp = try b.generatePublicKey(alloc);
        defer alloc.free(bp);

        const as = try a.generateSharedKey(alloc, bp);
        defer alloc.free(as);
        const bs = try b.generateSharedKey(alloc, ap);
        defer alloc.free(bs);
        try testing.expectEqualStrings(as, bs);
    }
}
