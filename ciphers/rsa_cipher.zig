//! RSA Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/rsa_cipher.py

const std = @import("std");
const testing = std.testing;

pub const RsaCipherError = error{
    InvalidBlockSize,
    BlockTooLarge,
};

pub const DEFAULT_BLOCK_SIZE: usize = 128;
pub const BYTE_SIZE: u128 = 256;

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

fn checkedBlockSize(block_size: usize) !void {
    if (block_size == 0) return RsaCipherError.InvalidBlockSize;
    if (block_size > 16) return RsaCipherError.BlockTooLarge;
}

/// Splits plaintext bytes into RSA integer blocks.
/// Time complexity: O(n), Space complexity: O(n)
pub fn getBlocksFromText(allocator: std.mem.Allocator, message: []const u8, block_size: usize) ![]u128 {
    try checkedBlockSize(block_size);

    var blocks = std.ArrayListUnmanaged(u128){};
    errdefer blocks.deinit(allocator);

    var block_start: usize = 0;
    while (block_start < message.len) : (block_start += block_size) {
        var block_int: u128 = 0;
        const end = @min(block_start + block_size, message.len);

        var i = block_start;
        while (i < end) : (i += 1) {
            const exp = @as(u7, @intCast(i % block_size));
            block_int += @as(u128, message[i]) * std.math.pow(u128, BYTE_SIZE, exp);
        }

        try blocks.append(allocator, block_int);
    }

    return try blocks.toOwnedSlice(allocator);
}

/// Reconstructs plaintext bytes from integer blocks.
/// Time complexity: O(num_blocks * block_size), Space complexity: O(n)
pub fn getTextFromBlocks(allocator: std.mem.Allocator, block_ints: []const u128, message_length: usize, block_size: usize) ![]u8 {
    try checkedBlockSize(block_size);

    var message = std.ArrayListUnmanaged(u8){};
    errdefer message.deinit(allocator);
    try message.ensureTotalCapacity(allocator, message_length);

    for (block_ints) |block_init| {
        var block_int = block_init;
        var block_message = std.ArrayListUnmanaged(u8){};
        defer block_message.deinit(allocator);

        var i: i32 = @intCast(block_size - 1);
        while (i >= 0) : (i -= 1) {
            const idx: usize = @intCast(i);
            if (message.items.len + idx < message_length) {
                const power = std.math.pow(u128, BYTE_SIZE, @as(u7, @intCast(idx)));
                const ascii_number = block_int / power;
                block_int %= power;
                try block_message.insert(allocator, 0, @intCast(ascii_number));
            }
        }

        try message.appendSlice(allocator, block_message.items);
    }

    return try message.toOwnedSlice(allocator);
}

/// Encrypts plaintext with RSA key (n,e).
/// Time complexity: O(num_blocks * log e), Space complexity: O(num_blocks)
pub fn encryptMessage(allocator: std.mem.Allocator, message: []const u8, key: [2]u64, block_size: usize) ![]u128 {
    const n: u128 = key[0];
    const e: u128 = key[1];

    const blocks = try getBlocksFromText(allocator, message, block_size);
    defer allocator.free(blocks);

    const encrypted = try allocator.alloc(u128, blocks.len);
    errdefer allocator.free(encrypted);

    for (blocks, 0..) |block, i| encrypted[i] = powMod(block, e, n);

    return encrypted;
}

/// Decrypts RSA blocks with key (n,d).
/// Time complexity: O(num_blocks * log d), Space complexity: O(n)
pub fn decryptMessage(allocator: std.mem.Allocator, encrypted_blocks: []const u128, message_length: usize, key: [2]u64, block_size: usize) ![]u8 {
    const n: u128 = key[0];
    const d: u128 = key[1];

    const decrypted_blocks = try allocator.alloc(u128, encrypted_blocks.len);
    defer allocator.free(decrypted_blocks);

    for (encrypted_blocks, 0..) |block, i| decrypted_blocks[i] = powMod(block, d, n);

    return getTextFromBlocks(allocator, decrypted_blocks, message_length, block_size);
}

test "rsa cipher: block conversion round trip" {
    const alloc = testing.allocator;

    const blocks = try getBlocksFromText(alloc, "HELLO", 2);
    defer alloc.free(blocks);

    const text = try getTextFromBlocks(alloc, blocks, 5, 2);
    defer alloc.free(text);

    try testing.expectEqualStrings("HELLO", text);
}

test "rsa cipher: encrypt/decrypt with toy key" {
    const alloc = testing.allocator;

    // toy RSA values: p=61, q=53, n=3233, e=17, d=2753
    const pub_key: [2]u64 = .{ 3233, 17 };
    const priv_key: [2]u64 = .{ 3233, 2753 };

    const message = "A";
    const enc = try encryptMessage(alloc, message, pub_key, 1);
    defer alloc.free(enc);

    const dec = try decryptMessage(alloc, enc, message.len, priv_key, 1);
    defer alloc.free(dec);

    try testing.expectEqualStrings(message, dec);
}

test "rsa cipher: invalid block size" {
    const alloc = testing.allocator;

    try testing.expectError(RsaCipherError.InvalidBlockSize, getBlocksFromText(alloc, "abc", 0));
    try testing.expectError(RsaCipherError.BlockTooLarge, getBlocksFromText(alloc, "abc", 17));
}
