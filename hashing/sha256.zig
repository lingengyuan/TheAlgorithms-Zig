//! SHA-256 - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/hashes/sha256.py

const std = @import("std");
const testing = std.testing;

const INITIAL_HASH = [_]u32{
    0x6a09e667,
    0xbb67ae85,
    0x3c6ef372,
    0xa54ff53a,
    0x510e527f,
    0x9b05688c,
    0x1f83d9ab,
    0x5be0cd19,
};

const ROUND_CONSTANTS = [_]u32{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

const HEX_DIGITS = "0123456789abcdef";

fn rotr(x: u32, shift: u5) u32 {
    return std.math.rotr(u32, x, shift);
}

fn smallSigma0(x: u32) u32 {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

fn smallSigma1(x: u32) u32 {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

fn bigSigma0(x: u32) u32 {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

fn bigSigma1(x: u32) u32 {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

fn choose(x: u32, y: u32, z: u32) u32 {
    return (x & y) ^ (~x & z);
}

fn majority(x: u32, y: u32, z: u32) u32 {
    return (x & y) ^ (x & z) ^ (y & z);
}

fn processChunk(chunk: []const u8, state: *[8]u32, schedule: *[64]u32) void {
    std.debug.assert(chunk.len == 64);

    for (0..16) |i| {
        const j = i * 4;
        schedule[i] = (@as(u32, chunk[j]) << 24) |
            (@as(u32, chunk[j + 1]) << 16) |
            (@as(u32, chunk[j + 2]) << 8) |
            @as(u32, chunk[j + 3]);
    }

    for (16..64) |i| {
        schedule[i] = smallSigma1(schedule[i - 2]) +%
            schedule[i - 7] +%
            smallSigma0(schedule[i - 15]) +%
            schedule[i - 16];
    }

    var a = state[0];
    var b = state[1];
    var c = state[2];
    var d = state[3];
    var e = state[4];
    var f = state[5];
    var g = state[6];
    var h = state[7];

    for (0..64) |i| {
        const t1 = h +% bigSigma1(e) +% choose(e, f, g) +% ROUND_CONSTANTS[i] +% schedule[i];
        const t2 = bigSigma0(a) +% majority(a, b, c);

        h = g;
        g = f;
        f = e;
        e = d +% t1;
        d = c;
        c = b;
        b = a;
        a = t1 +% t2;
    }

    state[0] +%= a;
    state[1] +%= b;
    state[2] +%= c;
    state[3] +%= d;
    state[4] +%= e;
    state[5] +%= f;
    state[6] +%= g;
    state[7] +%= h;
}

/// Returns SHA-256 digest bytes for the input.
/// Time complexity: O(n), where n is input length.
/// Space complexity: O(1), excluding the fixed-size output buffer.
pub fn sha256(data: []const u8) [32]u8 {
    var state = INITIAL_HASH;
    var schedule: [64]u32 = undefined;

    const full_len = data.len - (data.len % 64);
    var offset: usize = 0;
    while (offset < full_len) : (offset += 64) {
        processChunk(data[offset .. offset + 64], &state, &schedule);
    }

    var tail = [_]u8{0} ** 128;
    const remainder = data.len - full_len;
    @memcpy(tail[0..remainder], data[full_len..]);
    tail[remainder] = 0x80;

    const total_tail_len: usize = if (remainder <= 55) 64 else 128;
    const bit_length = @as(u64, @intCast(data.len)) *% 8;
    const length_bytes = tail[total_tail_len - 8 .. total_tail_len];
    length_bytes[0] = @intCast((bit_length >> 56) & 0xff);
    length_bytes[1] = @intCast((bit_length >> 48) & 0xff);
    length_bytes[2] = @intCast((bit_length >> 40) & 0xff);
    length_bytes[3] = @intCast((bit_length >> 32) & 0xff);
    length_bytes[4] = @intCast((bit_length >> 24) & 0xff);
    length_bytes[5] = @intCast((bit_length >> 16) & 0xff);
    length_bytes[6] = @intCast((bit_length >> 8) & 0xff);
    length_bytes[7] = @intCast(bit_length & 0xff);

    processChunk(tail[0..64], &state, &schedule);
    if (total_tail_len == 128) processChunk(tail[64..128], &state, &schedule);

    var digest: [32]u8 = undefined;
    for (state, 0..) |word, i| {
        const base = i * 4;
        digest[base] = @intCast((word >> 24) & 0xff);
        digest[base + 1] = @intCast((word >> 16) & 0xff);
        digest[base + 2] = @intCast((word >> 8) & 0xff);
        digest[base + 3] = @intCast(word & 0xff);
    }
    return digest;
}

/// Returns lowercase hexadecimal SHA-256 string.
/// Caller owns the returned slice.
/// Time complexity: O(n), where n is input length.
/// Space complexity: O(1) additional working space, plus O(64) output.
pub fn sha256Hex(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    const digest = sha256(data);
    const out = try allocator.alloc(u8, 64);
    errdefer allocator.free(out);

    for (digest, 0..) |b, i| {
        out[i * 2] = HEX_DIGITS[b >> 4];
        out[i * 2 + 1] = HEX_DIGITS[b & 0x0f];
    }
    return out;
}

fn expectSha256Hex(input: []const u8, expected: []const u8) !void {
    const alloc = testing.allocator;
    const hex = try sha256Hex(alloc, input);
    defer alloc.free(hex);
    try testing.expectEqualStrings(expected, hex);
}

test "sha256: known vectors" {
    try expectSha256Hex("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    try expectSha256Hex("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    try expectSha256Hex("hello world", "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
    try expectSha256Hex(
        "The quick brown fox jumps over the lazy dog",
        "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
    );
}

test "sha256: padding boundaries 55 56 64 bytes" {
    const alloc = testing.allocator;
    const lengths = [_]usize{ 55, 56, 64 };
    const expected = [_][]const u8{
        "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
        "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
        "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
    };

    for (lengths, 0..) |len, i| {
        const data = try alloc.alloc(u8, len);
        defer alloc.free(data);
        @memset(data, 'a');
        try expectSha256Hex(data, expected[i]);
    }
}

test "sha256: extreme long message one million bytes" {
    const alloc = testing.allocator;
    const data = try alloc.alloc(u8, 1_000_000);
    defer alloc.free(data);
    @memset(data, 'a');
    try expectSha256Hex(data, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}
