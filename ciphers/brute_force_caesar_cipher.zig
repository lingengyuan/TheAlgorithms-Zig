//! Brute Force Caesar Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/brute_force_caesar_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Candidate = struct {
    key: u8,
    text: []u8,
};

/// Returns all 26 Caesar decryptions for an uppercase message.
/// Non A-Z characters are preserved.
/// Time complexity: O(26 * n), Space complexity: O(26 * n)
pub fn bruteForceDecrypt(allocator: Allocator, message: []const u8) ![]Candidate {
    const candidates = try allocator.alloc(Candidate, 26);
    errdefer allocator.free(candidates);

    errdefer {
        for (0..26) |k| {
            if (candidates[k].text.len > 0) allocator.free(candidates[k].text);
        }
    }

    for (0..26) |k| {
        const key: u8 = @intCast(k);
        const out = try allocator.alloc(u8, message.len);

        for (message, 0..) |symbol, i| {
            if (symbol >= 'A' and symbol <= 'Z') {
                var num: i64 = @intCast(symbol - 'A');
                num -= @as(i64, key);
                if (num < 0) num += 26;
                out[i] = @as(u8, @intCast('A' + num));
            } else {
                out[i] = symbol;
            }
        }

        candidates[k] = Candidate{ .key = key, .text = out };
    }

    return candidates;
}

pub fn freeCandidates(allocator: Allocator, candidates: []Candidate) void {
    for (candidates) |cand| allocator.free(cand.text);
    allocator.free(candidates);
}

test "brute force caesar: python sample contains expected key 12" {
    const alloc = testing.allocator;
    const all = try bruteForceDecrypt(alloc, "TMDETUX PMDVU");
    defer freeCandidates(alloc, all);

    try testing.expectEqual(@as(usize, 26), all.len);
    try testing.expectEqual(@as(u8, 12), all[12].key);
    try testing.expectEqualStrings("HARSHIL DARJI", all[12].text);
}

test "brute force caesar: non letters preserved" {
    const alloc = testing.allocator;
    const all = try bruteForceDecrypt(alloc, "ABC-123");
    defer freeCandidates(alloc, all);

    try testing.expectEqualStrings("ABC-123", all[0].text);
    try testing.expectEqualStrings("ZAB-123", all[1].text);
}

test "brute force caesar: extreme long input" {
    const alloc = testing.allocator;

    const n: usize = 15000;
    const msg = try alloc.alloc(u8, n);
    defer alloc.free(msg);

    for (msg, 0..) |*ch, i| {
        ch.* = if (i % 7 == 0) '-' else @as(u8, @intCast('A' + (i % 26)));
    }

    const all = try bruteForceDecrypt(alloc, msg);
    defer freeCandidates(alloc, all);

    try testing.expectEqual(@as(usize, 26), all.len);
    for (all) |cand| {
        try testing.expectEqual(@as(usize, n), cand.text.len);
    }
}
