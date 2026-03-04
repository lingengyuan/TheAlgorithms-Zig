//! Baconian Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/baconian_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const CipherError = error{
    InvalidCharacter,
    InvalidSymbol,
    InvalidLength,
    InvalidCode,
};

const encode_table = [_][]const u8{
    "AAAAA", // a
    "AAAAB", // b
    "AAABA", // c
    "AAABB", // d
    "AABAA", // e
    "AABAB", // f
    "AABBA", // g
    "AABBB", // h
    "ABAAA", // i
    "BBBAA", // j
    "ABAAB", // k
    "ABABA", // l
    "ABABB", // m
    "ABBAA", // n
    "ABBAB", // o
    "ABBBA", // p
    "ABBBB", // q
    "BAAAA", // r
    "BAAAB", // s
    "BAABA", // t
    "BAABB", // u
    "BBBAB", // v
    "BABAA", // w
    "BABAB", // x
    "BABBA", // y
    "BABBB", // z
};

fn decodePattern(pattern: []const u8) ?u8 {
    for (encode_table, 0..) |p, idx| {
        if (std.mem.eql(u8, p, pattern)) return @intCast('a' + idx);
    }
    return null;
}

/// Encodes letters/spaces to Baconian A/B code.
/// Accepts only alphabetic characters and spaces.
/// Time complexity: O(n), Space complexity: O(n)
pub fn encode(allocator: Allocator, word: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    for (word) |ch| {
        if (ch == ' ') {
            try out.append(allocator, ' ');
            continue;
        }

        const lower = std.ascii.toLower(ch);
        if (!std.ascii.isAlphabetic(lower)) return CipherError.InvalidCharacter;

        const idx = lower - 'a';
        try out.appendSlice(allocator, encode_table[idx]);
    }

    return try out.toOwnedSlice(allocator);
}

/// Decodes Baconian A/B code.
/// Accepts only 'A', 'B', and spaces.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decode(allocator: Allocator, coded: []const u8) ![]u8 {
    for (coded) |ch| {
        if (ch != 'A' and ch != 'B' and ch != ' ') return CipherError.InvalidSymbol;
    }

    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    var i: usize = 0;
    while (i < coded.len) {
        if (coded[i] == ' ') {
            try out.append(allocator, ' ');
            i += 1;
            continue;
        }

        var j = i;
        while (j < coded.len and coded[j] != ' ') : (j += 1) {}
        const word = coded[i..j];
        if (word.len % 5 != 0) return CipherError.InvalidLength;

        var k: usize = 0;
        while (k < word.len) : (k += 5) {
            const pattern = word[k .. k + 5];
            const ch = decodePattern(pattern) orelse return CipherError.InvalidCode;
            try out.append(allocator, ch);
        }

        i = j;
    }

    // Python decode strips trailing spaces.
    while (out.items.len > 0 and out.items[out.items.len - 1] == ' ') {
        _ = out.pop();
    }

    return try out.toOwnedSlice(allocator);
}

test "baconian: python samples" {
    const alloc = testing.allocator;

    const e1 = try encode(alloc, "hello");
    defer alloc.free(e1);
    try testing.expectEqualStrings("AABBBAABAAABABAABABAABBAB", e1);

    const e2 = try encode(alloc, "hello world");
    defer alloc.free(e2);
    try testing.expectEqualStrings("AABBBAABAAABABAABABAABBAB BABAAABBABBAAAAABABAAAABB", e2);

    const d = try decode(alloc, "AABBBAABAAABABAABABAABBAB BABAAABBABBAAAAABABAAAABB");
    defer alloc.free(d);
    try testing.expectEqualStrings("hello world", d);
}

test "baconian: invalid inputs" {
    const alloc = testing.allocator;
    try testing.expectError(CipherError.InvalidCharacter, encode(alloc, "hello world!"));
    try testing.expectError(CipherError.InvalidSymbol, decode(alloc, "AABX"));
    try testing.expectError(CipherError.InvalidLength, decode(alloc, "AABBB AAB"));
}

test "baconian: empty input" {
    const alloc = testing.allocator;
    const e = try encode(alloc, "");
    defer alloc.free(e);
    try testing.expectEqual(@as(usize, 0), e.len);

    const d = try decode(alloc, "");
    defer alloc.free(d);
    try testing.expectEqual(@as(usize, 0), d.len);
}

test "baconian: extreme long round-trip" {
    const alloc = testing.allocator;
    const n: usize = 3000;
    const plain = try alloc.alloc(u8, n);
    defer alloc.free(plain);

    for (0..n) |idx| {
        plain[idx] = if (idx % 13 == 0) ' ' else @intCast('a' + (idx % 26));
    }

    const encoded = try encode(alloc, plain);
    defer alloc.free(encoded);
    const decoded = try decode(alloc, encoded);
    defer alloc.free(decoded);

    try testing.expectEqualSlices(u8, plain, decoded);
}
