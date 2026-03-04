//! Caesar Decryption via Chi-Squared - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/decrypt_caesar_with_chi_squared.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const DecryptResult = struct {
    shift: usize,
    chi_squared: f64,
    decoded: []u8,

    pub fn deinit(self: DecryptResult, allocator: Allocator) void {
        allocator.free(self.decoded);
    }
};

const ALPHABET = "abcdefghijklmnopqrstuvwxyz";

const FREQ = [_]f64{
    0.08497, 0.01492, 0.02202, 0.04253, 0.11162, 0.02228, 0.02015, 0.06094, 0.07546, 0.00153, 0.01292, 0.04025, 0.02406,
    0.06749, 0.07507, 0.01929, 0.00095, 0.07587, 0.06327, 0.09356, 0.02758, 0.00978, 0.02560, 0.00150, 0.01994, 0.00077,
};

fn indexInAlphabet(ch: u8) ?usize {
    return std.mem.indexOfScalar(u8, ALPHABET, ch);
}

fn countOccurrences(decoded: []const u8, target: u8, case_sensitive: bool) usize {
    var count: usize = 0;
    for (decoded) |ch| {
        const cur = if (case_sensitive) std.ascii.toLower(ch) else ch;
        if (cur == target) count += 1;
    }
    return count;
}

/// Decrypts Caesar ciphertext by selecting shift with minimum chi-squared statistic.
/// Caller owns returned `decoded` buffer and must free it.
/// Time complexity: O(26 * n^2), Space complexity: O(n)
pub fn decryptCaesarWithChiSquared(allocator: Allocator, ciphertext: []const u8, case_sensitive: bool) !DecryptResult {
    var normalized = try allocator.alloc(u8, ciphertext.len);
    defer allocator.free(normalized);

    if (case_sensitive) {
        @memcpy(normalized, ciphertext);
    } else {
        for (ciphertext, 0..) |ch, i| normalized[i] = std.ascii.toLower(ch);
    }

    var best_shift: usize = 0;
    var best_chi: f64 = std.math.inf(f64);
    var best_decoded: ?[]u8 = null;

    for (0..ALPHABET.len) |shift| {
        const decoded = try allocator.alloc(u8, normalized.len);
        errdefer allocator.free(decoded);

        for (normalized, 0..) |letter, i| {
            const lookup = std.ascii.toLower(letter);
            if (indexInAlphabet(lookup)) |idx| {
                const new_key = (idx + ALPHABET.len - shift) % ALPHABET.len;
                const mapped = ALPHABET[new_key];
                decoded[i] = if (case_sensitive and letter >= 'A' and letter <= 'Z') std.ascii.toUpper(mapped) else mapped;
            } else {
                decoded[i] = letter;
            }
        }

        var chi: f64 = 0.0;
        for (decoded) |raw_letter| {
            const letter = if (case_sensitive) std.ascii.toLower(raw_letter) else raw_letter;
            if (indexInAlphabet(letter)) |idx| {
                const occ = countOccurrences(decoded, letter, case_sensitive);
                const occ_f = @as(f64, @floatFromInt(occ));
                const expected = FREQ[idx] * occ_f;
                if (expected > 0) {
                    const delta = occ_f - expected;
                    chi += (delta * delta) / expected;
                }
            }
        }

        var better = false;
        if (best_decoded == null) {
            better = true;
        } else if (chi < best_chi) {
            better = true;
        } else if (chi == best_chi and std.mem.order(u8, decoded, best_decoded.?) == .lt) {
            better = true;
        }

        if (better) {
            if (best_decoded) |old| allocator.free(old);
            best_decoded = decoded;
            best_chi = chi;
            best_shift = shift;
        } else {
            allocator.free(decoded);
        }
    }

    return DecryptResult{
        .shift = best_shift,
        .chi_squared = best_chi,
        .decoded = best_decoded.?,
    };
}

test "chi-squared caesar: python long sample" {
    const alloc = testing.allocator;

    const result = try decryptCaesarWithChiSquared(
        alloc,
        "dof pz aol jhlzhy jpwoly zv wvwbshy? pa pz avv lhzf av jyhjr!",
        false,
    );
    defer result.deinit(alloc);

    try testing.expectEqual(@as(usize, 7), result.shift);
    try testing.expectEqualStrings("why is the caesar cipher so popular? it is too easy to crack!", result.decoded);
}

test "chi-squared caesar: python short samples" {
    const alloc = testing.allocator;

    const a = try decryptCaesarWithChiSquared(alloc, "crybd cdbsxq", false);
    defer a.deinit(alloc);
    try testing.expectEqual(@as(usize, 10), a.shift);
    try testing.expectEqualStrings("short string", a.decoded);

    const b = try decryptCaesarWithChiSquared(alloc, "Crybd Cdbsxq", true);
    defer b.deinit(alloc);
    try testing.expectEqual(@as(usize, 10), b.shift);
    try testing.expectEqualStrings("Short String", b.decoded);
}

test "chi-squared caesar: extreme long input" {
    const alloc = testing.allocator;
    const n: usize = 15000;
    const text = try alloc.alloc(u8, n);
    defer alloc.free(text);

    for (text, 0..) |*ch, i| {
        ch.* = if (i % 11 == 0) ' ' else @as(u8, @intCast('a' + (i % 26)));
    }

    const result = try decryptCaesarWithChiSquared(alloc, text, false);
    defer result.deinit(alloc);

    try testing.expect(result.shift < 26);
    try testing.expectEqual(@as(usize, n), result.decoded.len);
}
