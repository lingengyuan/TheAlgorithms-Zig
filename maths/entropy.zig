//! Entropy - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/entropy.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const EntropyResult = struct {
    single: f64,
    pair: f64,
    conditional: f64,
};

const Alphabet = " abcdefghijklmnopqrstuvwxyz";

/// Analyzes text and returns rounded single-character entropy,
/// two-character entropy, and their difference using the Python reference logic.
/// Time complexity: O(n + |alphabet|²), Space complexity: O(n) in hash tables
pub fn calculateProb(allocator: Allocator, text: []const u8) !EntropyResult {
    if (text.len == 0) return error.EmptyText;

    var single_counts = std.AutoHashMap(u8, usize).init(allocator);
    defer single_counts.deinit();
    var pair_counts = std.AutoHashMap(u16, usize).init(allocator);
    defer pair_counts.deinit();

    try incrementU8(&single_counts, text[text.len - 1]);
    try incrementU16(&pair_counts, pairKey(' ', text[0]));
    for (0..text.len - 1) |i| {
        try incrementU8(&single_counts, text[i]);
        try incrementU16(&pair_counts, pairKey(text[i], text[i + 1]));
    }

    var single_total: usize = 0;
    var single_it = single_counts.iterator();
    while (single_it.next()) |entry| single_total += entry.value_ptr.*;

    var pair_total: usize = 0;
    var pair_it = pair_counts.iterator();
    while (pair_it.next()) |entry| pair_total += entry.value_ptr.*;

    var first_sum: f64 = 0.0;
    for (Alphabet) |ch| {
        if (single_counts.get(ch)) |count| {
            const prob = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(single_total));
            first_sum += prob * std.math.log2(prob);
        }
    }

    var second_sum: f64 = 0.0;
    for (Alphabet) |ch0| {
        for (Alphabet) |ch1| {
            if (pair_counts.get(pairKey(ch0, ch1))) |count| {
                const prob = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(pair_total));
                second_sum += prob * std.math.log2(prob);
            }
        }
    }

    const single_entropy = roundedWhole(-first_sum);
    const pair_entropy = roundedWhole(-second_sum);
    return .{
        .single = single_entropy,
        .pair = pair_entropy,
        .conditional = roundedWhole(pair_entropy - single_entropy),
    };
}

fn pairKey(a: u8, b: u8) u16 {
    return (@as(u16, a) << 8) | @as(u16, b);
}

fn incrementU8(map: *std.AutoHashMap(u8, usize), key: u8) !void {
    const gop = try map.getOrPut(key);
    if (!gop.found_existing) gop.value_ptr.* = 0;
    gop.value_ptr.* += 1;
}

fn incrementU16(map: *std.AutoHashMap(u16, usize), key: u16) !void {
    const gop = try map.getOrPut(key);
    if (!gop.found_existing) gop.value_ptr.* = 0;
    gop.value_ptr.* += 1;
}

fn roundedWhole(value: f64) f64 {
    return @round(value);
}

test "entropy: python reference examples" {
    const alloc = testing.allocator;
    const text1 =
        "Behind Winston's back the voice from the telescreen was still babbling and the overfulfilment";
    const result1 = try calculateProb(alloc, text1);
    try testing.expectEqual(@as(f64, 4.0), result1.single);
    try testing.expectEqual(@as(f64, 6.0), result1.pair);
    try testing.expectEqual(@as(f64, 2.0), result1.conditional);

    const text2 =
        "The Ministry of Truth—Minitrue, in Newspeak [Newspeak was the officialface in elegant lettering, the three";
    const result2 = try calculateProb(alloc, text2);
    try testing.expectEqual(@as(f64, 4.0), result2.single);
    try testing.expectEqual(@as(f64, 5.0), result2.pair);
    try testing.expectEqual(@as(f64, 1.0), result2.conditional);
}

test "entropy: long reference text and edge cases" {
    const alloc = testing.allocator;
    const text =
        "Had repulsive dashwoods suspicion sincerity but advantage now him. "
        ++ "Remark easily garret nor nay.  Civil those mrs enjoy shy fat merry. "
        ++ "You greatest jointure saw horrible. He private he on be imagine "
        ++ "suppose. Fertile beloved evident through no service elderly is. Blind "
        ++ "there if every no so at. Own neglected you preferred way sincerity "
        ++ "delivered his attempted. To of message cottage windows do besides "
        ++ "against uncivil.  Delightful unreserved impossible few estimating "
        ++ "men favourable see entreaties. She propriety immediate was improving. "
        ++ "He or entrance humoured likewise moderate. Much nor game son say "
        ++ "feel. Fat make met can must form into gate. Me we offending prevailed "
        ++ "discovery.";
    const result = try calculateProb(alloc, text);
    try testing.expectEqual(@as(f64, 4.0), result.single);
    try testing.expectEqual(@as(f64, 7.0), result.pair);
    try testing.expectEqual(@as(f64, 3.0), result.conditional);
    try testing.expectError(error.EmptyText, calculateProb(alloc, ""));
}

