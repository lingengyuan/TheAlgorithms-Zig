//! Lempel-Ziv Bitstring Compression - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_compression/lempel_ziv.py
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_compression/lempel_ziv_decompress.py

const std = @import("std");
const testing = std.testing;

pub const LempelZivError = error{
    InvalidBitCharacter,
};

const Entry = struct {
    key: []const u8,
    value: []const u8,
};

fn isPowerOfTwo(value: usize) bool {
    return value != 0 and (value & (value - 1)) == 0;
}

fn validateBitString(bits: []const u8) LempelZivError!void {
    for (bits) |bit| {
        if (bit != '0' and bit != '1') {
            return LempelZivError.InvalidBitCharacter;
        }
    }
}

fn duplicateBytes(allocator: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    const dup = try allocator.alloc(u8, bytes.len);
    @memcpy(dup, bytes);
    return dup;
}

fn findEntry(entries: []const Entry, key: []const u8) ?usize {
    for (entries, 0..) |entry, i| {
        if (std.mem.eql(u8, entry.key, key)) {
            return i;
        }
    }
    return null;
}

fn setEntry(entries: *std.ArrayListUnmanaged(Entry), allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
    if (findEntry(entries.items, key)) |index| {
        entries.items[index].value = try duplicateBytes(allocator, value);
        return;
    }

    try entries.append(allocator, .{
        .key = try duplicateBytes(allocator, key),
        .value = try duplicateBytes(allocator, value),
    });
}

fn removeEntry(entries: *std.ArrayListUnmanaged(Entry), key: []const u8) void {
    if (findEntry(entries.items, key)) |index| {
        _ = entries.orderedRemove(index);
    }
}

fn prefixZeroToAllValues(entries: *std.ArrayListUnmanaged(Entry), allocator: std.mem.Allocator) !void {
    for (entries.items) |*entry| {
        const prefixed = try allocator.alloc(u8, entry.value.len + 1);
        prefixed[0] = '0';
        @memcpy(prefixed[1..], entry.value);
        entry.value = prefixed;
    }
}

fn prefixZeroToAllKeys(entries: *std.ArrayListUnmanaged(Entry), allocator: std.mem.Allocator) !void {
    for (entries.items) |*entry| {
        const prefixed = try allocator.alloc(u8, entry.key.len + 1);
        prefixed[0] = '0';
        @memcpy(prefixed[1..], entry.key);
        entry.key = prefixed;
    }
}

fn binaryString(allocator: std.mem.Allocator, value: usize) ![]const u8 {
    return std.fmt.allocPrint(allocator, "{b}", .{value});
}

/// Compresses bitstring data using the same dictionary evolution as Python's
/// reference `compress_data` implementation.
/// Caller owns the returned bitstring.
///
/// Time complexity: O(n^2)
/// Space complexity: O(n)
pub fn compressData(allocator: std.mem.Allocator, data_bits: []const u8) ![]u8 {
    try validateBitString(data_bits);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var lexicon = std.ArrayListUnmanaged(Entry){};
    defer lexicon.deinit(arena_allocator);

    try setEntry(&lexicon, arena_allocator, "0", "0");
    try setEntry(&lexicon, arena_allocator, "1", "1");

    var result = std.ArrayListUnmanaged(u8){};
    defer result.deinit(allocator);

    var current = std.ArrayListUnmanaged(u8){};
    defer current.deinit(allocator);

    var index: usize = lexicon.items.len;

    for (data_bits) |bit| {
        try current.append(allocator, bit);

        const current_index = findEntry(lexicon.items, current.items) orelse continue;
        const last_match_id = lexicon.items[current_index].value;
        try result.appendSlice(allocator, last_match_id);

        removeEntry(&lexicon, current.items);

        var key_zero = std.ArrayListUnmanaged(u8){};
        defer key_zero.deinit(allocator);
        try key_zero.appendSlice(allocator, current.items);
        try key_zero.append(allocator, '0');
        try setEntry(&lexicon, arena_allocator, key_zero.items, last_match_id);

        if (isPowerOfTwo(index)) {
            try prefixZeroToAllValues(&lexicon, arena_allocator);
        }

        var key_one = std.ArrayListUnmanaged(u8){};
        defer key_one.deinit(allocator);
        try key_one.appendSlice(allocator, current.items);
        try key_one.append(allocator, '1');

        const index_binary = try binaryString(arena_allocator, index);
        try setEntry(&lexicon, arena_allocator, key_one.items, index_binary);

        index += 1;
        current.clearRetainingCapacity();
    }

    while (current.items.len != 0 and findEntry(lexicon.items, current.items) == null) {
        try current.append(allocator, '0');
    }

    if (current.items.len != 0) {
        const final_index = findEntry(lexicon.items, current.items).?;
        try result.appendSlice(allocator, lexicon.items[final_index].value);
    }

    return result.toOwnedSlice(allocator);
}

/// Decompresses bitstring data using the same dictionary evolution as Python's
/// reference `decompress_data` implementation.
/// Caller owns the returned bitstring.
///
/// Time complexity: O(n^2)
/// Space complexity: O(n)
pub fn decompressData(allocator: std.mem.Allocator, data_bits: []const u8) ![]u8 {
    try validateBitString(data_bits);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var lexicon = std.ArrayListUnmanaged(Entry){};
    defer lexicon.deinit(arena_allocator);

    try setEntry(&lexicon, arena_allocator, "0", "0");
    try setEntry(&lexicon, arena_allocator, "1", "1");

    var result = std.ArrayListUnmanaged(u8){};
    defer result.deinit(allocator);

    var current = std.ArrayListUnmanaged(u8){};
    defer current.deinit(allocator);

    var index: usize = lexicon.items.len;

    for (data_bits) |bit| {
        try current.append(allocator, bit);

        const current_index = findEntry(lexicon.items, current.items) orelse continue;
        const last_match_id = lexicon.items[current_index].value;
        try result.appendSlice(allocator, last_match_id);

        var value_zero = std.ArrayListUnmanaged(u8){};
        defer value_zero.deinit(allocator);
        try value_zero.appendSlice(allocator, last_match_id);
        try value_zero.append(allocator, '0');
        try setEntry(&lexicon, arena_allocator, current.items, value_zero.items);

        if (isPowerOfTwo(index)) {
            try prefixZeroToAllKeys(&lexicon, arena_allocator);
        }

        var value_one = std.ArrayListUnmanaged(u8){};
        defer value_one.deinit(allocator);
        try value_one.appendSlice(allocator, last_match_id);
        try value_one.append(allocator, '1');

        const index_binary = try binaryString(arena_allocator, index);
        try setEntry(&lexicon, arena_allocator, index_binary, value_one.items);

        index += 1;
        current.clearRetainingCapacity();
    }

    return result.toOwnedSlice(allocator);
}

/// Removes the file-length gamma prefix from compressed bitstring, following the
/// reference Python implementation.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn removePrefix(data_bits: []const u8) []const u8 {
    var counter: usize = 0;
    while (counter < data_bits.len and data_bits[counter] == '0') : (counter += 1) {}

    if (counter >= data_bits.len) {
        return "";
    }

    const start = counter + counter + 1;
    if (start >= data_bits.len) {
        return "";
    }

    return data_bits[start..];
}

test "lempel ziv: python compression vectors" {
    const alloc = testing.allocator;

    const c1 = try compressData(alloc, "");
    defer alloc.free(c1);
    try testing.expectEqualStrings("", c1);

    const c2 = try compressData(alloc, "00110011");
    defer alloc.free(c2);
    try testing.expectEqualStrings("01001000100", c2);

    const c3 = try compressData(alloc, "010101010101");
    defer alloc.free(c3);
    try testing.expectEqualStrings("00110010001110", c3);

    const c4 = try compressData(alloc, "1111000011110000");
    defer alloc.free(c4);
    try testing.expectEqualStrings("110010000000110010000", c4);

    const c5 = try compressData(alloc, "1010010110110100010");
    defer alloc.free(c5);
    try testing.expectEqualStrings("100010111011010010011", c5);
}

test "lempel ziv: python decompression vectors and prefix removal" {
    const alloc = testing.allocator;

    const d1 = try decompressData(alloc, "00110010001110");
    defer alloc.free(d1);
    try testing.expectEqualStrings("010101010101", d1);

    const d2 = try decompressData(alloc, "110010000000110010000");
    defer alloc.free(d2);
    try testing.expectEqualStrings("11110000111100000", d2);

    try testing.expectEqualStrings("10101", removePrefix("0010110101"));
    try testing.expectEqualStrings("0110101", removePrefix("10110101"));
}

test "lempel ziv: extreme behavior and invalid input" {
    const alloc = testing.allocator;

    const repeated = "01" ** 10_000;
    const compressed = try compressData(alloc, repeated);
    defer alloc.free(compressed);

    const restored = try decompressData(alloc, compressed);
    defer alloc.free(restored);

    try testing.expect(restored.len >= repeated.len);
    try testing.expectEqualStrings(repeated, restored[0..repeated.len]);
    for (restored[repeated.len..]) |bit| {
        try testing.expect(bit == '0');
    }

    try testing.expectError(LempelZivError.InvalidBitCharacter, compressData(alloc, "010210"));
    try testing.expectError(LempelZivError.InvalidBitCharacter, decompressData(alloc, "10a1"));
}
