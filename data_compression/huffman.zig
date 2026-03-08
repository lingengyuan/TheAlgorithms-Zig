//! Huffman Coding - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_compression/huffman.py

const std = @import("std");
const testing = std.testing;

pub const HuffmanError = error{
    EmptyInput,
};

const Node = struct {
    freq: usize,
    symbol: ?u8,
    left: ?*Node,
    right: ?*Node,
    serial: usize,
};

pub const CodeMap = struct {
    allocator: std.mem.Allocator,
    codes: [256]?[]u8,

    pub fn init(allocator: std.mem.Allocator) CodeMap {
        return .{
            .allocator = allocator,
            .codes = [_]?[]u8{null} ** 256,
        };
    }

    pub fn get(self: *const CodeMap, symbol: u8) ?[]const u8 {
        if (self.codes[symbol]) |code| return code;
        return null;
    }

    pub fn deinit(self: *CodeMap) void {
        for (self.codes) |entry| {
            if (entry) |code| {
                self.allocator.free(code);
            }
        }
    }
};

fn sortBytesStableByFreq(bytes: []u8, freq: [256]usize) void {
    if (bytes.len < 2) return;

    var i: usize = 1;
    while (i < bytes.len) : (i += 1) {
        const key = bytes[i];
        const key_freq = freq[key];

        var j = i;
        while (j > 0 and freq[bytes[j - 1]] > key_freq) : (j -= 1) {
            bytes[j] = bytes[j - 1];
        }
        bytes[j] = key;
    }
}

fn nodeLess(a: *const Node, b: *const Node) bool {
    if (a.freq != b.freq) return a.freq < b.freq;
    return a.serial < b.serial;
}

fn sortNodes(nodes: []const *Node, out: []*Node) void {
    @memcpy(out, nodes);
    if (out.len < 2) return;

    var i: usize = 1;
    while (i < out.len) : (i += 1) {
        const key = out[i];
        var j = i;
        while (j > 0 and nodeLess(key, out[j - 1])) : (j -= 1) {
            out[j] = out[j - 1];
        }
        out[j] = key;
    }
}

fn assignCodes(node: *const Node, path: *std.ArrayListUnmanaged(u8), map: *CodeMap, allocator: std.mem.Allocator) !void {
    if (node.symbol) |symbol| {
        const code = try allocator.alloc(u8, path.items.len);
        @memcpy(code, path.items);
        map.codes[symbol] = code;
        return;
    }

    if (node.left) |left| {
        try path.append(allocator, '0');
        try assignCodes(left, path, map, allocator);
        path.items.len -= 1;
    }

    if (node.right) |right| {
        try path.append(allocator, '1');
        try assignCodes(right, path, map, allocator);
        path.items.len -= 1;
    }
}

/// Builds Huffman code map for an input byte string, following the same
/// stable ordering behavior as the Python reference implementation.
/// Caller owns returned map and must call `deinit`.
///
/// Time complexity: O(n + k^2)
/// Space complexity: O(k)
/// where k is number of unique symbols.
pub fn buildCodeMap(allocator: std.mem.Allocator, text: []const u8) !CodeMap {
    if (text.len == 0) {
        return HuffmanError.EmptyInput;
    }

    var freq = [_]usize{0} ** 256;
    var seen = [_]bool{false} ** 256;

    var order = std.ArrayListUnmanaged(u8){};
    defer order.deinit(allocator);

    for (text) |ch| {
        freq[ch] += 1;
        if (!seen[ch]) {
            seen[ch] = true;
            try order.append(allocator, ch);
        }
    }

    sortBytesStableByFreq(order.items, freq);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var response = std.ArrayListUnmanaged(*Node){};
    defer response.deinit(allocator);

    var serial: usize = 0;
    for (order.items) |ch| {
        const leaf = try arena_allocator.create(Node);
        leaf.* = .{
            .freq = freq[ch],
            .symbol = ch,
            .left = null,
            .right = null,
            .serial = serial,
        };
        serial += 1;
        try response.append(allocator, leaf);
    }

    var scratch = try allocator.alloc(*Node, response.items.len);
    defer allocator.free(scratch);

    while (response.items.len > 1) {
        sortNodes(response.items, scratch[0..response.items.len]);
        @memcpy(response.items, scratch[0..response.items.len]);

        const left = response.orderedRemove(0);
        const right = response.orderedRemove(0);

        const parent = try arena_allocator.create(Node);
        parent.* = .{
            .freq = left.freq + right.freq,
            .symbol = null,
            .left = left,
            .right = right,
            .serial = serial,
        };
        serial += 1;

        try response.append(allocator, parent);
    }

    var map = CodeMap.init(allocator);
    errdefer map.deinit();

    var path = std.ArrayListUnmanaged(u8){};
    defer path.deinit(allocator);

    try assignCodes(response.items[0], &path, &map, allocator);
    return map;
}

/// Encodes text as space-separated Huffman bitstrings, matching the Python
/// reference output formatting for equivalent code maps.
/// Caller owns returned buffer.
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn huffmanEncode(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var map = try buildCodeMap(allocator, text);
    defer map.deinit();

    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(allocator);

    for (text, 0..) |ch, i| {
        if (map.get(ch)) |code| {
            try out.appendSlice(allocator, code);
        }
        if (i + 1 < text.len) {
            try out.append(allocator, ' ');
        }
    }

    return out.toOwnedSlice(allocator);
}

test "huffman: python mapping and encoding samples" {
    const alloc = testing.allocator;

    var map1 = try buildCodeMap(alloc, "ababc");
    defer map1.deinit();
    try testing.expectEqualStrings("11", map1.get('a').?);
    try testing.expectEqualStrings("0", map1.get('b').?);
    try testing.expectEqualStrings("10", map1.get('c').?);

    const enc1 = try huffmanEncode(alloc, "ababc");
    defer alloc.free(enc1);
    try testing.expectEqualStrings("11 0 11 0 10", enc1);

    var map2 = try buildCodeMap(alloc, "TheAlgorithms");
    defer map2.deinit();
    try testing.expectEqualStrings("1000", map2.get('T').?);
    try testing.expectEqualStrings("011", map2.get('h').?);
    try testing.expectEqualStrings("001", map2.get('m').?);

    const enc2 = try huffmanEncode(alloc, "TheAlgorithms");
    defer alloc.free(enc2);
    try testing.expectEqualStrings("1000 011 1001 1010 1011 1100 1101 1110 1111 000 011 001 010", enc2);
}

test "huffman: single symbol, empty input, and extreme text" {
    const alloc = testing.allocator;

    var single_map = try buildCodeMap(alloc, "aaaa");
    defer single_map.deinit();
    try testing.expectEqual(@as(usize, 0), single_map.get('a').?.len);

    const single_encoded = try huffmanEncode(alloc, "aaaa");
    defer alloc.free(single_encoded);
    try testing.expectEqualStrings("   ", single_encoded);

    try testing.expectError(HuffmanError.EmptyInput, buildCodeMap(alloc, ""));

    const large = "abcdeffedcba" ** 4_000;
    const large_encoded = try huffmanEncode(alloc, large);
    defer alloc.free(large_encoded);
    try testing.expect(large_encoded.len > large.len);

    const large_encoded_again = try huffmanEncode(alloc, large);
    defer alloc.free(large_encoded_again);
    try testing.expectEqualStrings(large_encoded, large_encoded_again);
}
