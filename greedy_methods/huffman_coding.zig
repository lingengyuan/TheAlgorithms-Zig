//! Huffman Coding - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_compression/huffman.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const HuffmanCode = struct {
    symbol: u8,
    code: []u8,
};

pub const HuffmanError = error{ UnknownSymbol, InvalidBit, InvalidEncoding };

const Node = struct {
    freq: u64,
    symbol: ?u8,
    min_symbol: u8,
    left: ?*Node,
    right: ?*Node,
};

const DecodeNode = struct {
    symbol: ?u8 = null,
    zero: ?*DecodeNode = null,
    one: ?*DecodeNode = null,
};

const MinHeap = struct {
    const Self = @This();

    allocator: Allocator,
    items: std.ArrayListUnmanaged(*Node) = .{},

    fn init(allocator: Allocator) Self {
        return .{ .allocator = allocator };
    }

    fn deinit(self: *Self) void {
        self.items.deinit(self.allocator);
    }

    fn less(a: *Node, b: *Node) bool {
        if (a.freq != b.freq) return a.freq < b.freq;
        return a.min_symbol < b.min_symbol;
    }

    fn push(self: *Self, node: *Node) !void {
        try self.items.append(self.allocator, node);
        var idx = self.items.items.len - 1;
        while (idx > 0) {
            const parent = (idx - 1) / 2;
            if (!less(self.items.items[idx], self.items.items[parent])) break;
            std.mem.swap(*Node, &self.items.items[idx], &self.items.items[parent]);
            idx = parent;
        }
    }

    fn pop(self: *Self) ?*Node {
        const len = self.items.items.len;
        if (len == 0) return null;

        const out = self.items.items[0];
        self.items.items[0] = self.items.items[len - 1];
        _ = self.items.pop();

        var idx: usize = 0;
        while (idx < self.items.items.len) {
            const left = 2 * idx + 1;
            const right = 2 * idx + 2;
            var smallest = idx;

            if (left < self.items.items.len and less(self.items.items[left], self.items.items[smallest])) {
                smallest = left;
            }
            if (right < self.items.items.len and less(self.items.items[right], self.items.items[smallest])) {
                smallest = right;
            }
            if (smallest == idx) break;
            std.mem.swap(*Node, &self.items.items[idx], &self.items.items[smallest]);
            idx = smallest;
        }

        return out;
    }
};

fn freeTree(allocator: Allocator, root: ?*Node) void {
    if (root) |node| {
        freeTree(allocator, node.left);
        freeTree(allocator, node.right);
        allocator.destroy(node);
    }
}

fn freeDecodeTree(allocator: Allocator, root: *DecodeNode) void {
    if (root.zero) |child| {
        freeDecodeTree(allocator, child);
        allocator.destroy(child);
    }
    if (root.one) |child| {
        freeDecodeTree(allocator, child);
        allocator.destroy(child);
    }
}

fn buildTree(allocator: Allocator, text: []const u8) !?*Node {
    if (text.len == 0) return null;

    var freq = [_]u64{0} ** 256;
    for (text) |ch| freq[ch] += 1;

    var heap = MinHeap.init(allocator);
    defer heap.deinit();

    for (freq, 0..) |f, i| {
        if (f == 0) continue;

        const node = try allocator.create(Node);
        node.* = .{
            .freq = f,
            .symbol = @intCast(i),
            .min_symbol = @intCast(i),
            .left = null,
            .right = null,
        };
        try heap.push(node);
    }

    if (heap.items.items.len == 0) return null;
    if (heap.items.items.len == 1) return heap.pop();

    while (heap.items.items.len > 1) {
        const left = heap.pop().?;
        const right = heap.pop().?;

        const parent = try allocator.create(Node);
        parent.* = .{
            .freq = left.freq + right.freq,
            .symbol = null,
            .min_symbol = @min(left.min_symbol, right.min_symbol),
            .left = left,
            .right = right,
        };

        try heap.push(parent);
    }

    return heap.pop();
}

fn collectCodes(
    allocator: Allocator,
    root: *const Node,
    path: *[256]u8,
    depth: usize,
    out: *std.ArrayListUnmanaged(HuffmanCode),
) !void {
    if (root.symbol) |symbol| {
        const code_len: usize = if (depth == 0) 1 else depth;
        const code = try allocator.alloc(u8, code_len);
        if (depth == 0) {
            code[0] = '0';
        } else {
            @memcpy(code, path[0..depth]);
        }

        try out.append(allocator, .{ .symbol = symbol, .code = code });
        return;
    }

    if (root.left) |left| {
        path[depth] = '0';
        try collectCodes(allocator, left, path, depth + 1, out);
    }

    if (root.right) |right| {
        path[depth] = '1';
        try collectCodes(allocator, right, path, depth + 1, out);
    }
}

fn codeLess(_: void, a: HuffmanCode, b: HuffmanCode) bool {
    return a.symbol < b.symbol;
}

/// Builds Huffman codes for bytes in `text`.
/// Caller owns returned code slices and the outer list.
/// Time complexity: O(n + sigma log sigma), space complexity: O(sigma)
pub fn buildHuffmanCodes(allocator: Allocator, text: []const u8) ![]HuffmanCode {
    if (text.len == 0) return try allocator.alloc(HuffmanCode, 0);

    const root = (try buildTree(allocator, text)).?;
    defer freeTree(allocator, root);

    var codes = std.ArrayListUnmanaged(HuffmanCode){};
    errdefer {
        for (codes.items) |entry| allocator.free(entry.code);
        codes.deinit(allocator);
    }

    var path: [256]u8 = undefined;
    try collectCodes(allocator, root, &path, 0, &codes);
    std.sort.heap(HuffmanCode, codes.items, {}, codeLess);

    return try codes.toOwnedSlice(allocator);
}

/// Releases nested allocations from `buildHuffmanCodes`.
pub fn freeHuffmanCodes(allocator: Allocator, codes: []HuffmanCode) void {
    for (codes) |entry| allocator.free(entry.code);
    allocator.free(codes);
}

/// Encodes `text` into a bitstring represented as bytes '0'/'1'.
/// Caller owns returned slice.
pub fn encodeText(
    allocator: Allocator,
    text: []const u8,
    codes: []const HuffmanCode,
) (Allocator.Error || HuffmanError)![]u8 {
    if (text.len == 0) return try allocator.alloc(u8, 0);

    var lookup = [_]?[]const u8{null} ** 256;
    for (codes) |entry| lookup[entry.symbol] = entry.code;

    var total_bits: usize = 0;
    for (text) |ch| {
        const code = lookup[ch] orelse return HuffmanError.UnknownSymbol;
        total_bits += code.len;
    }

    const out = try allocator.alloc(u8, total_bits);
    errdefer allocator.free(out);

    var pos: usize = 0;
    for (text) |ch| {
        const code = lookup[ch].?;
        @memcpy(out[pos .. pos + code.len], code);
        pos += code.len;
    }

    return out;
}

fn insertDecodeCode(allocator: Allocator, root: *DecodeNode, symbol: u8, code: []const u8) (Allocator.Error || HuffmanError)!void {
    if (code.len == 0) return HuffmanError.InvalidEncoding;

    var cur = root;
    for (code, 0..) |bit, idx| {
        if (cur.symbol != null) return HuffmanError.InvalidEncoding;

        const is_last = idx + 1 == code.len;
        switch (bit) {
            '0' => {
                if (cur.zero == null) {
                    const node = try allocator.create(DecodeNode);
                    node.* = .{};
                    cur.zero = node;
                }
                cur = cur.zero.?;
            },
            '1' => {
                if (cur.one == null) {
                    const node = try allocator.create(DecodeNode);
                    node.* = .{};
                    cur.one = node;
                }
                cur = cur.one.?;
            },
            else => return HuffmanError.InvalidBit,
        }

        if (is_last) {
            if (cur.symbol != null or cur.zero != null or cur.one != null) {
                return HuffmanError.InvalidEncoding;
            }
            cur.symbol = symbol;
        }
    }
}

/// Decodes a bitstring produced by `encodeText`.
/// Caller owns returned slice.
pub fn decodeBits(
    allocator: Allocator,
    bits: []const u8,
    codes: []const HuffmanCode,
) (Allocator.Error || HuffmanError)![]u8 {
    if (bits.len == 0) return try allocator.alloc(u8, 0);

    const root = try allocator.create(DecodeNode);
    root.* = .{};
    defer {
        freeDecodeTree(allocator, root);
        allocator.destroy(root);
    }

    for (codes) |entry| {
        try insertDecodeCode(allocator, root, entry.symbol, entry.code);
    }

    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    var cur = root;
    for (bits) |bit| {
        cur = switch (bit) {
            '0' => cur.zero orelse return HuffmanError.InvalidEncoding,
            '1' => cur.one orelse return HuffmanError.InvalidEncoding,
            else => return HuffmanError.InvalidBit,
        };

        if (cur.symbol) |symbol| {
            try out.append(allocator, symbol);
            cur = root;
        }
    }

    if (cur != root) return HuffmanError.InvalidEncoding;
    return try out.toOwnedSlice(allocator);
}

test "huffman coding: round trip" {
    const alloc = testing.allocator;
    const text = "beep boop beer!";

    const codes = try buildHuffmanCodes(alloc, text);
    defer freeHuffmanCodes(alloc, codes);

    const encoded = try encodeText(alloc, text, codes);
    defer alloc.free(encoded);

    const decoded = try decodeBits(alloc, encoded, codes);
    defer alloc.free(decoded);

    try testing.expectEqualStrings(text, decoded);
    try testing.expect(encoded.len > 0);
}

test "huffman coding: single symbol" {
    const alloc = testing.allocator;
    const text = "aaaaaa";

    const codes = try buildHuffmanCodes(alloc, text);
    defer freeHuffmanCodes(alloc, codes);

    try testing.expectEqual(@as(usize, 1), codes.len);
    try testing.expectEqual(@as(u8, 'a'), codes[0].symbol);
    try testing.expectEqualStrings("0", codes[0].code);

    const encoded = try encodeText(alloc, text, codes);
    defer alloc.free(encoded);

    const decoded = try decodeBits(alloc, encoded, codes);
    defer alloc.free(decoded);

    try testing.expectEqualStrings(text, decoded);
}

test "huffman coding: empty input" {
    const alloc = testing.allocator;
    const codes = try buildHuffmanCodes(alloc, "");
    defer freeHuffmanCodes(alloc, codes);

    const encoded = try encodeText(alloc, "", codes);
    defer alloc.free(encoded);
    try testing.expectEqual(@as(usize, 0), encoded.len);
}

test "huffman coding: unknown symbol on encode" {
    const alloc = testing.allocator;
    const codes = try buildHuffmanCodes(alloc, "ab");
    defer freeHuffmanCodes(alloc, codes);

    try testing.expectError(HuffmanError.UnknownSymbol, encodeText(alloc, "abc", codes));
}

test "huffman coding: invalid bit while decoding" {
    const alloc = testing.allocator;
    const codes = try buildHuffmanCodes(alloc, "ab");
    defer freeHuffmanCodes(alloc, codes);

    try testing.expectError(HuffmanError.InvalidBit, decodeBits(alloc, "01x", codes));
}

test "huffman coding: extreme skew distribution" {
    const alloc = testing.allocator;
    var buffer: [8192]u8 = undefined;
    @memset(&buffer, 'a');
    for (0..128) |i| {
        buffer[i * 4] = 'z';
    }

    const codes = try buildHuffmanCodes(alloc, &buffer);
    defer freeHuffmanCodes(alloc, codes);

    const encoded = try encodeText(alloc, &buffer, codes);
    defer alloc.free(encoded);

    const decoded = try decodeBits(alloc, encoded, codes);
    defer alloc.free(decoded);

    try testing.expectEqualStrings(&buffer, decoded);
}
