//! LZ77 Compression - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_compression/lz77.py

const std = @import("std");
const testing = std.testing;

pub const Lz77Error = error{
    InvalidWindowConfiguration,
    EmptyInput,
    InvalidTokenOffset,
};

pub const Token = struct {
    offset: usize,
    length: usize,
    indicator: u8,
};

pub const Lz77Compressor = struct {
    window_size: usize,
    lookahead_buffer_size: usize,
    search_buffer_size: usize,

    pub fn init(window_size: usize, lookahead_buffer_size: usize) Lz77Error!Lz77Compressor {
        if (window_size == 0 or lookahead_buffer_size == 0 or lookahead_buffer_size >= window_size) {
            return Lz77Error.InvalidWindowConfiguration;
        }
        return .{
            .window_size = window_size,
            .lookahead_buffer_size = lookahead_buffer_size,
            .search_buffer_size = window_size - lookahead_buffer_size,
        };
    }

    fn matchLengthFromIndex(
        self: *const Lz77Compressor,
        text: []const u8,
        window: []const u8,
        text_index: usize,
        window_index: usize,
    ) usize {
        _ = self;
        var matched: usize = 0;

        while (true) {
            const t_idx = text_index + matched;
            if (t_idx >= text.len - 1) break;

            const pos = window_index + matched;
            var window_char: u8 = undefined;
            if (pos < window.len) {
                window_char = window[pos];
            } else {
                const appended_idx = text_index + (pos - window.len);
                if (appended_idx >= text.len - 1) break;
                window_char = text[appended_idx];
            }

            if (text[t_idx] != window_char) break;
            matched += 1;
        }

        return matched;
    }

    fn findEncodingToken(self: *const Lz77Compressor, text: []const u8, search_buffer: []const u8) Lz77Error!Token {
        if (text.len == 0) {
            return Lz77Error.EmptyInput;
        }
        if (search_buffer.len == 0) {
            return Token{ .offset = 0, .length = 0, .indicator = text[0] };
        }

        var best_length: usize = 0;
        var best_offset: usize = 0;

        for (search_buffer, 0..) |character, i| {
            if (character == text[0]) {
                const found_offset = search_buffer.len - i;
                const found_length = self.matchLengthFromIndex(text, search_buffer, 0, i);
                if (found_length >= best_length) {
                    best_offset = found_offset;
                    best_length = found_length;
                }
            }
        }

        return Token{ .offset = best_offset, .length = best_length, .indicator = text[best_length] };
    }

    /// Compresses text using LZ77 token stream.
    /// Caller owns returned token slice.
    ///
    /// Time complexity: O(n * window)
    /// Space complexity: O(n)
    pub fn compress(self: *const Lz77Compressor, allocator: std.mem.Allocator, text: []const u8) ![]Token {
        if (text.len == 0) {
            return allocator.alloc(Token, 0);
        }

        var output = std.ArrayListUnmanaged(Token){};
        defer output.deinit(allocator);

        var cursor: usize = 0;
        while (cursor < text.len) {
            const search_start = if (cursor > self.search_buffer_size) cursor - self.search_buffer_size else 0;
            const search_buffer = text[search_start..cursor];
            const token = try self.findEncodingToken(text[cursor..], search_buffer);
            try output.append(allocator, token);
            cursor += token.length + 1;
        }

        return output.toOwnedSlice(allocator);
    }

    /// Decompresses LZ77 token stream back to original text.
    /// Caller owns returned byte slice.
    ///
    /// Time complexity: O(output)
    /// Space complexity: O(output)
    pub fn decompress(self: *const Lz77Compressor, allocator: std.mem.Allocator, tokens: []const Token) ![]u8 {
        _ = self;
        var output = std.ArrayListUnmanaged(u8){};
        defer output.deinit(allocator);

        for (tokens) |token| {
            var i: usize = 0;
            while (i < token.length) : (i += 1) {
                if (token.offset == 0 or token.offset > output.items.len) {
                    return Lz77Error.InvalidTokenOffset;
                }
                const ch = output.items[output.items.len - token.offset];
                try output.append(allocator, ch);
            }
            try output.append(allocator, token.indicator);
        }

        return output.toOwnedSlice(allocator);
    }
};

test "lz77: python examples compress" {
    const alloc = testing.allocator;
    const compressor = try Lz77Compressor.init(13, 6);

    const t1 = try compressor.compress(alloc, "ababcbababaa");
    defer alloc.free(t1);
    const e1 = [_]Token{
        .{ .offset = 0, .length = 0, .indicator = 'a' },
        .{ .offset = 0, .length = 0, .indicator = 'b' },
        .{ .offset = 2, .length = 2, .indicator = 'c' },
        .{ .offset = 4, .length = 3, .indicator = 'a' },
        .{ .offset = 2, .length = 2, .indicator = 'a' },
    };
    try testing.expectEqualSlices(Token, &e1, t1);

    const t2 = try compressor.compress(alloc, "aacaacabcabaaac");
    defer alloc.free(t2);
    const e2 = [_]Token{
        .{ .offset = 0, .length = 0, .indicator = 'a' },
        .{ .offset = 1, .length = 1, .indicator = 'c' },
        .{ .offset = 3, .length = 4, .indicator = 'b' },
        .{ .offset = 3, .length = 3, .indicator = 'a' },
        .{ .offset = 1, .length = 2, .indicator = 'c' },
    };
    try testing.expectEqualSlices(Token, &e2, t2);
}

test "lz77: python examples decompress and edge cases" {
    const alloc = testing.allocator;
    const compressor = try Lz77Compressor.init(13, 6);

    const tokens = [_]Token{
        .{ .offset = 0, .length = 0, .indicator = 'c' },
        .{ .offset = 0, .length = 0, .indicator = 'a' },
        .{ .offset = 0, .length = 0, .indicator = 'b' },
        .{ .offset = 0, .length = 0, .indicator = 'r' },
        .{ .offset = 3, .length = 1, .indicator = 'c' },
        .{ .offset = 2, .length = 1, .indicator = 'd' },
        .{ .offset = 7, .length = 4, .indicator = 'r' },
        .{ .offset = 3, .length = 5, .indicator = 'd' },
    };

    const out = try compressor.decompress(alloc, &tokens);
    defer alloc.free(out);
    try testing.expectEqualStrings("cabracadabrarrarrad", out);

    const empty = try compressor.compress(alloc, "");
    defer alloc.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const invalid = [_]Token{.{ .offset = 3, .length = 1, .indicator = 'a' }};
    try testing.expectError(Lz77Error.InvalidTokenOffset, compressor.decompress(alloc, &invalid));

    const repeated = "a" ** 20_000;
    const compressed = try compressor.compress(alloc, repeated);
    defer alloc.free(compressed);
    const restored = try compressor.decompress(alloc, compressed);
    defer alloc.free(restored);
    try testing.expectEqualStrings(repeated, restored);
}
