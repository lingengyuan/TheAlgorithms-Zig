//! Run-Length Encoding - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_compression/run_length_encoding.py

const std = @import("std");
const testing = std.testing;

pub const RunLengthPair = struct {
    ch: u8,
    length: usize,
};

/// Encodes a byte string using run-length encoding.
/// Caller owns the returned slice.
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn runLengthEncode(allocator: std.mem.Allocator, text: []const u8) ![]RunLengthPair {
    if (text.len == 0) {
        return allocator.alloc(RunLengthPair, 0);
    }

    var run_count: usize = 1;
    for (1..text.len) |i| {
        if (text[i] != text[i - 1]) {
            run_count += 1;
        }
    }

    const encoded = try allocator.alloc(RunLengthPair, run_count);

    var out_index: usize = 0;
    var current = text[0];
    var count: usize = 1;

    for (text[1..]) |ch| {
        if (ch == current) {
            count += 1;
        } else {
            encoded[out_index] = .{ .ch = current, .length = count };
            out_index += 1;
            current = ch;
            count = 1;
        }
    }
    encoded[out_index] = .{ .ch = current, .length = count };

    return encoded;
}

/// Decodes run-length encoded pairs back to bytes.
/// Caller owns the returned slice.
///
/// Time complexity: O(total_output)
/// Space complexity: O(total_output)
pub fn runLengthDecode(allocator: std.mem.Allocator, encoded: []const RunLengthPair) ![]u8 {
    var total_len: usize = 0;
    for (encoded) |pair| {
        total_len += pair.length;
    }

    const out = try allocator.alloc(u8, total_len);
    var idx: usize = 0;
    for (encoded) |pair| {
        @memset(out[idx .. idx + pair.length], pair.ch);
        idx += pair.length;
    }

    return out;
}

test "run length encoding: python examples" {
    const alloc = testing.allocator;

    const encoded1 = try runLengthEncode(alloc, "AAAABBBCCDAA");
    defer alloc.free(encoded1);
    const expected1 = [_]RunLengthPair{
        .{ .ch = 'A', .length = 4 },
        .{ .ch = 'B', .length = 3 },
        .{ .ch = 'C', .length = 2 },
        .{ .ch = 'D', .length = 1 },
        .{ .ch = 'A', .length = 2 },
    };
    try testing.expectEqualSlices(RunLengthPair, &expected1, encoded1);

    const decoded1 = try runLengthDecode(alloc, encoded1);
    defer alloc.free(decoded1);
    try testing.expectEqualStrings("AAAABBBCCDAA", decoded1);

    const encoded2 = try runLengthEncode(alloc, "AAADDDDDDFFFCCCAAVVVV");
    defer alloc.free(encoded2);
    const decoded2 = try runLengthDecode(alloc, encoded2);
    defer alloc.free(decoded2);
    try testing.expectEqualStrings("AAADDDDDDFFFCCCAAVVVV", decoded2);
}

test "run length encoding: empty and extreme values" {
    const alloc = testing.allocator;

    const encoded_empty = try runLengthEncode(alloc, "");
    defer alloc.free(encoded_empty);
    try testing.expectEqual(@as(usize, 0), encoded_empty.len);

    const decoded_empty = try runLengthDecode(alloc, encoded_empty);
    defer alloc.free(decoded_empty);
    try testing.expectEqual(@as(usize, 0), decoded_empty.len);

    const large = "Z" ** 50_000;
    const encoded_large = try runLengthEncode(alloc, large);
    defer alloc.free(encoded_large);
    try testing.expectEqual(@as(usize, 1), encoded_large.len);
    try testing.expectEqual(@as(usize, 50_000), encoded_large[0].length);
}
