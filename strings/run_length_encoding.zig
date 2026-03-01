//! Run-Length Encoding (RLE) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_compression/run_length_encoding.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Run = struct {
    byte: u8,
    count: usize,
};

/// Encodes input text into (byte, run_length) pairs.
/// Caller owns returned slice.
/// Time complexity: O(n), space complexity: O(k) where k is number of runs.
pub fn runLengthEncode(allocator: Allocator, text: []const u8) ![]Run {
    if (text.len == 0) return try allocator.alloc(Run, 0);

    var out = std.ArrayListUnmanaged(Run){};
    defer out.deinit(allocator);

    var current = text[0];
    var count: usize = 1;
    for (text[1..]) |ch| {
        if (ch == current) {
            count += 1;
            continue;
        }
        try out.append(allocator, .{ .byte = current, .count = count });
        current = ch;
        count = 1;
    }
    try out.append(allocator, .{ .byte = current, .count = count });

    return try out.toOwnedSlice(allocator);
}

/// Decodes RLE runs back to the original byte string.
/// Returns `error.InvalidRunLength` for zero-length runs and
/// `error.CountOverflow` if total decoded length would overflow `usize`.
/// Caller owns returned slice.
pub fn runLengthDecode(allocator: Allocator, runs: []const Run) ![]u8 {
    var total_len: usize = 0;
    for (runs) |run| {
        if (run.count == 0) return error.InvalidRunLength;
        const with_overflow = @addWithOverflow(total_len, run.count);
        if (with_overflow[1] != 0) return error.CountOverflow;
        total_len = with_overflow[0];
    }

    const out = try allocator.alloc(u8, total_len);
    var pos: usize = 0;
    for (runs) |run| {
        @memset(out[pos .. pos + run.count], run.byte);
        pos += run.count;
    }
    return out;
}

test "run length encoding: basic examples" {
    const encoded = try runLengthEncode(testing.allocator, "AAAABBBCCDAA");
    defer testing.allocator.free(encoded);

    try testing.expectEqual(@as(usize, 5), encoded.len);
    try testing.expectEqual(Run{ .byte = 'A', .count = 4 }, encoded[0]);
    try testing.expectEqual(Run{ .byte = 'B', .count = 3 }, encoded[1]);
    try testing.expectEqual(Run{ .byte = 'C', .count = 2 }, encoded[2]);
    try testing.expectEqual(Run{ .byte = 'D', .count = 1 }, encoded[3]);
    try testing.expectEqual(Run{ .byte = 'A', .count = 2 }, encoded[4]);
}

test "run length encoding: decode examples" {
    const runs = [_]Run{
        .{ .byte = 'A', .count = 3 },
        .{ .byte = 'D', .count = 6 },
        .{ .byte = 'F', .count = 3 },
        .{ .byte = 'C', .count = 3 },
        .{ .byte = 'A', .count = 2 },
        .{ .byte = 'V', .count = 4 },
    };

    const decoded = try runLengthDecode(testing.allocator, &runs);
    defer testing.allocator.free(decoded);
    try testing.expectEqualStrings("AAADDDDDDFFFCCCAAVVVV", decoded);
}

test "run length encoding: round trip and empty input" {
    const text = "abccccccddeeeeeeeeeeeeeef";
    const encoded = try runLengthEncode(testing.allocator, text);
    defer testing.allocator.free(encoded);
    const decoded = try runLengthDecode(testing.allocator, encoded);
    defer testing.allocator.free(decoded);
    try testing.expectEqualStrings(text, decoded);

    const empty_encoded = try runLengthEncode(testing.allocator, "");
    defer testing.allocator.free(empty_encoded);
    try testing.expectEqual(@as(usize, 0), empty_encoded.len);

    const empty_decoded = try runLengthDecode(testing.allocator, &[_]Run{});
    defer testing.allocator.free(empty_decoded);
    try testing.expectEqual(@as(usize, 0), empty_decoded.len);
}

test "run length encoding: invalid run length rejected" {
    const invalid = [_]Run{
        .{ .byte = 'X', .count = 2 },
        .{ .byte = 'Y', .count = 0 },
    };
    try testing.expectError(error.InvalidRunLength, runLengthDecode(testing.allocator, &invalid));
}

test "run length encoding: extreme long run" {
    const n = 20_000;
    const text = try testing.allocator.alloc(u8, n);
    defer testing.allocator.free(text);
    @memset(text, 'z');

    const encoded = try runLengthEncode(testing.allocator, text);
    defer testing.allocator.free(encoded);
    try testing.expectEqual(@as(usize, 1), encoded.len);
    try testing.expectEqual(@as(usize, n), encoded[0].count);

    const decoded = try runLengthDecode(testing.allocator, encoded);
    defer testing.allocator.free(decoded);
    try testing.expectEqual(@as(usize, n), decoded.len);
    try testing.expect(decoded[0] == 'z' and decoded[n - 1] == 'z');
}

test "run length encoding: fuzz round trip" {
    return testing.fuzz({}, fuzzRunLengthRoundTrip, .{});
}

fn fuzzRunLengthRoundTrip(context: void, input: []const u8) anyerror!void {
    _ = context;

    const max_len = @min(input.len, @as(usize, 2048));
    const text = input[0..max_len];

    const encoded = try runLengthEncode(testing.allocator, text);
    defer testing.allocator.free(encoded);

    const decoded = try runLengthDecode(testing.allocator, encoded);
    defer testing.allocator.free(decoded);

    try testing.expectEqualSlices(u8, text, decoded);
}
