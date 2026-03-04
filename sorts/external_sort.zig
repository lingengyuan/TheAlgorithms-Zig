//! External Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/external_sort.py

const std = @import("std");
const testing = std.testing;

pub const ExternalSortError = error{InvalidBlockSize};

fn lessString(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.order(u8, lhs, rhs) == .lt;
}

/// Parses memory strings like "100M", "32k", "1G", or plain integer bytes.
pub fn parseMemory(input: []const u8) !usize {
    if (input.len == 0) return error.InvalidCharacter;

    const last = input[input.len - 1];
    if (last == 'k' or last == 'K') {
        const base = try std.fmt.parseInt(usize, input[0 .. input.len - 1], 10);
        return base * 1024;
    } else if (last == 'm' or last == 'M') {
        const base = try std.fmt.parseInt(usize, input[0 .. input.len - 1], 10);
        return base * 1024 * 1024;
    } else if (last == 'g' or last == 'G') {
        const base = try std.fmt.parseInt(usize, input[0 .. input.len - 1], 10);
        return base * 1024 * 1024 * 1024;
    } else {
        return try std.fmt.parseInt(usize, input, 10);
    }
}

/// Mirrors Python's block estimate: (total / block_size) + 1.
pub fn getNumberBlocks(total_size: usize, block_size: usize) ExternalSortError!usize {
    if (block_size == 0) return error.InvalidBlockSize;
    return (total_size / block_size) + 1;
}

/// Sorts lines with block splitting + N-way merge (in-memory version).
/// Caller owns returned outer slice; inner slices borrow from input.
/// Time complexity: O(n log b + n*b) where b=#blocks (scan-based merge)
/// Space complexity: O(n)
pub fn externalSortLines(
    allocator: std.mem.Allocator,
    lines: []const []const u8,
    block_size: usize,
) (ExternalSortError || std.mem.Allocator.Error)![][]const u8 {
    if (block_size == 0) return error.InvalidBlockSize;
    if (lines.len == 0) return try allocator.alloc([]const u8, 0);

    const num_blocks = try getNumberBlocks(lines.len - 1, block_size);
    const blocks = try allocator.alloc([][]const u8, num_blocks);
    defer allocator.free(blocks);

    var produced_blocks: usize = 0;
    errdefer {
        for (0..produced_blocks) |i| allocator.free(blocks[i]);
    }

    for (0..num_blocks) |bi| {
        const start = bi * block_size;
        if (start >= lines.len) {
            blocks[bi] = try allocator.alloc([]const u8, 0);
            produced_blocks += 1;
            continue;
        }
        const end = @min(start + block_size, lines.len);
        const block = try allocator.alloc([]const u8, end - start);
        @memcpy(block, lines[start..end]);
        std.mem.sort([]const u8, block, {}, lessString);
        blocks[bi] = block;
        produced_blocks += 1;
    }

    defer {
        for (blocks) |b| allocator.free(b);
    }

    const positions = try allocator.alloc(usize, num_blocks);
    defer allocator.free(positions);
    @memset(positions, 0);

    const out = try allocator.alloc([]const u8, lines.len);
    var out_i: usize = 0;
    while (out_i < out.len) : (out_i += 1) {
        var best_idx: ?usize = null;
        var best_line: []const u8 = undefined;

        for (blocks, 0..) |block, bi| {
            if (positions[bi] >= block.len) continue;
            const candidate = block[positions[bi]];
            if (best_idx == null or std.mem.order(u8, candidate, best_line) == .lt) {
                best_idx = bi;
                best_line = candidate;
            }
        }

        const chosen = best_idx.?;
        out[out_i] = best_line;
        positions[chosen] += 1;
    }

    return out;
}

test "external sort: parse memory examples" {
    try testing.expectEqual(@as(usize, 1024), try parseMemory("1k"));
    try testing.expectEqual(@as(usize, 2 * 1024 * 1024), try parseMemory("2M"));
    try testing.expectEqual(@as(usize, 3 * 1024 * 1024 * 1024), try parseMemory("3g"));
    try testing.expectEqual(@as(usize, 256), try parseMemory("256"));
}

test "external sort: block estimate and basic sorting" {
    try testing.expectEqual(@as(usize, 4), try getNumberBlocks(300, 100));
    try testing.expectError(error.InvalidBlockSize, getNumberBlocks(10, 0));

    const alloc = testing.allocator;
    const lines = [_][]const u8{ "zeta\n", "alpha\n", "delta\n", "beta\n", "gamma\n" };
    const out = try externalSortLines(alloc, &lines, 2);
    defer alloc.free(out);
    try testing.expectEqualStrings("alpha\n", out[0]);
    try testing.expectEqualStrings("beta\n", out[1]);
    try testing.expectEqualStrings("delta\n", out[2]);
    try testing.expectEqualStrings("gamma\n", out[3]);
    try testing.expectEqualStrings("zeta\n", out[4]);
}

test "external sort: edge and extreme cases" {
    const alloc = testing.allocator;

    const empty = [_][]const u8{};
    const sorted_empty = try externalSortLines(alloc, &empty, 4);
    defer alloc.free(sorted_empty);
    try testing.expectEqual(@as(usize, 0), sorted_empty.len);

    try testing.expectError(error.InvalidBlockSize, externalSortLines(alloc, &[_][]const u8{"a"}, 0));

    const n: usize = 60_000;
    const lines = try alloc.alloc([]const u8, n);
    defer alloc.free(lines);
    const backing = try alloc.alloc(u8, n * 10);
    defer alloc.free(backing);

    for (0..n) |i| {
        const value = n - 1 - i;
        const start = i * 10;
        backing[start] = 'l';
        backing[start + 1] = @as(u8, '0' + @as(u8, @intCast((value / 10000) % 10)));
        backing[start + 2] = @as(u8, '0' + @as(u8, @intCast((value / 1000) % 10)));
        backing[start + 3] = @as(u8, '0' + @as(u8, @intCast((value / 100) % 10)));
        backing[start + 4] = @as(u8, '0' + @as(u8, @intCast((value / 10) % 10)));
        backing[start + 5] = @as(u8, '0' + @as(u8, @intCast(value % 10)));
        backing[start + 6] = '_';
        backing[start + 7] = 'x';
        backing[start + 8] = '\n';
        backing[start + 9] = 0; // sentinel not part of slice
        lines[i] = backing[start .. start + 9];
    }

    const sorted = try externalSortLines(alloc, lines, 1024);
    defer alloc.free(sorted);
    for (1..sorted.len) |i| {
        try testing.expect(std.mem.order(u8, sorted[i - 1], sorted[i]) != .gt);
    }
}
