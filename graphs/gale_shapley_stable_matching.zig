//! Gale-Shapley Stable Matching - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/graphs/gale_shapley_bigraph.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Computes donor-optimal stable matching.
/// `donor_pref` and `recipient_pref` must both be n x n preference matrices
/// with values in [0, n) and no duplicates per row.
/// Returns array `match` where `match[donor] = recipient`.
/// Time complexity: O(n^2), Space complexity: O(n^2)
pub fn stableMatching(
    allocator: Allocator,
    donor_pref: []const []const usize,
    recipient_pref: []const []const usize,
) ![]usize {
    const n = donor_pref.len;
    if (n != recipient_pref.len) return error.InvalidInput;
    if (n == 0) return try allocator.alloc(usize, 0);

    try validatePreferenceMatrix(allocator, donor_pref, n);
    try validatePreferenceMatrix(allocator, recipient_pref, n);

    const recipient_rank = try allocator.alloc(usize, n * n);
    defer allocator.free(recipient_rank);
    for (recipient_pref, 0..) |row, recipient| {
        for (row, 0..) |donor, rank| {
            recipient_rank[recipient * n + donor] = rank;
        }
    }

    const none = std.math.maxInt(usize);
    const donor_match = try allocator.alloc(usize, n);
    defer allocator.free(donor_match);
    const recipient_match = try allocator.alloc(usize, n);
    defer allocator.free(recipient_match);
    const next_choice = try allocator.alloc(usize, n);
    defer allocator.free(next_choice);

    @memset(donor_match, none);
    @memset(recipient_match, none);
    @memset(next_choice, 0);

    var queue = std.ArrayListUnmanaged(usize){};
    defer queue.deinit(allocator);
    for (0..n) |donor| try queue.append(allocator, donor);

    var head: usize = 0;
    while (head < queue.items.len) {
        const donor = queue.items[head];
        head += 1;

        if (next_choice[donor] >= n) return error.InvalidInput;
        const recipient = donor_pref[donor][next_choice[donor]];
        next_choice[donor] += 1;

        const current = recipient_match[recipient];
        if (current == none) {
            recipient_match[recipient] = donor;
            donor_match[donor] = recipient;
            continue;
        }

        const rank_new = recipient_rank[recipient * n + donor];
        const rank_old = recipient_rank[recipient * n + current];
        if (rank_new < rank_old) {
            recipient_match[recipient] = donor;
            donor_match[donor] = recipient;
            donor_match[current] = none;
            try queue.append(allocator, current);
        } else {
            try queue.append(allocator, donor);
        }
    }

    const out = try allocator.alloc(usize, n);
    for (0..n) |donor| {
        if (donor_match[donor] == none) return error.InternalInvariantBroken;
        out[donor] = donor_match[donor];
    }
    return out;
}

fn validatePreferenceMatrix(
    allocator: Allocator,
    matrix: []const []const usize,
    n: usize,
) !void {
    for (matrix) |row| {
        if (row.len != n) return error.InvalidInput;
        const seen = try allocator.alloc(bool, n);
        defer allocator.free(seen);
        @memset(seen, false);

        for (row) |value| {
            if (value >= n) return error.InvalidInput;
            if (seen[value]) return error.InvalidInput;
            seen[value] = true;
        }
    }
}

test "stable matching: python sample" {
    const alloc = testing.allocator;
    const donor_pref = [_][]const usize{
        &[_]usize{ 0, 1, 3, 2 },
        &[_]usize{ 0, 2, 3, 1 },
        &[_]usize{ 1, 0, 2, 3 },
        &[_]usize{ 0, 3, 1, 2 },
    };
    const recipient_pref = [_][]const usize{
        &[_]usize{ 3, 1, 2, 0 },
        &[_]usize{ 3, 1, 0, 2 },
        &[_]usize{ 0, 3, 1, 2 },
        &[_]usize{ 1, 0, 3, 2 },
    };

    const match = try stableMatching(alloc, &donor_pref, &recipient_pref);
    defer alloc.free(match);
    try testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 3, 0 }, match);
}

test "stable matching: invalid input matrices" {
    const alloc = testing.allocator;
    const donor_pref = [_][]const usize{
        &[_]usize{ 0, 1 },
        &[_]usize{ 1, 0 },
    };
    const bad_size = [_][]const usize{
        &[_]usize{ 0, 1 },
    };
    try testing.expectError(error.InvalidInput, stableMatching(alloc, &donor_pref, &bad_size));

    const bad_value = [_][]const usize{
        &[_]usize{ 0, 2 },
        &[_]usize{ 1, 0 },
    };
    try testing.expectError(error.InvalidInput, stableMatching(alloc, &donor_pref, &bad_value));
}

test "stable matching: extreme identity preferences" {
    const alloc = testing.allocator;
    const n: usize = 96;

    const donor_rows_mut = try alloc.alloc([]usize, n);
    defer {
        for (donor_rows_mut) |row| alloc.free(row);
        alloc.free(donor_rows_mut);
    }
    const recipient_rows_mut = try alloc.alloc([]usize, n);
    defer {
        for (recipient_rows_mut) |row| alloc.free(row);
        alloc.free(recipient_rows_mut);
    }

    for (0..n) |_| {}
    for (0..n) |i| {
        donor_rows_mut[i] = try alloc.alloc(usize, n);
        recipient_rows_mut[i] = try alloc.alloc(usize, n);
        for (0..n) |j| {
            donor_rows_mut[i][j] = j;
            recipient_rows_mut[i][j] = j;
        }
    }

    const donor_pref = try alloc.alloc([]const usize, n);
    defer alloc.free(donor_pref);
    const recipient_pref = try alloc.alloc([]const usize, n);
    defer alloc.free(recipient_pref);
    for (0..n) |i| {
        donor_pref[i] = donor_rows_mut[i];
        recipient_pref[i] = recipient_rows_mut[i];
    }

    const match = try stableMatching(alloc, donor_pref, recipient_pref);
    defer alloc.free(match);

    try testing.expectEqual(n, match.len);
    for (match, 0..) |recipient, donor| {
        try testing.expectEqual(donor, recipient);
    }
}
