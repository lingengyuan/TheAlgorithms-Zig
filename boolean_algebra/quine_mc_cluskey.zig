//! Quine-McCluskey (Reference-Compatible Variant) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/boolean_algebra/quine_mc_cluskey.py

const std = @import("std");
const testing = std.testing;

fn freeStringArrayList(allocator: std.mem.Allocator, list: *std.ArrayListUnmanaged([]u8)) void {
    for (list.items) |item| allocator.free(item);
    list.deinit(allocator);
}

pub fn freeStringList(allocator: std.mem.Allocator, list: [][]u8) void {
    for (list) |item| allocator.free(item);
    allocator.free(list);
}

pub fn freeByteMatrix(allocator: std.mem.Allocator, matrix: [][]u8) void {
    for (matrix) |row| allocator.free(row);
    allocator.free(matrix);
}

fn formatPythonFloat(allocator: std.mem.Allocator, value: f64) std.mem.Allocator.Error![]u8 {
    if (std.math.isFinite(value) and std.math.floor(value) == value) {
        return std.fmt.allocPrint(allocator, "{d:.1}", .{value});
    }
    return std.fmt.allocPrint(allocator, "{d}", .{value});
}

/// Compares two equal-length bitstrings and replaces at most one differing position with `_`.
/// Returns `null` if they differ in more than one position.
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn compareString(
    allocator: std.mem.Allocator,
    string1: []const u8,
    string2: []const u8,
) std.mem.Allocator.Error!?[]u8 {
    if (string1.len != string2.len) return null;

    const out = try allocator.dupe(u8, string1);
    var diff_count: usize = 0;
    for (out, string2) |*left, right| {
        if (left.* == right) continue;
        diff_count += 1;
        left.* = '_';
    }

    if (diff_count > 1) {
        allocator.free(out);
        return null;
    }
    return out;
}

/// Python-reference-compatible `check` behavior, including its current reduction semantics.
///
/// Time complexity: O(n^2 * m) for pairwise comparisons in each iteration.
/// Space complexity: O(n * m)
pub fn check(
    allocator: std.mem.Allocator,
    binary: []const []const u8,
) std.mem.Allocator.Error![][]u8 {
    var prime_implicants = std.ArrayListUnmanaged([]u8){};
    errdefer freeStringArrayList(allocator, &prime_implicants);

    var current = std.ArrayListUnmanaged([]u8){};
    errdefer freeStringArrayList(allocator, &current);
    for (binary) |item| {
        try current.append(allocator, try allocator.dupe(u8, item));
    }

    while (true) {
        const check1 = try allocator.alloc(u8, current.items.len);
        defer allocator.free(check1);
        @memset(check1, '$');

        var temp_count: usize = 0;
        for (current.items, 0..) |lhs, i| {
            var j = i + 1;
            while (j < current.items.len) : (j += 1) {
                const compared = try compareString(allocator, lhs, current.items[j]);
                if (compared) |value| {
                    allocator.free(value);
                } else {
                    check1[i] = '*';
                    check1[j] = '*';
                    temp_count += 1;
                }
            }
        }

        for (check1, 0..) |marker, i| {
            if (marker == '$') {
                try prime_implicants.append(allocator, try allocator.dupe(u8, current.items[i]));
            }
        }

        if (temp_count == 0) {
            freeStringArrayList(allocator, &current);
            return prime_implicants.toOwnedSlice(allocator);
        }

        freeStringArrayList(allocator, &current);
        try current.append(allocator, try allocator.dupe(u8, "X"));
    }
}

/// Converts decimal minterms to binary-like strings using the Python reference's float semantics.
///
/// Time complexity: O(k * v), k=minterms count, v=variable count
/// Space complexity: O(k * v)
pub fn decimalToBinary(
    allocator: std.mem.Allocator,
    no_of_variable: usize,
    minterms: []const f64,
) std.mem.Allocator.Error![][]u8 {
    var out = std.ArrayListUnmanaged([]u8){};
    errdefer freeStringArrayList(allocator, &out);

    for (minterms) |start_value| {
        var value = start_value;
        var built = try allocator.dupe(u8, "");
        errdefer allocator.free(built);

        for (0..no_of_variable) |_| {
            const remainder = @mod(value, 2.0);
            const rem_text = try formatPythonFloat(allocator, remainder);
            defer allocator.free(rem_text);

            const merged = try std.fmt.allocPrint(allocator, "{s}{s}", .{ rem_text, built });
            allocator.free(built);
            built = merged;
            value = @floor(value / 2.0);
        }

        try out.append(allocator, built);
    }

    return out.toOwnedSlice(allocator);
}

/// Returns true when Hamming distance between strings equals `count`.
pub fn isForTable(string1: []const u8, string2: []const u8, count: usize) bool {
    if (string1.len != string2.len) return false;

    var count_n: usize = 0;
    for (string1, string2) |left, right| {
        if (left != right) count_n += 1;
    }
    return count_n == count;
}

/// Builds prime implicant coverage chart.
///
/// Time complexity: O(p * b * m), p=prime implicants, b=binary terms, m=term width
/// Space complexity: O(p * b)
pub fn primeImplicantChart(
    allocator: std.mem.Allocator,
    prime_implicants: []const []const u8,
    binary: []const []const u8,
) std.mem.Allocator.Error![][]u8 {
    const chart = try allocator.alloc([]u8, prime_implicants.len);
    errdefer allocator.free(chart);

    var built_rows: usize = 0;
    errdefer {
        for (chart[0..built_rows]) |row| allocator.free(row);
    }

    for (prime_implicants, 0..) |implicant, i| {
        chart[i] = try allocator.alloc(u8, binary.len);
        built_rows += 1;
        @memset(chart[i], 0);

        const underscore_count = std.mem.count(u8, implicant, "_");
        for (binary, 0..) |term, j| {
            if (isForTable(implicant, term, underscore_count)) {
                chart[i][j] = 1;
            }
        }
    }

    return chart;
}

/// Selects essential prime implicants and then greedily covers remaining columns.
/// The input chart is modified in place, matching Python behavior.
///
/// Time complexity: O(r * c * (r + c))
/// Space complexity: O(r + selected)
pub fn selection(
    allocator: std.mem.Allocator,
    chart: []([]u8),
    prime_implicants: []const []const u8,
) std.mem.Allocator.Error![][]u8 {
    var selected = std.ArrayListUnmanaged([]u8){};
    errdefer freeStringArrayList(allocator, &selected);

    if (chart.len == 0) return selected.toOwnedSlice(allocator);
    const cols = chart[0].len;

    const select = try allocator.alloc(u8, chart.len);
    defer allocator.free(select);
    @memset(select, 0);

    for (0..cols) |col| {
        var count: usize = 0;
        var rem: usize = 0;
        for (chart, 0..) |row, r| {
            if (row[col] == 1) {
                count += 1;
                rem = r;
            }
        }
        if (count == 1) select[rem] = 1;
    }

    for (select, 0..) |take, i| {
        if (take != 1) continue;
        for (0..cols) |col| {
            if (chart[i][col] != 1) continue;
            for (chart) |row| row[col] = 0;
        }
        try selected.append(allocator, try allocator.dupe(u8, prime_implicants[i]));
    }

    while (true) {
        var max_n: usize = 0;
        var rem: usize = 0;

        for (chart, 0..) |row, r| {
            var row_count: usize = 0;
            for (row) |cell| {
                if (cell == 1) row_count += 1;
            }
            if (row_count > max_n) {
                max_n = row_count;
                rem = r;
            }
        }

        if (max_n == 0) return selected.toOwnedSlice(allocator);

        try selected.append(allocator, try allocator.dupe(u8, prime_implicants[rem]));
        for (0..cols) |col| {
            if (chart[rem][col] != 1) continue;
            for (chart) |row| row[col] = 0;
        }
    }
}

test "quine mc cluskey: compare string doctests" {
    const alloc = testing.allocator;

    const c1 = (try compareString(alloc, "0010", "0110")).?;
    defer alloc.free(c1);
    try testing.expectEqualStrings("0_10", c1);

    const c2 = try compareString(alloc, "0110", "1101");
    try testing.expect(c2 == null);
}

test "quine mc cluskey: check doctests and edge" {
    const alloc = testing.allocator;

    const one = [_][]const u8{"0.00.01.5"};
    const r1 = try check(alloc, &one);
    defer freeStringList(alloc, r1);
    try testing.expectEqual(@as(usize, 1), r1.len);
    try testing.expectEqualStrings("0.00.01.5", r1[0]);

    const two = [_][]const u8{ "000", "001" };
    const r2 = try check(alloc, &two);
    defer freeStringList(alloc, r2);
    try testing.expectEqual(@as(usize, 2), r2.len);
    try testing.expectEqualStrings("000", r2[0]);
    try testing.expectEqualStrings("001", r2[1]);
}

test "quine mc cluskey: decimal to binary doctest and boundary" {
    const alloc = testing.allocator;

    const values = [_]f64{1.5};
    const out = try decimalToBinary(alloc, 3, &values);
    defer freeStringList(alloc, out);
    try testing.expectEqual(@as(usize, 1), out.len);
    try testing.expectEqualStrings("0.00.01.5", out[0]);

    const none = [_]f64{2.0};
    const out_zero = try decimalToBinary(alloc, 0, &none);
    defer freeStringList(alloc, out_zero);
    try testing.expectEqualStrings("", out_zero[0]);
}

test "quine mc cluskey: table helpers doctests" {
    try testing.expect(isForTable("__1", "011", 2));
    try testing.expect(!isForTable("01_", "001", 1));
}

test "quine mc cluskey: chart and selection doctests" {
    const alloc = testing.allocator;

    const prime = [_][]const u8{"0.00.01.5"};
    const binary = [_][]const u8{"0.00.01.5"};
    const chart = try primeImplicantChart(alloc, &prime, &binary);
    defer freeByteMatrix(alloc, chart);

    try testing.expectEqual(@as(usize, 1), chart.len);
    try testing.expectEqual(@as(usize, 1), chart[0].len);
    try testing.expectEqual(@as(u8, 1), chart[0][0]);

    var mutable_row = [_]u8{1};
    var mutable_chart = [_][]u8{mutable_row[0..]};
    const selected = try selection(alloc, &mutable_chart, &prime);
    defer freeStringList(alloc, selected);
    try testing.expectEqual(@as(usize, 1), selected.len);
    try testing.expectEqualStrings("0.00.01.5", selected[0]);
}

test "quine mc cluskey: extreme sparse chart selection" {
    const alloc = testing.allocator;

    var row = [_]u8{ 0, 0, 0 };
    var chart = [_][]u8{row[0..]};
    const prime = [_][]const u8{"X"};
    const selected = try selection(alloc, &chart, &prime);
    defer freeStringList(alloc, selected);
    try testing.expectEqual(@as(usize, 0), selected.len);
}
