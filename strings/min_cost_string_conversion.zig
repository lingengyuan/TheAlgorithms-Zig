//! Min Cost String Conversion - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/min_cost_string_conversion.py

const std = @import("std");
const testing = std.testing;

pub const Operation = union(enum) {
    zero,
    copy: u8,
    replace: struct { from: u8, to: u8 },
    delete: u8,
    insert: u8,
};

pub const TransformTables = struct {
    rows: usize,
    cols: usize,
    costs: []i64,
    ops: []Operation,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *TransformTables) void {
        self.allocator.free(self.costs);
        self.allocator.free(self.ops);
    }

    pub fn costAt(self: TransformTables, i: usize, j: usize) i64 {
        return self.costs[index(self.cols, i, j)];
    }

    pub fn opAt(self: TransformTables, i: usize, j: usize) Operation {
        return self.ops[index(self.cols, i, j)];
    }
};

fn index(cols: usize, i: usize, j: usize) usize {
    return i * cols + j;
}

fn operationToString(op: Operation, buffer: *[3]u8) []const u8 {
    return switch (op) {
        .zero => "0",
        .copy => |c| blk: {
            buffer[0] = 'C';
            buffer[1] = c;
            break :blk buffer[0..2];
        },
        .replace => |r| blk: {
            buffer[0] = 'R';
            buffer[1] = r.from;
            buffer[2] = r.to;
            break :blk buffer[0..3];
        },
        .delete => |c| blk: {
            buffer[0] = 'D';
            buffer[1] = c;
            break :blk buffer[0..2];
        },
        .insert => |c| blk: {
            buffer[0] = 'I';
            buffer[1] = c;
            break :blk buffer[0..2];
        },
    };
}

/// Computes cost and operation dynamic programming tables.
/// Time complexity: O(m * n), Space complexity: O(m * n)
pub fn computeTransformTables(
    allocator: std.mem.Allocator,
    sourceString: []const u8,
    destinationString: []const u8,
    copyCost: i64,
    replaceCost: i64,
    deleteCost: i64,
    insertCost: i64,
) !TransformTables {
    const rows = sourceString.len + 1;
    const cols = destinationString.len + 1;
    const size = rows * cols;

    const costs = try allocator.alloc(i64, size);
    errdefer allocator.free(costs);

    const ops = try allocator.alloc(Operation, size);
    errdefer allocator.free(ops);

    for (0..size) |idx| {
        costs[idx] = 0;
        ops[idx] = .zero;
    }

    for (1..rows) |i| {
        costs[index(cols, i, 0)] = @as(i64, @intCast(i)) * deleteCost;
        ops[index(cols, i, 0)] = .{ .delete = sourceString[i - 1] };
    }

    for (1..cols) |j| {
        costs[index(cols, 0, j)] = @as(i64, @intCast(j)) * insertCost;
        ops[index(cols, 0, j)] = .{ .insert = destinationString[j - 1] };
    }

    for (1..rows) |i| {
        for (1..cols) |j| {
            const diag_idx = index(cols, i - 1, j - 1);
            const up_idx = index(cols, i - 1, j);
            const left_idx = index(cols, i, j - 1);
            const cur_idx = index(cols, i, j);

            var best_cost: i64 = undefined;
            var best_op: Operation = undefined;

            if (sourceString[i - 1] == destinationString[j - 1]) {
                best_cost = costs[diag_idx] + copyCost;
                best_op = .{ .copy = sourceString[i - 1] };
            } else {
                best_cost = costs[diag_idx] + replaceCost;
                best_op = .{ .replace = .{ .from = sourceString[i - 1], .to = destinationString[j - 1] } };
            }

            const delete_candidate = costs[up_idx] + deleteCost;
            if (delete_candidate < best_cost) {
                best_cost = delete_candidate;
                best_op = .{ .delete = sourceString[i - 1] };
            }

            const insert_candidate = costs[left_idx] + insertCost;
            if (insert_candidate < best_cost) {
                best_cost = insert_candidate;
                best_op = .{ .insert = destinationString[j - 1] };
            }

            costs[cur_idx] = best_cost;
            ops[cur_idx] = best_op;
        }
    }

    return .{
        .rows = rows,
        .cols = cols,
        .costs = costs,
        .ops = ops,
        .allocator = allocator,
    };
}

/// Reconstructs operation sequence from ops table endpoint.
/// Caller owns returned operation slice.
/// Time complexity: O(m + n), Space complexity: O(m + n)
pub fn assembleTransformation(
    allocator: std.mem.Allocator,
    tables: TransformTables,
    start_i: usize,
    start_j: usize,
) ![]Operation {
    var i = start_i;
    var j = start_j;

    var reversed = std.ArrayListUnmanaged(Operation){};
    errdefer reversed.deinit(allocator);

    while (!(i == 0 and j == 0)) {
        const op = tables.opAt(i, j);
        try reversed.append(allocator, op);

        switch (op) {
            .copy, .replace => {
                i -= 1;
                j -= 1;
            },
            .delete => i -= 1,
            .insert => j -= 1,
            .zero => break,
        }
    }

    const out = try allocator.alloc(Operation, reversed.items.len);
    for (reversed.items, 0..) |op, idx_rev| {
        out[reversed.items.len - 1 - idx_rev] = op;
    }
    reversed.deinit(allocator);
    return out;
}

fn expectOp(expected: []const u8, op: Operation) !void {
    var buffer: [3]u8 = undefined;
    try testing.expectEqualStrings(expected, operationToString(op, &buffer));
}

test "min cost string conversion: python reference table example" {
    var tables = try computeTransformTables(testing.allocator, "cat", "cut", 1, 2, 3, 3);
    defer tables.deinit();

    try testing.expectEqual(@as(i64, 0), tables.costAt(0, 0));
    try testing.expectEqual(@as(i64, 3), tables.costAt(0, 1));
    try testing.expectEqual(@as(i64, 6), tables.costAt(0, 2));
    try testing.expectEqual(@as(i64, 9), tables.costAt(0, 3));

    try testing.expectEqual(@as(i64, 6), tables.costAt(2, 0));
    try testing.expectEqual(@as(i64, 4), tables.costAt(2, 1));
    try testing.expectEqual(@as(i64, 3), tables.costAt(2, 2));
    try testing.expectEqual(@as(i64, 6), tables.costAt(2, 3));

    try expectOp("0", tables.opAt(0, 0));
    try expectOp("Ic", tables.opAt(0, 1));
    try expectOp("Iu", tables.opAt(0, 2));
    try expectOp("It", tables.opAt(0, 3));

    try expectOp("Dt", tables.opAt(3, 0));
    try expectOp("Dt", tables.opAt(3, 1));
    try expectOp("Rtu", tables.opAt(3, 2));
    try expectOp("Ct", tables.opAt(3, 3));
}

test "min cost string conversion: assemble transformation and empty case" {
    var tables = try computeTransformTables(testing.allocator, "cat", "cut", 1, 2, 3, 3);
    defer tables.deinit();

    const seq = try assembleTransformation(testing.allocator, tables, tables.rows - 1, tables.cols - 1);
    defer testing.allocator.free(seq);
    try testing.expectEqual(@as(usize, 3), seq.len);
    try expectOp("Cc", seq[0]);
    try expectOp("Rau", seq[1]);
    try expectOp("Ct", seq[2]);

    var empty_tables = try computeTransformTables(testing.allocator, "", "", 1, 2, 3, 3);
    defer empty_tables.deinit();
    try testing.expectEqual(@as(i64, 0), empty_tables.costAt(0, 0));
    try expectOp("0", empty_tables.opAt(0, 0));

    const empty_seq = try assembleTransformation(testing.allocator, empty_tables, 0, 0);
    defer testing.allocator.free(empty_seq);
    try testing.expectEqual(@as(usize, 0), empty_seq.len);
}

test "min cost string conversion: extreme case" {
    const alloc = testing.allocator;
    const len: usize = 600;

    const source = try alloc.alloc(u8, len);
    defer alloc.free(source);
    @memset(source, 'a');

    const destination = try alloc.alloc(u8, len);
    defer alloc.free(destination);
    @memcpy(destination, source);
    destination[len - 1] = 'b';

    var tables = try computeTransformTables(alloc, source, destination, 1, 2, 3, 3);
    defer tables.deinit();
    try testing.expectEqual(@as(i64, 601), tables.costAt(len, len));
}
