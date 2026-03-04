//! Assignment Using Bitmask DP - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/bitmask.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const AssignmentBitmaskError = error{
    TooManyPeople,
    Overflow,
};

/// Counts the number of ways to assign tasks:
/// - each person gets at most one task,
/// - each task can be assigned to at most one person.
/// `task_performed[i]` lists tasks that person `i` can do.
/// Tasks are considered in range [1, total_tasks], matching Python reference.
/// Time complexity: O(2^P * T * avg_people_per_task), Space complexity: O(2^P * T)
pub fn countAssignmentsUsingBitmask(
    allocator: Allocator,
    task_performed: []const []const usize,
    total_tasks: usize,
) (AssignmentBitmaskError || Allocator.Error)!u64 {
    if (task_performed.len > 63) return AssignmentBitmaskError.TooManyPeople;

    var task_to_people = try allocator.alloc(std.ArrayListUnmanaged(usize), total_tasks + 1);
    defer allocator.free(task_to_people);
    for (task_to_people) |*entry| {
        entry.* = .{};
    }
    defer {
        for (task_to_people) |*entry| {
            entry.deinit(allocator);
        }
    }

    for (task_performed, 0..) |tasks, person| {
        for (tasks) |task_id| {
            if (task_id == 0 or task_id > total_tasks) continue;
            try task_to_people[task_id].append(allocator, person);
        }
    }

    const mask_count = @as(usize, 1) << @intCast(task_performed.len);
    const final_mask: u64 = if (task_performed.len == 0) 0 else ((@as(u64, 1) << @intCast(task_performed.len)) - 1);

    const cols = @addWithOverflow(total_tasks, @as(usize, 2));
    if (cols[1] != 0) return AssignmentBitmaskError.Overflow;
    const memo_size = @mulWithOverflow(mask_count, cols[0]);
    if (memo_size[1] != 0) return AssignmentBitmaskError.Overflow;

    const memo = try allocator.alloc(?u64, memo_size[0]);
    defer allocator.free(memo);
    @memset(memo, null);

    var ctx = Context{
        .total_tasks = total_tasks,
        .final_mask = final_mask,
        .task_to_people = task_to_people,
        .memo = memo,
        .cols = cols[0],
    };
    return ctx.countWays(0, 1);
}

const Context = struct {
    total_tasks: usize,
    final_mask: u64,
    task_to_people: []std.ArrayListUnmanaged(usize),
    memo: []?u64,
    cols: usize,

    fn memoIndex(self: Context, mask: u64, task_no: usize) usize {
        return @as(usize, @intCast(mask)) * self.cols + task_no;
    }

    fn countWays(self: *Context, mask: u64, task_no: usize) (AssignmentBitmaskError || Allocator.Error)!u64 {
        if (mask == self.final_mask) return 1;
        if (task_no > self.total_tasks) return 0;

        const memo_idx = self.memoIndex(mask, task_no);
        if (self.memo[memo_idx]) |cached| return cached;

        var total = try self.countWays(mask, task_no + 1); // skip this task

        for (self.task_to_people[task_no].items) |person| {
            const bit: u64 = @as(u64, 1) << @intCast(person);
            if ((mask & bit) != 0) continue;

            const addend = try self.countWays(mask | bit, task_no + 1);
            const next = @addWithOverflow(total, addend);
            if (next[1] != 0) return AssignmentBitmaskError.Overflow;
            total = next[0];
        }

        self.memo[memo_idx] = total;
        return total;
    }
};

test "bitmask assignment: python example" {
    const task_performed = [_][]const usize{
        &[_]usize{ 1, 3, 4 },
        &[_]usize{ 1, 2, 5 },
        &[_]usize{ 3, 4 },
    };
    try testing.expectEqual(@as(u64, 10), try countAssignmentsUsingBitmask(testing.allocator, &task_performed, 5));
}

test "bitmask assignment: boundary behavior" {
    const empty = [_][]const usize{};
    try testing.expectEqual(@as(u64, 1), try countAssignmentsUsingBitmask(testing.allocator, &empty, 0));

    const impossible = [_][]const usize{
        &[_]usize{1},
        &[_]usize{1},
    };
    try testing.expectEqual(@as(u64, 0), try countAssignmentsUsingBitmask(testing.allocator, &impossible, 1));
}

test "bitmask assignment: extreme dense case" {
    var tasks: [10]usize = undefined;
    for (0..tasks.len) |i| tasks[i] = i + 1;

    var people: [10][]const usize = undefined;
    for (0..people.len) |i| people[i] = &tasks;

    // 10! ways when everyone can do any of the 10 tasks.
    try testing.expectEqual(@as(u64, 3_628_800), try countAssignmentsUsingBitmask(testing.allocator, &people, 10));
}

test "bitmask assignment: overflow detection" {
    var tasks: [50]usize = undefined;
    for (0..tasks.len) |i| tasks[i] = i + 1;

    var people: [13][]const usize = undefined;
    for (0..people.len) |i| people[i] = tasks[0..40];

    // 40P13 exceeds u64.
    try testing.expectError(AssignmentBitmaskError.Overflow, countAssignmentsUsingBitmask(testing.allocator, &people, 40));
}
