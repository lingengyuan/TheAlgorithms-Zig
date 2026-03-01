//! Job Sequencing with Deadlines - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/scheduling/job_sequencing_with_deadline.py

const std = @import("std");
const testing = std.testing;

pub const Job = struct {
    id: i64,
    deadline: usize,
    profit: i64,
};

pub const JobSchedule = struct {
    count: usize,
    max_profit: i64,
    slots: []i64,
};

fn byProfitDesc(_: void, a: Job, b: Job) bool {
    if (a.profit != b.profit) return a.profit > b.profit;
    return a.deadline < b.deadline;
}

/// Schedules jobs to maximize total profit under deadline constraints.
/// Each job takes one slot and must be scheduled at or before its deadline.
/// Returns selected job count, profit, and slot assignment (`-1` means empty).
/// Caller owns `slots` in the returned struct.
/// Time complexity: O(n log n + n * d), where d = max deadline.
pub fn jobSequencingWithDeadlines(
    allocator: std.mem.Allocator,
    jobs: []const Job,
) !JobSchedule {
    if (jobs.len == 0) {
        return .{ .count = 0, .max_profit = 0, .slots = try allocator.alloc(i64, 0) };
    }

    const sorted = try allocator.dupe(Job, jobs);
    defer allocator.free(sorted);
    std.sort.heap(Job, sorted, {}, byProfitDesc);

    var max_deadline: usize = 0;
    for (sorted) |job| {
        if (job.deadline > max_deadline) max_deadline = job.deadline;
    }

    if (max_deadline == 0) {
        return .{ .count = 0, .max_profit = 0, .slots = try allocator.alloc(i64, 0) };
    }

    const slots = try allocator.alloc(i64, max_deadline);
    errdefer allocator.free(slots);
    @memset(slots, -1);

    var count: usize = 0;
    var profit: i64 = 0;

    for (sorted) |job| {
        if (job.deadline == 0 or job.profit <= 0) continue;

        var t = @min(job.deadline, max_deadline);
        while (t > 0) {
            t -= 1;
            if (slots[t] == -1) {
                slots[t] = job.id;
                count += 1;
                profit += job.profit;
                break;
            }
        }
    }

    return .{ .count = count, .max_profit = profit, .slots = slots };
}

test "job sequencing: python examples" {
    const alloc = testing.allocator;

    const jobs1 = [_]Job{
        .{ .id = 1, .deadline = 4, .profit = 20 },
        .{ .id = 2, .deadline = 1, .profit = 10 },
        .{ .id = 3, .deadline = 1, .profit = 40 },
        .{ .id = 4, .deadline = 1, .profit = 30 },
    };
    const r1 = try jobSequencingWithDeadlines(alloc, &jobs1);
    defer alloc.free(r1.slots);
    try testing.expectEqual(@as(usize, 2), r1.count);
    try testing.expectEqual(@as(i64, 60), r1.max_profit);

    const jobs2 = [_]Job{
        .{ .id = 1, .deadline = 2, .profit = 100 },
        .{ .id = 2, .deadline = 1, .profit = 19 },
        .{ .id = 3, .deadline = 2, .profit = 27 },
        .{ .id = 4, .deadline = 1, .profit = 25 },
        .{ .id = 5, .deadline = 1, .profit = 15 },
    };
    const r2 = try jobSequencingWithDeadlines(alloc, &jobs2);
    defer alloc.free(r2.slots);
    try testing.expectEqual(@as(usize, 2), r2.count);
    try testing.expectEqual(@as(i64, 127), r2.max_profit);
}

test "job sequencing: empty and zero deadline" {
    const alloc = testing.allocator;

    const empty = try jobSequencingWithDeadlines(alloc, &[_]Job{});
    defer alloc.free(empty.slots);
    try testing.expectEqual(@as(usize, 0), empty.count);
    try testing.expectEqual(@as(i64, 0), empty.max_profit);

    const jobs = [_]Job{
        .{ .id = 1, .deadline = 0, .profit = 10 },
        .{ .id = 2, .deadline = 0, .profit = 99 },
    };
    const r = try jobSequencingWithDeadlines(alloc, &jobs);
    defer alloc.free(r.slots);
    try testing.expectEqual(@as(usize, 0), r.count);
    try testing.expectEqual(@as(i64, 0), r.max_profit);
}

test "job sequencing: non-positive profits are skipped" {
    const alloc = testing.allocator;
    const jobs = [_]Job{
        .{ .id = 1, .deadline = 2, .profit = 40 },
        .{ .id = 2, .deadline = 2, .profit = -5 },
        .{ .id = 3, .deadline = 3, .profit = 0 },
    };

    const r = try jobSequencingWithDeadlines(alloc, &jobs);
    defer alloc.free(r.slots);

    try testing.expectEqual(@as(usize, 1), r.count);
    try testing.expectEqual(@as(i64, 40), r.max_profit);
}

test "job sequencing: extreme many same deadlines" {
    const alloc = testing.allocator;
    var jobs: [512]Job = undefined;

    for (0..jobs.len) |i| {
        jobs[i] = .{
            .id = @intCast(i + 1),
            .deadline = 1,
            .profit = @as(i64, @intCast(i + 1)),
        };
    }

    const r = try jobSequencingWithDeadlines(alloc, &jobs);
    defer alloc.free(r.slots);

    try testing.expectEqual(@as(usize, 1), r.count);
    try testing.expectEqual(@as(i64, 512), r.max_profit);
    try testing.expectEqual(@as(i64, 512), r.slots[0]);
}
