//! Job Sequencing With Deadline - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/scheduling/job_sequencing_with_deadline.py

const std = @import("std");
const testing = std.testing;

pub const Job = struct {
    job_id: i64,
    deadline: i64,
    profit: i64,
};

pub const JobSequencingResult = struct {
    count: i64,
    max_profit: i64,
};

/// Returns scheduled job count and max profit using greedy slot assignment.
///
/// Time complexity: O(n log n + n * d), d = max deadline
/// Space complexity: O(d)
pub fn jobSequencingWithDeadlines(
    allocator: std.mem.Allocator,
    jobs_input: []const Job,
) std.mem.Allocator.Error!JobSequencingResult {
    if (jobs_input.len == 0) {
        return JobSequencingResult{ .count = 0, .max_profit = 0 };
    }

    const jobs = try allocator.alloc(Job, jobs_input.len);
    defer allocator.free(jobs);
    @memcpy(jobs, jobs_input);

    std.sort.insertion(Job, jobs, {}, struct {
        fn lessThan(_: void, a: Job, b: Job) bool {
            return a.profit > b.profit;
        }
    }.lessThan);

    var max_deadline: i64 = 0;
    for (jobs) |job| {
        if (job.deadline > max_deadline) max_deadline = job.deadline;
    }
    if (max_deadline <= 0) {
        return JobSequencingResult{ .count = 0, .max_profit = 0 };
    }

    const slot_count: usize = @intCast(max_deadline);
    const time_slots = try allocator.alloc(i64, slot_count);
    defer allocator.free(time_slots);
    @memset(time_slots, -1);

    var count: i64 = 0;
    var max_profit: i64 = 0;
    for (jobs) |job| {
        var i: i64 = job.deadline - 1;
        while (i >= 0) : (i -= 1) {
            const idx: usize = @intCast(i);
            if (idx >= time_slots.len) continue;
            if (time_slots[idx] == -1) {
                time_slots[idx] = job.job_id;
                count += 1;
                max_profit += job.profit;
                break;
            }
        }
    }

    return JobSequencingResult{ .count = count, .max_profit = max_profit };
}

test "job sequencing with deadline: python examples" {
    const alloc = testing.allocator;

    const r1 = try jobSequencingWithDeadlines(alloc, &[_]Job{
        .{ .job_id = 1, .deadline = 4, .profit = 20 },
        .{ .job_id = 2, .deadline = 1, .profit = 10 },
        .{ .job_id = 3, .deadline = 1, .profit = 40 },
        .{ .job_id = 4, .deadline = 1, .profit = 30 },
    });
    try testing.expectEqual(@as(i64, 2), r1.count);
    try testing.expectEqual(@as(i64, 60), r1.max_profit);

    const r2 = try jobSequencingWithDeadlines(alloc, &[_]Job{
        .{ .job_id = 1, .deadline = 2, .profit = 100 },
        .{ .job_id = 2, .deadline = 1, .profit = 19 },
        .{ .job_id = 3, .deadline = 2, .profit = 27 },
        .{ .job_id = 4, .deadline = 1, .profit = 25 },
        .{ .job_id = 5, .deadline = 1, .profit = 15 },
    });
    try testing.expectEqual(@as(i64, 2), r2.count);
    try testing.expectEqual(@as(i64, 127), r2.max_profit);
}

test "job sequencing with deadline: boundary and edge cases" {
    const alloc = testing.allocator;

    const empty = try jobSequencingWithDeadlines(alloc, &[_]Job{});
    try testing.expectEqual(@as(i64, 0), empty.count);
    try testing.expectEqual(@as(i64, 0), empty.max_profit);

    const non_positive_deadline = try jobSequencingWithDeadlines(alloc, &[_]Job{
        .{ .job_id = 1, .deadline = 0, .profit = 10 },
        .{ .job_id = 2, .deadline = -2, .profit = 20 },
    });
    try testing.expectEqual(@as(i64, 0), non_positive_deadline.count);
    try testing.expectEqual(@as(i64, 0), non_positive_deadline.max_profit);

    const stress = try jobSequencingWithDeadlines(alloc, &[_]Job{
        .{ .job_id = 1, .deadline = 10, .profit = 10_000 },
        .{ .job_id = 2, .deadline = 9, .profit = 9_000 },
        .{ .job_id = 3, .deadline = 8, .profit = 8_000 },
        .{ .job_id = 4, .deadline = 7, .profit = 7_000 },
        .{ .job_id = 5, .deadline = 6, .profit = 6_000 },
    });
    try testing.expectEqual(@as(i64, 5), stress.count);
    try testing.expectEqual(@as(i64, 40_000), stress.max_profit);
}
