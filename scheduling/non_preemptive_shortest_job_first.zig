//! Non-Preemptive Shortest Job First - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/scheduling/non_preemptive_shortest_job_first.py

const std = @import("std");
const testing = std.testing;

pub const NonPreemptiveSjfError = error{
    InputLengthMismatch,
    NonPositiveBurstTime,
};

/// Computes waiting times for non-preemptive SJF.
///
/// Time complexity: O(n^2) worst-case
/// Space complexity: O(n)
pub fn calculateWaitingTime(
    allocator: std.mem.Allocator,
    arrival_time: []const i64,
    burst_time: []const i64,
    no_of_processes: usize,
) (std.mem.Allocator.Error || NonPreemptiveSjfError)![]i64 {
    if (arrival_time.len < no_of_processes or burst_time.len < no_of_processes) {
        return NonPreemptiveSjfError.InputLengthMismatch;
    }
    for (burst_time[0..no_of_processes]) |b| {
        if (b <= 0) return NonPreemptiveSjfError.NonPositiveBurstTime;
    }

    const waiting_time = try allocator.alloc(i64, no_of_processes);
    @memset(waiting_time, 0);
    if (no_of_processes == 0) return waiting_time;

    const remaining_time = try allocator.alloc(i64, no_of_processes);
    defer allocator.free(remaining_time);
    @memcpy(remaining_time, burst_time[0..no_of_processes]);

    var completed: usize = 0;
    var total_time: i64 = 0;

    while (completed != no_of_processes) {
        var target_process: ?usize = null;

        for (0..no_of_processes) |i| {
            if (arrival_time[i] <= total_time and remaining_time[i] > 0) {
                if (target_process == null or remaining_time[i] < remaining_time[target_process.?]) {
                    target_process = i;
                }
            }
        }

        if (target_process) |idx| {
            total_time += burst_time[idx];
            completed += 1;
            remaining_time[idx] = 0;
            waiting_time[idx] = total_time - arrival_time[idx] - burst_time[idx];
        } else {
            total_time += 1;
        }
    }

    return waiting_time;
}

/// Computes turnaround times with formula:
/// turnaround[i] = burst[i] + waiting[i].
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn calculateTurnaroundTime(
    allocator: std.mem.Allocator,
    burst_time: []const i64,
    no_of_processes: usize,
    waiting_time: []const i64,
) (std.mem.Allocator.Error || NonPreemptiveSjfError)![]i64 {
    if (burst_time.len < no_of_processes or waiting_time.len < no_of_processes) {
        return NonPreemptiveSjfError.InputLengthMismatch;
    }

    const turn_around_time = try allocator.alloc(i64, no_of_processes);
    for (0..no_of_processes) |i| {
        turn_around_time[i] = burst_time[i] + waiting_time[i];
    }
    return turn_around_time;
}

test "non preemptive sjf: python waiting examples" {
    const alloc = testing.allocator;

    const w1 = try calculateWaitingTime(alloc, &[_]i64{ 0, 1, 2 }, &[_]i64{ 10, 5, 8 }, 3);
    defer alloc.free(w1);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 9, 13 }, w1);

    const w2 = try calculateWaitingTime(alloc, &[_]i64{ 1, 2, 2, 4 }, &[_]i64{ 4, 6, 3, 1 }, 4);
    defer alloc.free(w2);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 7, 4, 1 }, w2);

    const w3 = try calculateWaitingTime(alloc, &[_]i64{ 0, 0, 0 }, &[_]i64{ 12, 2, 10 }, 3);
    defer alloc.free(w3);
    try testing.expectEqualSlices(i64, &[_]i64{ 12, 0, 2 }, w3);
}

test "non preemptive sjf: python turnaround examples" {
    const alloc = testing.allocator;

    const t1 = try calculateTurnaroundTime(alloc, &[_]i64{ 0, 1, 2 }, 3, &[_]i64{ 0, 10, 15 });
    defer alloc.free(t1);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 11, 17 }, t1);

    const t2 = try calculateTurnaroundTime(alloc, &[_]i64{ 1, 2, 2, 4 }, 4, &[_]i64{ 1, 8, 5, 4 });
    defer alloc.free(t2);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 10, 7, 8 }, t2);

    const t3 = try calculateTurnaroundTime(alloc, &[_]i64{ 0, 0, 0 }, 3, &[_]i64{ 12, 0, 2 });
    defer alloc.free(t3);
    try testing.expectEqualSlices(i64, &[_]i64{ 12, 0, 2 }, t3);
}

test "non preemptive sjf: boundary and extreme cases" {
    const alloc = testing.allocator;

    const empty = try calculateWaitingTime(alloc, &[_]i64{}, &[_]i64{}, 0);
    defer alloc.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const idle_gap = try calculateWaitingTime(alloc, &[_]i64{ 10, 20 }, &[_]i64{ 1, 2 }, 2);
    defer alloc.free(idle_gap);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 0 }, idle_gap);

    try testing.expectError(NonPreemptiveSjfError.InputLengthMismatch, calculateWaitingTime(alloc, &[_]i64{1}, &[_]i64{}, 1));
    try testing.expectError(NonPreemptiveSjfError.NonPositiveBurstTime, calculateWaitingTime(alloc, &[_]i64{0}, &[_]i64{0}, 1));

    const stress = try calculateWaitingTime(alloc, &[_]i64{ 0, 1, 2, 3, 4 }, &[_]i64{ 9, 8, 7, 6, 5 }, 5);
    defer alloc.free(stress);
    try testing.expectEqual(@as(usize, 5), stress.len);
}
