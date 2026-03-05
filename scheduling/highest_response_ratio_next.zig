//! Highest Response Ratio Next Scheduling - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/scheduling/highest_response_ratio_next.py

const std = @import("std");
const testing = std.testing;

pub const HrrnError = error{
    InputLengthMismatch,
    NonPositiveBurstTime,
};

fn argsortByArrival(
    allocator: std.mem.Allocator,
    arrival_time: []const i64,
    no_of_process: usize,
) std.mem.Allocator.Error![]usize {
    const idx = try allocator.alloc(usize, no_of_process);
    for (0..no_of_process) |i| idx[i] = i;

    std.sort.insertion(usize, idx, arrival_time, struct {
        fn lessThan(context: []const i64, a: usize, b: usize) bool {
            return context[a] < context[b];
        }
    }.lessThan);

    return idx;
}

/// Calculates per-process turnaround time using HRRN scheduling.
/// Output order matches the Python reference implementation after sorting by arrival time.
///
/// Time complexity: O(n^2)
/// Space complexity: O(n)
pub fn calculateTurnAroundTime(
    allocator: std.mem.Allocator,
    process_name: []const []const u8,
    arrival_time: []const i64,
    burst_time: []const i64,
    no_of_process: usize,
) (std.mem.Allocator.Error || HrrnError)![]i64 {
    if (process_name.len < no_of_process or arrival_time.len < no_of_process or burst_time.len < no_of_process) {
        return HrrnError.InputLengthMismatch;
    }
    for (burst_time[0..no_of_process]) |b| {
        if (b <= 0) return HrrnError.NonPositiveBurstTime;
    }

    const turn_around_time = try allocator.alloc(i64, no_of_process);
    @memset(turn_around_time, 0);
    if (no_of_process == 0) return turn_around_time;

    const sorted_idx = try argsortByArrival(allocator, arrival_time, no_of_process);
    defer allocator.free(sorted_idx);

    const sorted_arrival = try allocator.alloc(i64, no_of_process);
    defer allocator.free(sorted_arrival);
    const sorted_burst = try allocator.alloc(i64, no_of_process);
    defer allocator.free(sorted_burst);
    for (0..no_of_process) |i| {
        sorted_arrival[i] = arrival_time[sorted_idx[i]];
        sorted_burst[i] = burst_time[sorted_idx[i]];
    }

    var current_time: i64 = 0;
    var finished_process_count: usize = 0;
    const finished_process = try allocator.alloc(u8, no_of_process);
    defer allocator.free(finished_process);
    @memset(finished_process, 0);

    while (no_of_process > finished_process_count) {
        var i: usize = 0;
        while (finished_process[i] == 1) : (i += 1) {}
        current_time = @max(current_time, sorted_arrival[i]);

        var response_ratio: f64 = 0;
        var loc: usize = 0;
        var temp: f64 = 0;
        for (0..no_of_process) |j| {
            if (finished_process[j] == 0 and sorted_arrival[j] <= current_time) {
                temp = @as(f64, @floatFromInt(sorted_burst[j] + (current_time - sorted_arrival[j]))) /
                    @as(f64, @floatFromInt(sorted_burst[j]));
            }
            if (response_ratio < temp) {
                response_ratio = temp;
                loc = j;
            }
        }

        turn_around_time[loc] = current_time + sorted_burst[loc] - sorted_arrival[loc];
        current_time += sorted_burst[loc];
        finished_process[loc] = 1;
        finished_process_count += 1;
    }

    return turn_around_time;
}

/// Calculates waiting times from turnaround and burst arrays:
/// waiting[i] = turnaround[i] - burst[i].
///
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn calculateWaitingTime(
    allocator: std.mem.Allocator,
    process_name: []const []const u8,
    turn_around_time: []const i64,
    burst_time: []const i64,
    no_of_process: usize,
) (std.mem.Allocator.Error || HrrnError)![]i64 {
    _ = process_name; // kept for API parity with Python reference

    if (turn_around_time.len < no_of_process or burst_time.len < no_of_process) {
        return HrrnError.InputLengthMismatch;
    }

    const waiting_time = try allocator.alloc(i64, no_of_process);
    for (0..no_of_process) |i| {
        waiting_time[i] = turn_around_time[i] - burst_time[i];
    }
    return waiting_time;
}

test "highest response ratio next: python turnaround examples" {
    const alloc = testing.allocator;

    const ta1 = try calculateTurnAroundTime(
        alloc,
        &[_][]const u8{ "A", "B", "C" },
        &[_]i64{ 3, 5, 8 },
        &[_]i64{ 2, 4, 6 },
        3,
    );
    defer alloc.free(ta1);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 4, 7 }, ta1);

    const ta2 = try calculateTurnAroundTime(
        alloc,
        &[_][]const u8{ "A", "B", "C" },
        &[_]i64{ 0, 2, 4 },
        &[_]i64{ 3, 5, 7 },
        3,
    );
    defer alloc.free(ta2);
    try testing.expectEqualSlices(i64, &[_]i64{ 3, 6, 11 }, ta2);
}

test "highest response ratio next: python waiting examples" {
    const alloc = testing.allocator;

    const w1 = try calculateWaitingTime(
        alloc,
        &[_][]const u8{ "A", "B", "C" },
        &[_]i64{ 2, 4, 7 },
        &[_]i64{ 2, 4, 6 },
        3,
    );
    defer alloc.free(w1);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 0, 1 }, w1);

    const w2 = try calculateWaitingTime(
        alloc,
        &[_][]const u8{ "A", "B", "C" },
        &[_]i64{ 3, 6, 11 },
        &[_]i64{ 3, 5, 7 },
        3,
    );
    defer alloc.free(w2);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 1, 4 }, w2);
}

test "highest response ratio next: boundary and extreme cases" {
    const alloc = testing.allocator;

    const empty_ta = try calculateTurnAroundTime(alloc, &[_][]const u8{}, &[_]i64{}, &[_]i64{}, 0);
    defer alloc.free(empty_ta);
    try testing.expectEqual(@as(usize, 0), empty_ta.len);

    try testing.expectError(
        HrrnError.InputLengthMismatch,
        calculateTurnAroundTime(
            alloc,
            &[_][]const u8{"A"},
            &[_]i64{1},
            &[_]i64{},
            1,
        ),
    );
    try testing.expectError(
        HrrnError.NonPositiveBurstTime,
        calculateTurnAroundTime(
            alloc,
            &[_][]const u8{"A"},
            &[_]i64{1},
            &[_]i64{0},
            1,
        ),
    );

    const stress_ta = try calculateTurnAroundTime(
        alloc,
        &[_][]const u8{ "A", "B", "C", "D", "E" },
        &[_]i64{ 1, 2, 3, 4, 5 },
        &[_]i64{ 1, 2, 3, 4, 5 },
        5,
    );
    defer alloc.free(stress_ta);
    try testing.expectEqual(@as(usize, 5), stress_ta.len);
}
