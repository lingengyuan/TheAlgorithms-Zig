//! Project Euler Problem 79: Passcode Derivation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_079/sol1.py

const std = @import("std");
const testing = std.testing;

const keylog_file = @embedFile("problem_079_keylog.txt");
const keylog_test_file = @embedFile("problem_079_keylog_test.txt");

pub const Problem079Error = error{ OutOfMemory, InvalidLogin, NoSolution };

fn parseLogins(allocator: std.mem.Allocator, data: []const u8) Problem079Error![][]const u8 {
    var lines = std.ArrayListUnmanaged([]const u8){};
    errdefer lines.deinit(allocator);

    var it = std.mem.tokenizeAny(u8, data, "\r\n");
    while (it.next()) |line| {
        if (line.len != 3) return error.InvalidLogin;
        try lines.append(allocator, line);
    }
    return lines.toOwnedSlice(allocator);
}

/// Returns the shortest passcode satisfying all ordered login constraints.
/// Time complexity: O(logins + digits^2)
/// Space complexity: O(digits^2)
pub fn findSecretPasscode(allocator: std.mem.Allocator, logins: []const []const u8) Problem079Error!u64 {
    _ = allocator;
    var present = [_]bool{false} ** 10;
    var edges = [_][10]bool{[_]bool{false} ** 10} ** 10;
    var indegree = [_]u8{0} ** 10;

    for (logins) |login| {
        if (login.len != 3) return error.InvalidLogin;
        for (login) |ch| {
            if (ch < '0' or ch > '9') return error.InvalidLogin;
            present[ch - '0'] = true;
        }
        const a = login[0] - '0';
        const b = login[1] - '0';
        const c = login[2] - '0';
        if (!edges[a][b]) {
            edges[a][b] = true;
            indegree[b] += 1;
        }
        if (!edges[b][c]) {
            edges[b][c] = true;
            indegree[c] += 1;
        }
    }

    var result: u64 = 0;
    var used: usize = 0;
    const total = blk: {
        var count: usize = 0;
        for (present) |flag| {
            if (flag) count += 1;
        }
        break :blk count;
    };

    while (used < total) {
        var candidate: ?u8 = null;
        for (present, 0..) |flag, digit| {
            if (!flag) continue;
            if (indegree[digit] != 0) continue;
            candidate = @intCast(digit);
            break;
        }
        if (candidate == null) return error.NoSolution;

        const digit = candidate.?;
        present[digit] = false;
        result = result * 10 + digit;
        used += 1;

        for (edges[digit], 0..) |has_edge, next| {
            if (has_edge) indegree[next] -= 1;
        }
    }

    return result;
}

pub fn solution(allocator: std.mem.Allocator) Problem079Error!u64 {
    const logins = try parseLogins(allocator, keylog_file);
    defer allocator.free(logins);
    return findSecretPasscode(allocator, logins);
}

test "problem 079: python reference" {
    try testing.expectEqual(@as(u64, 73_162_890), try solution(testing.allocator));
}

test "problem 079: sample datasets and edge parsing" {
    const alloc = testing.allocator;
    const test_logins = try parseLogins(alloc, keylog_test_file);
    defer alloc.free(test_logins);
    try testing.expectEqual(@as(u64, 6_312_980), try findSecretPasscode(alloc, test_logins));

    const sample1 = [_][]const u8{ "135", "259", "235", "189", "690", "168", "120", "136", "289", "589", "160", "165", "580", "369", "250", "280" };
    try testing.expectEqual(@as(u64, 12_365_890), try findSecretPasscode(alloc, &sample1));

    const sample2 = [_][]const u8{ "426", "281", "061", "819", "268", "406", "420", "428", "209", "689", "019", "421", "469", "261", "681", "201" };
    try testing.expectEqual(@as(u64, 4_206_819), try findSecretPasscode(alloc, &sample2));

    try testing.expectError(error.InvalidLogin, parseLogins(alloc, "12\n"));
}
