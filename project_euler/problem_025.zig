//! Project Euler Problem 25: 1000-digit Fibonacci Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_025/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem025Error = error{
    OutOfMemory,
};

fn addDecimalDigitsLittleEndian(out: *std.ArrayListUnmanaged(u8), a: []const u8, b: []const u8, allocator: std.mem.Allocator) Problem025Error!void {
    out.clearRetainingCapacity();

    const max_len = if (a.len > b.len) a.len else b.len;
    try out.ensureTotalCapacity(allocator, max_len + 1);

    var carry: u8 = 0;
    var i: usize = 0;
    while (i < max_len) : (i += 1) {
        const da: u8 = if (i < a.len) a[i] else 0;
        const db: u8 = if (i < b.len) b[i] else 0;

        const sum: u8 = da + db + carry;
        out.appendAssumeCapacity(sum % 10);
        carry = sum / 10;
    }

    if (carry > 0) {
        out.appendAssumeCapacity(carry);
    }
}

/// Returns the Python-reference index for first Fibonacci term containing
/// `target_digits` decimal digits.
///
/// Note: This mirrors the Python implementation semantics:
/// - if target_digits <= 0, returns 2
/// - search starts from index 3
///
/// Time complexity: O(index * digits)
/// Space complexity: O(digits)
pub fn solution(target_digits: i32, allocator: std.mem.Allocator) Problem025Error!u32 {
    if (target_digits <= 0) return 2;

    // Python reference starts with index=2 and checks fibonacci(index) from index 3.
    var index: u32 = 2;
    var digits_count: i32 = 0;

    // F1 = 1, F2 = 1 in sequence used by Python's fibonacci(index>=2) flow.
    var prev = std.ArrayListUnmanaged(u8){};
    defer prev.deinit(allocator);
    try prev.append(allocator, 1);

    var curr = std.ArrayListUnmanaged(u8){};
    defer curr.deinit(allocator);
    try curr.append(allocator, 1);

    var next = std.ArrayListUnmanaged(u8){};
    defer next.deinit(allocator);

    while (digits_count < target_digits) {
        index += 1;

        try addDecimalDigitsLittleEndian(&next, prev.items, curr.items, allocator);
        digits_count = @intCast(next.items.len);

        const tmp = prev;
        prev = curr;
        curr = next;
        next = tmp;
    }

    return index;
}

test "problem 025: python reference" {
    const allocator = testing.allocator;

    try testing.expectEqual(@as(u32, 4782), try solution(1000, allocator));
    try testing.expectEqual(@as(u32, 476), try solution(100, allocator));
    try testing.expectEqual(@as(u32, 237), try solution(50, allocator));
    try testing.expectEqual(@as(u32, 12), try solution(3, allocator));
}

test "problem 025: boundaries and edge semantics" {
    const allocator = testing.allocator;

    // These match Python's current implementation semantics.
    try testing.expectEqual(@as(u32, 3), try solution(1, allocator));
    try testing.expectEqual(@as(u32, 7), try solution(2, allocator));
    try testing.expectEqual(@as(u32, 2), try solution(0, allocator));
    try testing.expectEqual(@as(u32, 2), try solution(-1, allocator));
}
