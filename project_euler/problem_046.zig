//! Project Euler Problem 46: Goldbach's Other Conjecture - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_046/sol1.py

const std = @import("std");
const testing = std.testing;

pub const Problem046Error = error{
    InvalidCount,
    OutOfMemory,
};

/// Checks primality in O(sqrt(n)), matching the Python helper semantics.
///
/// Time complexity: O(sqrt(n))
/// Space complexity: O(1)
pub fn isPrime(number: u64) bool {
    if (number > 1 and number < 4) return true;
    if (number < 2 or number % 2 == 0 or number % 3 == 0) return false;

    var i: u64 = 5;
    while (i * i <= number) : (i += 6) {
        if (number % i == 0 or number % (i + 2) == 0) return false;
    }
    return true;
}

fn canBeWrittenAsPrimePlusTwiceSquare(n: u64) bool {
    var i: u64 = 0;
    while (2 * i * i <= n) : (i += 1) {
        const remainder = n - 2 * i * i;
        if (isPrime(remainder)) return true;
    }
    return false;
}

/// Returns the first `count` odd composites that disprove the conjecture.
///
/// Time complexity: depends on the search horizon, roughly O(k * sqrt(n))
/// Space complexity: O(count)
pub fn computeNums(count: usize, allocator: std.mem.Allocator) Problem046Error![]u64 {
    if (count == 0) return error.InvalidCount;

    var out = std.ArrayListUnmanaged(u64){};
    defer out.deinit(allocator);

    var candidate: u64 = 3;
    while (out.items.len < count) : (candidate += 2) {
        if (isPrime(candidate)) continue;
        if (!canBeWrittenAsPrimePlusTwiceSquare(candidate)) {
            try out.append(allocator, candidate);
        }
    }

    return try allocator.dupe(u64, out.items);
}

/// Returns the smallest odd composite that cannot be written as `prime + 2 * square`.
pub fn solution(allocator: std.mem.Allocator) Problem046Error!u64 {
    const nums = try computeNums(1, allocator);
    defer allocator.free(nums);
    return nums[0];
}

test "problem 046: python reference" {
    const allocator = testing.allocator;
    try testing.expectEqual(@as(u64, 5777), try solution(allocator));

    const first_two = try computeNums(2, allocator);
    defer allocator.free(first_two);
    try testing.expectEqualSlices(u64, &[_]u64{ 5777, 5993 }, first_two);
}

test "problem 046: helper semantics and extremes" {
    const allocator = testing.allocator;

    try testing.expect(!isPrime(0));
    try testing.expect(!isPrime(1));
    try testing.expect(isPrime(563));
    try testing.expect(!isPrime(67_483));
    try testing.expect(canBeWrittenAsPrimePlusTwiceSquare(9));
    try testing.expect(canBeWrittenAsPrimePlusTwiceSquare(15));
    try testing.expect(canBeWrittenAsPrimePlusTwiceSquare(33));
    try testing.expect(!canBeWrittenAsPrimePlusTwiceSquare(5777));
    try testing.expectError(error.InvalidCount, computeNums(0, allocator));
}
