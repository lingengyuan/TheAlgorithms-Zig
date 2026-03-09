//! Nth Fibonacci Using Matrix Exponentiation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/matrix/nth_fibonacci_using_matrix_exponentiation.py

const std = @import("std");
const testing = std.testing;

fn multiply(matrix_a: [2][2]i128, matrix_b: [2][2]i128) [2][2]i128 {
    var matrix_c: [2][2]i128 = undefined;
    for (0..2) |i| {
        for (0..2) |j| {
            matrix_c[i][j] = 0;
            for (0..2) |k| {
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }
    return matrix_c;
}

fn identity() [2][2]i128 {
    return .{
        .{ 1, 0 },
        .{ 0, 1 },
    };
}

/// Returns the nth Fibonacci number using matrix exponentiation.
/// For n <= 1, returns n directly to match the Python reference.
/// Time complexity: O(log n), Space complexity: O(1)
pub fn nthFibonacciMatrix(n: i64) i128 {
    if (n <= 1) return n;

    var res_matrix = identity();
    var fibonacci_matrix: [2][2]i128 = .{
        .{ 1, 1 },
        .{ 1, 0 },
    };
    var power: u64 = @intCast(n - 1);

    while (power > 0) {
        if (power & 1 == 1) {
            res_matrix = multiply(res_matrix, fibonacci_matrix);
        }
        fibonacci_matrix = multiply(fibonacci_matrix, fibonacci_matrix);
        power >>= 1;
    }
    return res_matrix[0][0];
}

/// Returns the nth Fibonacci number using the iterative definition.
/// Time complexity: O(n), Space complexity: O(1)
pub fn nthFibonacciBruteforce(n: i64) i128 {
    if (n <= 1) return n;
    var fib0: i128 = 0;
    var fib1: i128 = 1;
    var i: i64 = 2;
    while (i <= n) : (i += 1) {
        const next = fib0 + fib1;
        fib0 = fib1;
        fib1 = next;
    }
    return fib1;
}

test "nth fibonacci matrix: python reference examples" {
    try testing.expectEqual(@as(i128, 354224848179261915075), nthFibonacciMatrix(100));
    try testing.expectEqual(@as(i128, -100), nthFibonacciMatrix(-100));
    try testing.expectEqual(@as(i128, 354224848179261915075), nthFibonacciBruteforce(100));
}

test "nth fibonacci matrix: edge and extreme cases" {
    try testing.expectEqual(@as(i128, 0), nthFibonacciMatrix(0));
    try testing.expectEqual(@as(i128, 1), nthFibonacciMatrix(1));
    try testing.expectEqual(@as(i128, 1), nthFibonacciMatrix(2));
    try testing.expectEqual(@as(i128, 12586269025), nthFibonacciMatrix(50));
    try testing.expectEqual(nthFibonacciBruteforce(50), nthFibonacciMatrix(50));
}
