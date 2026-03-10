//! Pollard's Rho Factorization - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/pollard_rho.py

const std = @import("std");
const testing = std.testing;

pub const PollardRhoError = error{InvalidInput};

fn gcd128(a: u128, b: u128) u128 {
    var x = a;
    var y = b;
    while (y != 0) {
        const temp = y;
        y = x % y;
        x = temp;
    }
    return x;
}

fn addMod(a: u128, b: u128, modulus: u128) u128 {
    const left = a % modulus;
    const right = b % modulus;
    if (left >= modulus - right) return left - (modulus - right);
    return left + right;
}

fn mulMod(a: u128, b: u128, modulus: u128) u128 {
    var result: u128 = 0;
    var left = a % modulus;
    var right = b;

    while (right > 0) {
        if (right & 1 == 1) result = addMod(result, left, modulus);
        left = addMod(left, left, modulus);
        right >>= 1;
    }

    return result;
}

fn randFn(value: u128, step: u128, modulus: u128) u128 {
    return (mulMod(value, value, modulus) + step) % modulus;
}

/// Returns a non-trivial factor of `num`, or `null` if no factor is found within
/// the requested number of deterministic attempts.
/// Time complexity: probabilistic; expected sub-exponential on composite inputs.
/// Space complexity: O(1)
pub fn pollardRho(num: u128, seed: u128, step: u128, attempts: u32) PollardRhoError!?u128 {
    if (num < 2) return error.InvalidInput;
    if (num > 2 and num % 2 == 0) return 2;

    var current_seed = seed;
    var current_step = step;
    var attempt_index: u32 = 0;

    while (attempt_index < attempts) : (attempt_index += 1) {
        var tortoise = current_seed;
        var hare = current_seed;

        while (true) {
            tortoise = randFn(tortoise, current_step, num);
            hare = randFn(hare, current_step, num);
            hare = randFn(hare, current_step, num);

            const difference = if (hare >= tortoise) hare - tortoise else tortoise - hare;
            const divisor = gcd128(difference, num);

            if (divisor == 1) {
                continue;
            } else if (divisor == num) {
                current_seed = hare;
                current_step += 1;
                break;
            } else {
                return divisor;
            }
        }
    }

    return null;
}

test "pollard rho: python reference examples" {
    try testing.expectEqual(@as(?u128, 274177), try pollardRho(18_446_744_073_709_551_617, 2, 1, 3));
    try testing.expectEqual(@as(?u128, 9_876_543_191), try pollardRho(97_546_105_601_219_326_301, 2, 1, 3));
    try testing.expectEqual(@as(?u128, 2), try pollardRho(100, 2, 1, 3));
    try testing.expectEqual(@as(?u128, null), try pollardRho(17, 2, 1, 3));
    try testing.expectEqual(@as(?u128, 17), try pollardRho(17 * 17 * 17, 2, 1, 3));
    try testing.expectEqual(@as(?u128, null), try pollardRho(17 * 17 * 17, 2, 1, 1));
    try testing.expectEqual(@as(?u128, 21), try pollardRho(3 * 5 * 7, 2, 1, 3));
}

test "pollard rho: invalid input and extreme edge cases" {
    try testing.expectError(error.InvalidInput, pollardRho(1, 2, 1, 3));
    try testing.expectEqual(@as(?u128, 2), try pollardRho(1 << 64, 2, 1, 3));
    try testing.expectEqual(@as(?u128, 3), try pollardRho(3 * 97, 2, 1, 3));
}
