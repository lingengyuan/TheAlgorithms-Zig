//! Wheatstone Bridge Solver - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/wheatstone_bridge.py

const std = @import("std");
const testing = std.testing;

pub const WheatstoneBridgeError = error{
    NonPositiveResistance,
};

/// Computes unknown resistor Rx in a balanced Wheatstone bridge:
/// Rx = (R2 / R1) * R3.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn wheatstoneSolver(
    resistance_1: f64,
    resistance_2: f64,
    resistance_3: f64,
) WheatstoneBridgeError!f64 {
    if (resistance_1 <= 0 or resistance_2 <= 0 or resistance_3 <= 0) {
        return WheatstoneBridgeError.NonPositiveResistance;
    }
    return (resistance_2 / resistance_1) * resistance_3;
}

test "wheatstone bridge: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 10.0), try wheatstoneSolver(2, 4, 5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 641.5280898876405), try wheatstoneSolver(356, 234, 976), 1e-12);

    try testing.expectError(WheatstoneBridgeError.NonPositiveResistance, wheatstoneSolver(2, -1, 2));
    try testing.expectError(WheatstoneBridgeError.NonPositiveResistance, wheatstoneSolver(0, 0, 2));
}

test "wheatstone bridge: boundary and extreme values" {
    const huge = try wheatstoneSolver(1e-9, 1e9, 1e9);
    try testing.expect(std.math.isFinite(huge));
    try testing.expect(huge > 0);
}
