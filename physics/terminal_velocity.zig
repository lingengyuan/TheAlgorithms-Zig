//! Terminal Velocity - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/physics/terminal_velocity.py

const std = @import("std");
const testing = std.testing;

pub const TerminalVelocityError = error{
    NonPositiveInput,
};

const gravity: f64 = 9.80665;

/// Computes terminal velocity:
/// Vt = sqrt((2 * m * g) / (density * area * drag_coefficient)).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn terminalVelocity(
    mass: f64,
    density: f64,
    area: f64,
    drag_coefficient: f64,
) TerminalVelocityError!f64 {
    if (mass <= 0 or density <= 0 or area <= 0 or drag_coefficient <= 0) {
        return TerminalVelocityError.NonPositiveInput;
    }
    return @sqrt((2.0 * mass * gravity) / (density * area * drag_coefficient));
}

test "terminal velocity: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 1.3031197996044768), try terminalVelocity(1, 25, 0.6, 0.77), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.9467947148674276), try terminalVelocity(2, 100, 0.45, 0.23), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 4.428690551393267), try terminalVelocity(5, 50, 0.2, 0.5), 1e-12);

    try testing.expectError(TerminalVelocityError.NonPositiveInput, terminalVelocity(-5, 50, -0.2, -2));
    try testing.expectError(TerminalVelocityError.NonPositiveInput, terminalVelocity(3, -20, -1, 2));
    try testing.expectError(TerminalVelocityError.NonPositiveInput, terminalVelocity(-2, -1, -0.44, -1));
}

test "terminal velocity: boundary and extreme values" {
    const tiny = try terminalVelocity(1e-9, 1e3, 1e-6, 1e-3);
    try testing.expect(std.math.isFinite(tiny));
    try testing.expect(tiny > 0);

    const huge = try terminalVelocity(1e20, 1e-6, 1e-6, 1e-6);
    try testing.expect(std.math.isFinite(huge));
    try testing.expect(huge > 0);
}
