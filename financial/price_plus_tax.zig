//! Price Plus Tax - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/financial/price_plus_tax.py

const std = @import("std");
const testing = std.testing;

/// Computes final price after applying tax rate:
/// price * (1 + tax_rate).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn pricePlusTax(price: f64, tax_rate: f64) f64 {
    return price * (1.0 + tax_rate);
}

test "price plus tax: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 125.0), pricePlusTax(100, 0.25), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 131.775), pricePlusTax(125.50, 0.05), 1e-12);
}

test "price plus tax: boundary and extreme values" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), pricePlusTax(0.0, 0.2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 80.0), pricePlusTax(100.0, -0.2), 1e-12);

    const huge = pricePlusTax(1e300, 1e-3);
    try testing.expect(std.math.isFinite(huge));
    try testing.expect(huge > 0);
}
