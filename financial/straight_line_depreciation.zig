//! Straight Line Depreciation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/financial/straight_line_depreciation.py

const std = @import("std");
const testing = std.testing;

pub const DepreciationError = error{
    UsefulYearsLessThanOne,
    PurchaseValueNegative,
    PurchaseBelowResidual,
};

/// Returns yearly depreciation expenses using straight-line method.
///
/// Time complexity: O(years)
/// Space complexity: O(years)
pub fn straightLineDepreciation(
    allocator: std.mem.Allocator,
    useful_years: i64,
    purchase_value: f64,
    residual_value: f64,
) (std.mem.Allocator.Error || DepreciationError)![]f64 {
    if (useful_years < 1) return DepreciationError.UsefulYearsLessThanOne;
    if (purchase_value < 0.0) return DepreciationError.PurchaseValueNegative;
    if (purchase_value < residual_value) return DepreciationError.PurchaseBelowResidual;

    const years: usize = @intCast(useful_years);
    const depreciable_cost = purchase_value - residual_value;
    const annual_expense = depreciable_cost / @as(f64, @floatFromInt(useful_years));

    const expenses = try allocator.alloc(f64, years);
    var accumulated: f64 = 0.0;
    for (0..years) |period| {
        if (period != years - 1) {
            accumulated += annual_expense;
            expenses[period] = annual_expense;
        } else {
            expenses[period] = depreciable_cost - accumulated;
        }
    }
    return expenses;
}

test "straight line depreciation: python examples" {
    const alloc = testing.allocator;

    const d1 = try straightLineDepreciation(alloc, 10, 1100.0, 100.0);
    defer alloc.free(d1);
    for (d1) |v| try testing.expectApproxEqAbs(@as(f64, 100.0), v, 1e-12);

    const d2 = try straightLineDepreciation(alloc, 6, 1250.0, 50.0);
    defer alloc.free(d2);
    for (d2) |v| try testing.expectApproxEqAbs(@as(f64, 200.0), v, 1e-12);

    const d3 = try straightLineDepreciation(alloc, 4, 1001.0, 0.0);
    defer alloc.free(d3);
    for (d3) |v| try testing.expectApproxEqAbs(@as(f64, 250.25), v, 1e-12);

    const d4 = try straightLineDepreciation(alloc, 11, 380.0, 50.0);
    defer alloc.free(d4);
    for (d4) |v| try testing.expectApproxEqAbs(@as(f64, 30.0), v, 1e-12);

    const d5 = try straightLineDepreciation(alloc, 1, 4985.0, 100.0);
    defer alloc.free(d5);
    try testing.expectEqual(@as(usize, 1), d5.len);
    try testing.expectApproxEqAbs(@as(f64, 4885.0), d5[0], 1e-12);
}

test "straight line depreciation: validation and boundary cases" {
    const alloc = testing.allocator;

    try testing.expectError(DepreciationError.UsefulYearsLessThanOne, straightLineDepreciation(alloc, 0, 100.0, 0.0));
    try testing.expectError(DepreciationError.PurchaseValueNegative, straightLineDepreciation(alloc, 5, -1.0, 0.0));
    try testing.expectError(DepreciationError.PurchaseBelowResidual, straightLineDepreciation(alloc, 5, 10.0, 20.0));

    const zero_dep = try straightLineDepreciation(alloc, 3, 100.0, 100.0);
    defer alloc.free(zero_dep);
    for (zero_dep) |v| try testing.expectApproxEqAbs(@as(f64, 0.0), v, 1e-12);
}
