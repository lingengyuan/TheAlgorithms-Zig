//! Resistor Color Code - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/resistor_color_code.py

const std = @import("std");
const testing = std.testing;

pub const ResistorColorCodeError = error{
    InvalidBandCount,
    ColorCountMismatch,
    InvalidColor,
    InvalidSignificantFigureColor,
    InvalidMultiplierColor,
    InvalidToleranceColor,
    InvalidTemperatureCoefficientColor,
};

fn isValidColor(color: []const u8) bool {
    return std.mem.eql(u8, color, "Black") or
        std.mem.eql(u8, color, "Brown") or
        std.mem.eql(u8, color, "Red") or
        std.mem.eql(u8, color, "Orange") or
        std.mem.eql(u8, color, "Yellow") or
        std.mem.eql(u8, color, "Green") or
        std.mem.eql(u8, color, "Blue") or
        std.mem.eql(u8, color, "Violet") or
        std.mem.eql(u8, color, "Grey") or
        std.mem.eql(u8, color, "White") or
        std.mem.eql(u8, color, "Gold") or
        std.mem.eql(u8, color, "Silver");
}

fn significantDigitValue(color: []const u8) ResistorColorCodeError!u8 {
    if (std.mem.eql(u8, color, "Black")) return 0;
    if (std.mem.eql(u8, color, "Brown")) return 1;
    if (std.mem.eql(u8, color, "Red")) return 2;
    if (std.mem.eql(u8, color, "Orange")) return 3;
    if (std.mem.eql(u8, color, "Yellow")) return 4;
    if (std.mem.eql(u8, color, "Green")) return 5;
    if (std.mem.eql(u8, color, "Blue")) return 6;
    if (std.mem.eql(u8, color, "Violet")) return 7;
    if (std.mem.eql(u8, color, "Grey")) return 8;
    if (std.mem.eql(u8, color, "White")) return 9;
    return ResistorColorCodeError.InvalidSignificantFigureColor;
}

fn multiplierValue(color: []const u8) ResistorColorCodeError!f64 {
    if (std.mem.eql(u8, color, "Black")) return 1;
    if (std.mem.eql(u8, color, "Brown")) return 10;
    if (std.mem.eql(u8, color, "Red")) return 1e2;
    if (std.mem.eql(u8, color, "Orange")) return 1e3;
    if (std.mem.eql(u8, color, "Yellow")) return 1e4;
    if (std.mem.eql(u8, color, "Green")) return 1e5;
    if (std.mem.eql(u8, color, "Blue")) return 1e6;
    if (std.mem.eql(u8, color, "Violet")) return 1e7;
    if (std.mem.eql(u8, color, "Grey")) return 1e8;
    if (std.mem.eql(u8, color, "White")) return 1e9;
    if (std.mem.eql(u8, color, "Gold")) return 1e-1;
    if (std.mem.eql(u8, color, "Silver")) return 1e-2;
    return ResistorColorCodeError.InvalidMultiplierColor;
}

fn toleranceValue(color: []const u8) ResistorColorCodeError!f64 {
    if (std.mem.eql(u8, color, "Brown")) return 1;
    if (std.mem.eql(u8, color, "Red")) return 2;
    if (std.mem.eql(u8, color, "Orange")) return 0.05;
    if (std.mem.eql(u8, color, "Yellow")) return 0.02;
    if (std.mem.eql(u8, color, "Green")) return 0.5;
    if (std.mem.eql(u8, color, "Blue")) return 0.25;
    if (std.mem.eql(u8, color, "Violet")) return 0.1;
    if (std.mem.eql(u8, color, "Grey")) return 0.01;
    if (std.mem.eql(u8, color, "Gold")) return 5;
    if (std.mem.eql(u8, color, "Silver")) return 10;
    return ResistorColorCodeError.InvalidToleranceColor;
}

fn temperatureCoefficientValue(color: []const u8) ResistorColorCodeError!u16 {
    if (std.mem.eql(u8, color, "Black")) return 250;
    if (std.mem.eql(u8, color, "Brown")) return 100;
    if (std.mem.eql(u8, color, "Red")) return 50;
    if (std.mem.eql(u8, color, "Orange")) return 15;
    if (std.mem.eql(u8, color, "Yellow")) return 25;
    if (std.mem.eql(u8, color, "Green")) return 20;
    if (std.mem.eql(u8, color, "Blue")) return 10;
    if (std.mem.eql(u8, color, "Violet")) return 5;
    if (std.mem.eql(u8, color, "Grey")) return 1;
    return ResistorColorCodeError.InvalidTemperatureCoefficientColor;
}

fn trimTrailingZeros(number_text: []const u8) []const u8 {
    var end = number_text.len;
    while (end > 0 and number_text[end - 1] == '0') {
        end -= 1;
    }
    if (end > 0 and number_text[end - 1] == '.') {
        end -= 1;
    }
    return number_text[0..end];
}

fn formatPythonLikeNumber(value: f64, buffer: []u8) []const u8 {
    const nearest_integer = @round(value);
    if (@abs(value - nearest_integer) < 1e-12 and
        nearest_integer >= @as(f64, @floatFromInt(std.math.minInt(i64))) and
        nearest_integer <= @as(f64, @floatFromInt(std.math.maxInt(i64))))
    {
        return std.fmt.bufPrint(buffer, "{d}", .{@as(i64, @intFromFloat(nearest_integer))}) catch unreachable;
    }

    const raw = std.fmt.bufPrint(buffer, "{d:.12}", .{value}) catch unreachable;
    return trimTrailingZeros(raw);
}

/// Calculates resistor value from color bands, aligned to Python reference output semantics.
/// Returned string is allocator-owned and must be freed by the caller.
///
/// Time complexity: O(bands)
/// Space complexity: O(1) auxiliary
pub fn calculateResistance(
    allocator: std.mem.Allocator,
    number_of_bands: u8,
    color_code_list: []const []const u8,
) (std.mem.Allocator.Error || ResistorColorCodeError)![]u8 {
    if (number_of_bands < 3 or number_of_bands > 6) {
        return ResistorColorCodeError.InvalidBandCount;
    }
    if (color_code_list.len != number_of_bands) {
        return ResistorColorCodeError.ColorCountMismatch;
    }

    for (color_code_list) |color| {
        if (!isValidColor(color)) {
            return ResistorColorCodeError.InvalidColor;
        }
    }

    const significant_band_count: u8 = if (number_of_bands <= 4) 2 else 3;

    var significant_digits: u32 = 0;
    for (0..significant_band_count) |i| {
        const digit = try significantDigitValue(color_code_list[i]);
        significant_digits = significant_digits * 10 + digit;
    }

    const multiplier = try multiplierValue(color_code_list[significant_band_count]);
    const tolerance = if (number_of_bands == 3)
        @as(f64, 20)
    else
        try toleranceValue(color_code_list[significant_band_count + 1]);

    const temp_coefficient: ?u16 = if (number_of_bands == 6)
        try temperatureCoefficientValue(color_code_list[significant_band_count + 2])
    else
        null;

    const resistance = @as(f64, @floatFromInt(significant_digits)) * multiplier;

    var resistance_buffer: [64]u8 = undefined;
    var tolerance_buffer: [32]u8 = undefined;
    const resistance_text = formatPythonLikeNumber(resistance, resistance_buffer[0..]);
    const tolerance_text = formatPythonLikeNumber(tolerance, tolerance_buffer[0..]);

    if (temp_coefficient) |temp| {
        return std.fmt.allocPrint(allocator, "{s}Ω ±{s}% {d} ppm/K", .{ resistance_text, tolerance_text, temp });
    }

    return std.fmt.allocPrint(allocator, "{s}Ω ±{s}% ", .{ resistance_text, tolerance_text });
}

test "resistor color code: python examples" {
    const allocator = testing.allocator;

    const colors_3 = [_][]const u8{ "Black", "Blue", "Orange" };
    const result_3 = try calculateResistance(allocator, 3, &colors_3);
    defer allocator.free(result_3);
    try testing.expectEqualStrings("6000Ω ±20% ", result_3);

    const colors_4 = [_][]const u8{ "Orange", "Green", "Blue", "Gold" };
    const result_4 = try calculateResistance(allocator, 4, &colors_4);
    defer allocator.free(result_4);
    try testing.expectEqualStrings("35000000Ω ±5% ", result_4);

    const colors_5 = [_][]const u8{ "Violet", "Brown", "Grey", "Silver", "Green" };
    const result_5 = try calculateResistance(allocator, 5, &colors_5);
    defer allocator.free(result_5);
    try testing.expectEqualStrings("7.18Ω ±0.5% ", result_5);

    const colors_6 = [_][]const u8{ "Red", "Green", "Blue", "Yellow", "Orange", "Grey" };
    const result_6 = try calculateResistance(allocator, 6, &colors_6);
    defer allocator.free(result_6);
    try testing.expectEqualStrings("2560000Ω ±0.05% 1 ppm/K", result_6);
}

test "resistor color code: validation and extreme cases" {
    const allocator = testing.allocator;

    const wrong_count = [_][]const u8{ "Violet", "Brown", "Grey", "Silver", "Green" };
    try testing.expectError(ResistorColorCodeError.InvalidBandCount, calculateResistance(allocator, 0, &wrong_count));
    try testing.expectError(ResistorColorCodeError.ColorCountMismatch, calculateResistance(allocator, 4, &wrong_count));

    const invalid_sig = [_][]const u8{ "Violet", "Silver", "Brown", "Grey" };
    try testing.expectError(ResistorColorCodeError.InvalidSignificantFigureColor, calculateResistance(allocator, 4, &invalid_sig));

    const invalid_color = [_][]const u8{ "Violet", "Blue", "Lime", "Grey" };
    try testing.expectError(ResistorColorCodeError.InvalidColor, calculateResistance(allocator, 4, &invalid_color));

    const extreme = [_][]const u8{ "White", "White", "White", "White", "Brown", "Black" };
    const extreme_result = try calculateResistance(allocator, 6, &extreme);
    defer allocator.free(extreme_result);
    try testing.expectEqualStrings("999000000000Ω ±1% 250 ppm/K", extreme_result);
}
