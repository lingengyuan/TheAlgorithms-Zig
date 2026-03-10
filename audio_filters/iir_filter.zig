//! IIR Filter - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/audio_filters/iir_filter.py

const std = @import("std");
const testing = std.testing;

pub const IIRFilterError = error{
    InvalidACoefficients,
    InvalidBCoefficients,
};

/// General N-order IIR filter over normalized floating-point samples.
/// Caller owns the filter buffers and must call `deinit`.
/// Time complexity: O(order), Space complexity: O(order)
pub const IIRFilter = struct {
    allocator: std.mem.Allocator,
    order: usize,
    a_coeffs: []f64,
    b_coeffs: []f64,
    input_history: []f64,
    output_history: []f64,

    pub fn init(allocator: std.mem.Allocator, order: usize) !IIRFilter {
        const a_coeffs = try allocator.alloc(f64, order + 1);
        errdefer allocator.free(a_coeffs);
        const b_coeffs = try allocator.alloc(f64, order + 1);
        errdefer allocator.free(b_coeffs);
        const input_history = try allocator.alloc(f64, order);
        errdefer allocator.free(input_history);
        const output_history = try allocator.alloc(f64, order);
        errdefer allocator.free(output_history);

        @memset(a_coeffs, 0.0);
        @memset(b_coeffs, 0.0);
        @memset(input_history, 0.0);
        @memset(output_history, 0.0);
        a_coeffs[0] = 1.0;
        b_coeffs[0] = 1.0;

        return .{
            .allocator = allocator,
            .order = order,
            .a_coeffs = a_coeffs,
            .b_coeffs = b_coeffs,
            .input_history = input_history,
            .output_history = output_history,
        };
    }

    pub fn deinit(self: *IIRFilter) void {
        self.allocator.free(self.a_coeffs);
        self.allocator.free(self.b_coeffs);
        self.allocator.free(self.input_history);
        self.allocator.free(self.output_history);
    }

    /// Sets the denominator (`a`) and numerator (`b`) coefficients.
    /// Python's runtime behavior requires both slices to have `order + 1` elements.
    pub fn setCoefficients(self: *IIRFilter, a_coeffs: []const f64, b_coeffs: []const f64) IIRFilterError!void {
        if (a_coeffs.len != self.order + 1) return error.InvalidACoefficients;
        if (b_coeffs.len != self.order + 1) return error.InvalidBCoefficients;
        @memcpy(self.a_coeffs, a_coeffs);
        @memcpy(self.b_coeffs, b_coeffs);
    }

    /// Processes one input sample and returns the filtered output sample.
    pub fn process(self: *IIRFilter, sample: f64) f64 {
        var result: f64 = 0.0;
        for (1..self.order + 1) |i| {
            result += self.b_coeffs[i] * self.input_history[i - 1] - self.a_coeffs[i] * self.output_history[i - 1];
        }
        result = (result + self.b_coeffs[0] * sample) / self.a_coeffs[0];

        if (self.order > 1) {
            var idx = self.order - 1;
            while (idx > 0) : (idx -= 1) {
                self.input_history[idx] = self.input_history[idx - 1];
                self.output_history[idx] = self.output_history[idx - 1];
            }
        }
        if (self.order > 0) {
            self.input_history[0] = sample;
            self.output_history[0] = result;
        }
        return result;
    }
};

fn expectApproxEqualSlices(expected: []const f64, actual: []const f64, tolerance: f64) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try testing.expectApproxEqAbs(e, a, tolerance);
    }
}

test "iir filter: init defaults" {
    var filt = try IIRFilter.init(testing.allocator, 2);
    defer filt.deinit();

    try expectApproxEqualSlices(&[_]f64{ 1.0, 0.0, 0.0 }, filt.a_coeffs, 1e-12);
    try expectApproxEqualSlices(&[_]f64{ 1.0, 0.0, 0.0 }, filt.b_coeffs, 1e-12);
    try expectApproxEqualSlices(&[_]f64{ 0.0, 0.0 }, filt.input_history, 1e-12);
    try expectApproxEqualSlices(&[_]f64{ 0.0, 0.0 }, filt.output_history, 1e-12);
}

test "iir filter: invalid coefficient lengths" {
    var filt = try IIRFilter.init(testing.allocator, 2);
    defer filt.deinit();

    try testing.expectError(error.InvalidACoefficients, filt.setCoefficients(&[_]f64{ 1.0, 2.0 }, &[_]f64{ 1.0, 2.0, 3.0 }));
    try testing.expectError(error.InvalidBCoefficients, filt.setCoefficients(&[_]f64{ 1.0, 2.0, 3.0 }, &[_]f64{ 1.0, 2.0 }));
}

test "iir filter: process default passthrough" {
    var filt = try IIRFilter.init(testing.allocator, 2);
    defer filt.deinit();

    try testing.expectApproxEqAbs(0.5, filt.process(0.5), 1e-12);
    try testing.expectApproxEqAbs(-0.25, filt.process(-0.25), 1e-12);
}

test "iir filter: process known lowpass impulse" {
    var filt = try IIRFilter.init(testing.allocator, 2);
    defer filt.deinit();
    try filt.setCoefficients(
        &[_]f64{ 1.0922959556412573, -1.9828897227476208, 0.9077040443587427 },
        &[_]f64{ 0.004277569313094809, 0.008555138626189618, 0.004277569313094809 },
    );

    try testing.expectApproxEqAbs(0.003916126660547383, filt.process(1.0), 1e-12);
    try testing.expectApproxEqAbs(0.014941358933061078, filt.process(0.0), 1e-12);
    try testing.expectApproxEqAbs(0.02778546621966332, filt.process(0.0), 1e-12);
}

test "iir filter: extreme stable constant input" {
    var filt = try IIRFilter.init(testing.allocator, 4);
    defer filt.deinit();
    try filt.setCoefficients(
        &[_]f64{ 1.0, -0.3, 0.2, -0.1, 0.05 },
        &[_]f64{ 0.2, 0.1, 0.05, 0.025, 0.0125 },
    );

    var sample: f64 = 0.0;
    for (0..10_000) |_| {
        sample = filt.process(0.25);
        try testing.expect(std.math.isFinite(sample));
    }
}
