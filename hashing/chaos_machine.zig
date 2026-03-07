//! Chaos Machine PRNG - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/hashes/chaos_machine.py

const std = @import("std");
const testing = std.testing;

pub const ChaosMachineError = error{
    ZeroBufferValue,
};

pub const ChaosMachine = struct {
    const m: usize = 5;
    const t: usize = 3;
    const initial_k = [_]f64{ 0.33, 0.44, 0.55, 0.44, 0.33 };

    buffer_space: [m]f64,
    params_space: [m]f64,
    machine_time: u64,

    pub fn init() ChaosMachine {
        return ChaosMachine{
            .buffer_space = initial_k,
            .params_space = [_]f64{0} ** m,
            .machine_time = 0,
        };
    }

    pub fn reset(self: *ChaosMachine) void {
        self.buffer_space = initial_k;
        self.params_space = [_]f64{0} ** m;
        self.machine_time = 0;
    }

    fn roundTo10(value: f64) f64 {
        return @round(value * 1e10) / 1e10;
    }

    fn fracPart01(value: f64) f64 {
        var r = @mod(value, 1.0);
        if (r < 0) r += 1.0;
        return r;
    }

    fn xorshift(x: u64, y: u64) u64 {
        var xx = x;
        var yy = y;
        xx ^= yy >> 13;
        yy ^= xx << 17;
        xx ^= yy >> 5;
        return xx;
    }

    /// Pushes one seed value into chaotic state.
    ///
    /// Time complexity: O(1)
    /// Space complexity: O(1)
    pub fn push(self: *ChaosMachine, seed: u32) ChaosMachineError!void {
        for (0..m) |key| {
            const value = self.buffer_space[key];
            if (value == 0.0) {
                return ChaosMachineError.ZeroBufferValue;
            }

            const e = @as(f64, @floatFromInt(seed)) / value;
            const evolved = fracPart01(self.buffer_space[(key + 1) % m] + e);
            const r = fracPart01(self.params_space[key] + e) + 3.0;

            self.buffer_space[key] = roundTo10(r * evolved * (1.0 - evolved));
            self.params_space[key] = r;
        }
        self.machine_time += 1;
    }

    /// Pulls one pseudorandom 32-bit value.
    ///
    /// Time complexity: O(1)
    /// Space complexity: O(1)
    pub fn pull(self: *ChaosMachine) u32 {
        const key: usize = @intCast(self.machine_time % m);

        for (0..t) |_| {
            const r = self.params_space[key];
            const value = self.buffer_space[key];

            self.buffer_space[key] = roundTo10(r * value * (1.0 - value));
            self.params_space[key] = fracPart01(@as(f64, @floatFromInt(self.machine_time)) * 0.01 + r * 1.01) + 3.0;
        }

        const x: u64 = @intFromFloat(self.buffer_space[(key + 2) % m] * 1e10);
        const y: u64 = @intFromFloat(self.buffer_space[(key + m - 2) % m] * 1e10);

        self.machine_time += 1;
        return @as(u32, @intCast(xorshift(x, y) % 0xFFFFFFFF));
    }
};

test "chaos machine: python deterministic sequence" {
    var cm = ChaosMachine.init();

    try testing.expectEqual(@as(u64, 0), cm.machine_time);
    try testing.expectApproxEqAbs(@as(f64, 0.33), cm.buffer_space[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), cm.params_space[0], 1e-12);

    try cm.push(12345);
    try cm.push(67890);
    try cm.push(42);
    try cm.push(987654321);

    const expected_buffer = [_]f64{ 0.9070077181, 0.6845885647, 0.6290548754, 0.4103739540, 0.8544145620 };
    const expected_params = [_]f64{ 3.8922653198242188, 3.22891902923584, 3.5207910537719727, 3.2320261001586914, 3.479146957397461 };

    for (expected_buffer, 0..) |v, i| {
        try testing.expectApproxEqAbs(v, cm.buffer_space[i], 1e-10);
    }
    for (expected_params, 0..) |v, i| {
        try testing.expectApproxEqAbs(v, cm.params_space[i], 1e-12);
    }
    try testing.expectEqual(@as(u64, 4), cm.machine_time);

    try testing.expectEqual(@as(u32, 2516730783), cm.pull());
    try testing.expectEqual(@as(u32, 698192756), cm.pull());
    try testing.expectEqual(@as(u32, 720813214), cm.pull());
    try testing.expectEqual(@as(u32, 335383369), cm.pull());
    try testing.expectEqual(@as(u32, 3866805556), cm.pull());
}

test "chaos machine: reset and boundary behavior" {
    var cm = ChaosMachine.init();
    for (0..100) |i| {
        try cm.push(@intCast(i + 1));
    }
    const sample = cm.pull();
    try testing.expect(sample <= 0xFFFFFFFF - 1);

    cm.reset();
    try testing.expectEqual(@as(u64, 0), cm.machine_time);
    try testing.expectApproxEqAbs(@as(f64, 0.33), cm.buffer_space[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), cm.params_space[4], 1e-12);
}
