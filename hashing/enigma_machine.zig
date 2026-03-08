//! Enigma Machine (ASCII 32-125) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/hashes/enigma_machine.py

const std = @import("std");
const testing = std.testing;

pub const EnigmaMachineError = error{
    CharacterOutOfRange,
};

const alphabet_start: u8 = 32;
const alphabet_len: usize = 94; // ASCII 32..125 inclusive

fn indexOfValue(haystack: *const [alphabet_len]u8, needle: u8) usize {
    for (haystack, 0..) |value, i| {
        if (value == needle) return i;
    }
    unreachable;
}

pub const EnigmaMachine = struct {
    gear_one: [alphabet_len]u8,
    gear_two: [alphabet_len]u8,
    gear_three: [alphabet_len]u8,
    gear_one_pos: usize,
    gear_two_pos: usize,
    gear_three_pos: usize,

    pub fn init() EnigmaMachine {
        var machine = EnigmaMachine{
            .gear_one = undefined,
            .gear_two = undefined,
            .gear_three = undefined,
            .gear_one_pos = 0,
            .gear_two_pos = 0,
            .gear_three_pos = 0,
        };

        for (0..alphabet_len) |i| {
            const value: u8 = @intCast(i);
            machine.gear_one[i] = value;
            machine.gear_two[i] = value;
            machine.gear_three[i] = value;
        }

        return machine;
    }

    fn rotator(self: *EnigmaMachine) void {
        const g1 = self.gear_one[0];
        std.mem.copyForwards(u8, self.gear_one[0 .. alphabet_len - 1], self.gear_one[1..]);
        self.gear_one[alphabet_len - 1] = g1;
        self.gear_one_pos += 1;

        if (self.gear_one_pos % alphabet_len == 0) {
            const g2 = self.gear_two[0];
            std.mem.copyForwards(u8, self.gear_two[0 .. alphabet_len - 1], self.gear_two[1..]);
            self.gear_two[alphabet_len - 1] = g2;
            self.gear_two_pos += 1;

            if (self.gear_two_pos % alphabet_len == 0) {
                const g3 = self.gear_three[0];
                std.mem.copyForwards(u8, self.gear_three[0 .. alphabet_len - 1], self.gear_three[1..]);
                self.gear_three[alphabet_len - 1] = g3;
                self.gear_three_pos += 1;
            }
        }
    }

    pub fn setToken(self: *EnigmaMachine, token: usize) void {
        for (0..token) |_| {
            self.rotator();
        }
    }

    pub fn encodeChar(self: *EnigmaMachine, input_character: u8) EnigmaMachineError!u8 {
        if (input_character < alphabet_start or input_character >= alphabet_start + alphabet_len) {
            return EnigmaMachineError.CharacterOutOfRange;
        }

        var target: usize = input_character - alphabet_start;
        target = self.gear_one[target];
        target = self.gear_two[target];
        target = self.gear_three[target];
        target = (alphabet_len - 1) - target;
        target = indexOfValue(&self.gear_three, @intCast(target));
        target = indexOfValue(&self.gear_two, @intCast(target));
        target = indexOfValue(&self.gear_one, @intCast(target));

        const out: u8 = alphabet_start + @as(u8, @intCast(target));
        self.rotator();
        return out;
    }

    /// Encodes/decodes input using the current machine state.
    /// Caller owns returned buffer.
    ///
    /// Time complexity: O(n * alphabet_len)
    /// Space complexity: O(n)
    pub fn process(self: *EnigmaMachine, allocator: std.mem.Allocator, message: []const u8) ![]u8 {
        const out = try allocator.alloc(u8, message.len);
        errdefer allocator.free(out);

        for (message, 0..) |ch, i| {
            out[i] = try self.encodeChar(ch);
        }

        return out;
    }
};

/// Convenience wrapper matching Python script flow:
/// initialize machine, rotate token steps, then process message.
pub fn encode(allocator: std.mem.Allocator, message: []const u8, token: usize) ![]u8 {
    var machine = EnigmaMachine.init();
    machine.setToken(token);
    return machine.process(allocator, message);
}

test "enigma machine: python deterministic vectors" {
    const alloc = testing.allocator;

    const e0 = try encode(alloc, "Hello, World!", 0);
    defer alloc.free(e0);
    try testing.expectEqualStrings("U6-+&gq8|w{#d", e0);

    const e1 = try encode(alloc, "Hello, World!", 1);
    defer alloc.free(e1);
    try testing.expectEqualStrings("S4+)$eo6zuy!b", e1);

    const e123 = try encode(alloc, "Hello, World!", 123);
    defer alloc.free(e123);
    try testing.expectEqualStrings("wXOMH+5Z@;?E(", e123);

    const edge0 = try encode(alloc, " }", 0);
    defer alloc.free(edge0);
    try testing.expectEqualStrings("}|", edge0);

    const edge1 = try encode(alloc, " }", 1);
    defer alloc.free(edge1);
    try testing.expectEqualStrings("{z", edge1);
}

test "enigma machine: symmetric decoding and extreme length" {
    const alloc = testing.allocator;

    const encrypted = try encode(alloc, "TheAlgorithms-Zig", 42);
    defer alloc.free(encrypted);
    try testing.expectEqualStrings("S=>`36,'.!+$z`1  ", encrypted);

    const decrypted = try encode(alloc, encrypted, 42);
    defer alloc.free(decrypted);
    try testing.expectEqualStrings("TheAlgorithms-Zig", decrypted);

    const n: usize = 20_000;
    const input = try alloc.alloc(u8, n);
    defer alloc.free(input);
    for (input, 0..) |*ch, i| {
        ch.* = switch (i % 5) {
            0 => 'A',
            1 => 'z',
            2 => ' ',
            3 => '}',
            else => '0',
        };
    }

    const long_enc = try encode(alloc, input, 777);
    defer alloc.free(long_enc);
    const long_dec = try encode(alloc, long_enc, 777);
    defer alloc.free(long_dec);

    try testing.expectEqualSlices(u8, input, long_dec);
}

test "enigma machine: invalid character" {
    try testing.expectError(EnigmaMachineError.CharacterOutOfRange, encode(testing.allocator, "~", 0));
}
