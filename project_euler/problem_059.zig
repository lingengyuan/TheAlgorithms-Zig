//! Project Euler Problem 59: XOR Decryption - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_059/sol1.py

const std = @import("std");
const testing = std.testing;

const Allocator = std.mem.Allocator;
const cipher_file = @embedFile("problem_059_cipher.txt");

pub const Problem059Error = error{
    OutOfMemory,
    InvalidCiphertext,
    NoSolution,
};

const COMMON_WORDS = [_][]const u8{ "the", "be", "to", "of", "and", "in", "that", "have" };

fn isValidDecodedChar(ch: u32) bool {
    return switch (ch) {
        'a'...'z', 'A'...'Z', '0'...'9' => true,
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ', '\t', '\n', '\r', '\x0b', '\x0c' => true,
        else => false,
    };
}

pub fn parseCiphertext(allocator: Allocator, data: []const u8) Problem059Error![]u32 {
    var values = std.ArrayListUnmanaged(u32){};
    errdefer values.deinit(allocator);

    var tokens = std.mem.tokenizeAny(u8, data, ",\r\n ");
    while (tokens.next()) |token| {
        const value = std.fmt.parseInt(u32, token, 10) catch return error.InvalidCiphertext;
        try values.append(allocator, value);
    }
    if (values.items.len == 0) return error.InvalidCiphertext;
    return values.toOwnedSlice(allocator);
}

/// Tries a three-byte lowercase key and returns the decoded text when every decoded
/// character is within the Python reference's allowed ASCII set.
/// Caller owns the returned slice.
/// Time complexity: O(n)
/// Space complexity: O(n)
pub fn tryKey(allocator: Allocator, ciphertext: []const u32, key: [3]u8) Problem059Error!?[]u8 {
    var decoded = try allocator.alloc(u8, ciphertext.len);
    errdefer allocator.free(decoded);

    for (ciphertext, 0..) |value, idx| {
        const decoded_value = value ^ key[idx % key.len];
        if (decoded_value > 255 or !isValidDecodedChar(decoded_value)) {
            allocator.free(decoded);
            return null;
        }
        decoded[idx] = @intCast(decoded_value);
    }

    return decoded;
}

fn containsAsciiCaseInsensitive(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;

    var start: usize = 0;
    while (start + needle.len <= haystack.len) : (start += 1) {
        var matched = true;
        for (needle, 0..) |ch, offset| {
            if (std.ascii.toLower(haystack[start + offset]) != ch) {
                matched = false;
                break;
            }
        }
        if (matched) return true;
    }
    return false;
}

fn sumAscii(text: []const u8) usize {
    var total: usize = 0;
    for (text) |ch| total += ch;
    return total;
}

/// Returns the sum of the ASCII values in the uniquely identified decoded message.
/// Time complexity: O(26^3 * n * common_words)
/// Space complexity: O(candidates * n)
pub fn solveData(allocator: Allocator, data: []const u8) Problem059Error!usize {
    const ciphertext = try parseCiphertext(allocator, data);
    defer allocator.free(ciphertext);

    var possible_messages = std.ArrayListUnmanaged([]u8){};
    defer {
        for (possible_messages.items) |message| allocator.free(message);
        possible_messages.deinit(allocator);
    }

    var first: u8 = 'a';
    while (first <= 'z') : (first += 1) {
        var second: u8 = 'a';
        while (second <= 'z') : (second += 1) {
            var third: u8 = 'a';
            while (third <= 'z') : (third += 1) {
                const decoded = try tryKey(allocator, ciphertext, .{ first, second, third });
                if (decoded) |message| try possible_messages.append(allocator, message);
            }
        }
    }

    for (COMMON_WORDS) |common_word| {
        var write_idx: usize = 0;
        for (possible_messages.items) |message| {
            if (containsAsciiCaseInsensitive(message, common_word)) {
                possible_messages.items[write_idx] = message;
                write_idx += 1;
            } else {
                allocator.free(message);
            }
        }
        possible_messages.items.len = write_idx;
        if (write_idx == 1) break;
    }

    if (possible_messages.items.len == 0) return error.NoSolution;
    return sumAscii(possible_messages.items[0]);
}

pub fn solution(allocator: Allocator) Problem059Error!usize {
    return solveData(allocator, cipher_file);
}

test "problem 059: python reference dataset" {
    try testing.expectEqual(@as(usize, 129448), try solution(testing.allocator));
}

test "problem 059: helper semantics" {
    const ciphertext = [_]u32{ 0, 17, 20, 4, 27 };
    const decoded = (try tryKey(testing.allocator, &ciphertext, .{ 'h', 't', 'x' })).?;
    defer testing.allocator.free(decoded);
    try testing.expectEqualStrings("hello", decoded);

    const invalid = [_]u32{ 68, 10, 300, 4, 27 };
    try testing.expect((try tryKey(testing.allocator, &invalid, .{ 'h', 't', 'x' })) == null);
}

test "problem 059: sample cipher and edge parsing" {
    try testing.expectEqual(@as(usize, 3000), try solveData(testing.allocator, @embedFile("problem_059_test_cipher.txt")));
    try testing.expectError(error.InvalidCiphertext, parseCiphertext(testing.allocator, ""));
    try testing.expect(containsAsciiCaseInsensitive("The enemy's gate is down", "the"));
    try testing.expect(!containsAsciiCaseInsensitive("xyz", "the"));
}
