//! Detecting English Programmatically - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/detecting_english_programmatically.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const ascii_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
pub const letters_and_space = ascii_letters ++ " \t\n";

/// Removes every character that is not an ASCII letter or whitespace accepted by the Python reference.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn removeNonLetters(allocator: Allocator, message: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(allocator);
    for (message) |char| {
        if (std.mem.indexOfScalar(u8, letters_and_space, char) != null) {
            try out.append(allocator, char);
        }
    }
    return out.toOwnedSlice(allocator);
}

pub fn getEnglishCount(allocator: Allocator, message: []const u8, dictionary: []const []const u8) !f64 {
    const filtered = try removeNonLetters(allocator, message);
    defer allocator.free(filtered);

    const uppercase = try allocator.dupe(u8, filtered);
    defer allocator.free(uppercase);
    _ = std.ascii.upperString(uppercase, filtered);

    var tokenizer = std.mem.tokenizeAny(u8, uppercase, " \t\n");
    var total_words: usize = 0;
    var matches: usize = 0;
    while (tokenizer.next()) |word| {
        total_words += 1;
        for (dictionary) |entry| {
            if (std.ascii.eqlIgnoreCase(word, entry)) {
                matches += 1;
                break;
            }
        }
    }
    if (total_words == 0) return 0.0;
    return @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(total_words));
}

pub fn isEnglish(
    allocator: Allocator,
    message: []const u8,
    dictionary: []const []const u8,
    word_percentage: u32,
    letter_percentage: u32,
) !bool {
    const words_match = (try getEnglishCount(allocator, message, dictionary)) * 100.0 >= @as(f64, @floatFromInt(word_percentage));
    const filtered = try removeNonLetters(allocator, message);
    defer allocator.free(filtered);
    if (message.len == 0) return false;
    const message_letters_percentage = (@as(f64, @floatFromInt(filtered.len)) / @as(f64, @floatFromInt(message.len))) * 100.0;
    return words_match and message_letters_percentage >= @as(f64, @floatFromInt(letter_percentage));
}

test "detecting english: remove non letters" {
    const one = try removeNonLetters(testing.allocator, "Hi! how are you?");
    defer testing.allocator.free(one);
    try testing.expectEqualStrings("Hi how are you", one);

    const two = try removeNonLetters(testing.allocator, "P^y%t)h@o*n");
    defer testing.allocator.free(two);
    try testing.expectEqualStrings("Python", two);

    const three = try removeNonLetters(testing.allocator, "1+1=2");
    defer testing.allocator.free(three);
    try testing.expectEqualStrings("", three);
}

test "detecting english: english count and classification" {
    const dictionary = [_][]const u8{ "HELLO", "WORLD", "HOW", "ARE", "YOU" };
    try testing.expectApproxEqRel(1.0, try getEnglishCount(testing.allocator, "Hello World", &dictionary), 1e-12);
    try testing.expect(try isEnglish(testing.allocator, "Hello World", &dictionary, 20, 85));
    try testing.expect(!(try isEnglish(testing.allocator, "llold HorWd", &dictionary, 20, 85)));
}
