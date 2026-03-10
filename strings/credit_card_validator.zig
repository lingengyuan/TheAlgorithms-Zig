//! Credit Card Validator - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/credit_card_validator.py

const std = @import("std");
const testing = std.testing;

/// Checks whether the credit card prefix is accepted by the Python reference.
/// Time complexity: O(1), Space complexity: O(1)
pub fn validateInitialDigits(credit_card_number: []const u8) bool {
    return std.mem.startsWith(u8, credit_card_number, "34") or
        std.mem.startsWith(u8, credit_card_number, "35") or
        std.mem.startsWith(u8, credit_card_number, "37") or
        std.mem.startsWith(u8, credit_card_number, "4") or
        std.mem.startsWith(u8, credit_card_number, "5") or
        std.mem.startsWith(u8, credit_card_number, "6");
}

pub fn luhnValidation(credit_card_number: []const u8) bool {
    var total: u32 = 0;
    var double_digit = false;
    var index = credit_card_number.len;
    while (index > 0) {
        index -= 1;
        const char = credit_card_number[index];
        if (!std.ascii.isDigit(char)) return false;
        var digit: u32 = char - '0';
        if (double_digit) {
            digit *= 2;
            if (digit > 9) digit = (digit % 10) + 1;
        }
        total += digit;
        double_digit = !double_digit;
    }
    return total % 10 == 0;
}

pub fn validateCreditCardNumber(credit_card_number: []const u8) bool {
    for (credit_card_number) |char| {
        if (!std.ascii.isDigit(char)) return false;
    }
    if (credit_card_number.len < 13 or credit_card_number.len > 16) return false;
    if (!validateInitialDigits(credit_card_number)) return false;
    return luhnValidation(credit_card_number);
}

test "credit card validator: python samples" {
    try testing.expect(validateInitialDigits("4111111111111111"));
    try testing.expect(validateInitialDigits("34"));
    try testing.expect(!validateInitialDigits("14"));

    try testing.expect(luhnValidation("4111111111111111"));
    try testing.expect(luhnValidation("36111111111111"));
    try testing.expect(!luhnValidation("41111111111111"));

    try testing.expect(validateCreditCardNumber("4111111111111111"));
    try testing.expect(!validateCreditCardNumber("helloworld$"));
    try testing.expect(!validateCreditCardNumber("32323"));
    try testing.expect(!validateCreditCardNumber("32323323233232332323"));
    try testing.expect(!validateCreditCardNumber("36111111111111"));
    try testing.expect(!validateCreditCardNumber("41111111111111"));
}

test "credit card validator: edge and extreme" {
    try testing.expect(validateCreditCardNumber("378282246310005"));
    try testing.expect(validateCreditCardNumber("5555555555554444"));
    try testing.expect(!validateCreditCardNumber("0000000000000000"));
    try testing.expect(!validateCreditCardNumber("4"));
}
