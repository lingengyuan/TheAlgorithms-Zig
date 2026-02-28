//! Python vs Zig benchmark harness (Zig side, alignable algorithms).

const std = @import("std");
const Allocator = std.mem.Allocator;

const bubble_sort = @import("../../sorts/bubble_sort.zig");
const insertion_sort = @import("../../sorts/insertion_sort.zig");
const merge_sort = @import("../../sorts/merge_sort.zig");
const quick_sort = @import("../../sorts/quick_sort.zig");
const heap_sort = @import("../../sorts/heap_sort.zig");
const radix_sort = @import("../../sorts/radix_sort.zig");
const bucket_sort = @import("../../sorts/bucket_sort.zig");
const selection_sort = @import("../../sorts/selection_sort.zig");
const shell_sort = @import("../../sorts/shell_sort.zig");
const counting_sort = @import("../../sorts/counting_sort.zig");
const cocktail_shaker_sort = @import("../../sorts/cocktail_shaker_sort.zig");
const gnome_sort = @import("../../sorts/gnome_sort.zig");

const linear_search = @import("../../searches/linear_search.zig");
const binary_search = @import("../../searches/binary_search.zig");
const exponential_search = @import("../../searches/exponential_search.zig");
const interpolation_search = @import("../../searches/interpolation_search.zig");
const jump_search = @import("../../searches/jump_search.zig");
const ternary_search = @import("../../searches/ternary_search.zig");

const gcd_mod = @import("../../maths/gcd.zig");
const lcm_mod = @import("../../maths/lcm.zig");
const fibonacci_mod = @import("../../maths/fibonacci.zig");
const factorial_mod = @import("../../maths/factorial.zig");
const power_mod = @import("../../maths/power.zig");
const prime_check_mod = @import("../../maths/prime_check.zig");
const sieve_mod = @import("../../maths/sieve_of_eratosthenes.zig");
const collatz_mod = @import("../../maths/collatz_sequence.zig");

const climbing_stairs_mod = @import("../../dynamic_programming/climbing_stairs.zig");
const fibonacci_dp_mod = @import("../../dynamic_programming/fibonacci_dp.zig");
const coin_change_mod = @import("../../dynamic_programming/coin_change.zig");
const max_subarray_mod = @import("../../dynamic_programming/max_subarray_sum.zig");
const lcs_mod = @import("../../dynamic_programming/longest_common_subsequence.zig");
const edit_distance_mod = @import("../../dynamic_programming/edit_distance.zig");
const knapsack_mod = @import("../../dynamic_programming/knapsack.zig");

const bfs_mod = @import("../../graphs/bfs.zig");
const dfs_mod = @import("../../graphs/dfs.zig");
const dijkstra_mod = @import("../../graphs/dijkstra.zig");

const is_power_two_mod = @import("../../bit_manipulation/is_power_of_two.zig");
const count_set_bits_mod = @import("../../bit_manipulation/count_set_bits.zig");
const find_unique_mod = @import("../../bit_manipulation/find_unique_number.zig");
const reverse_bits_mod = @import("../../bit_manipulation/reverse_bits.zig");
const missing_number_mod = @import("../../bit_manipulation/missing_number.zig");
const power_of_4_mod = @import("../../bit_manipulation/power_of_4.zig");

const dec_to_bin_mod = @import("../../conversions/decimal_to_binary.zig");
const bin_to_dec_mod = @import("../../conversions/binary_to_decimal.zig");
const dec_to_hex_mod = @import("../../conversions/decimal_to_hexadecimal.zig");
const bin_to_hex_mod = @import("../../conversions/binary_to_hexadecimal.zig");

const palindrome_mod = @import("../../strings/palindrome.zig");
const reverse_words_mod = @import("../../strings/reverse_words.zig");
const anagram_mod = @import("../../strings/anagram.zig");
const hamming_mod = @import("../../strings/hamming_distance.zig");
const naive_search_mod = @import("../../strings/naive_string_search.zig");
const kmp_mod = @import("../../strings/knuth_morris_pratt.zig");
const rabin_karp_mod = @import("../../strings/rabin_karp.zig");
const z_function_mod = @import("../../strings/z_function.zig");
const levenshtein_mod = @import("../../strings/levenshtein_distance.zig");
const is_pangram_mod = @import("../../strings/is_pangram.zig");

const best_stock_mod = @import("../../greedy_methods/best_time_to_buy_sell_stock.zig");
const min_coin_mod = @import("../../greedy_methods/minimum_coin_change.zig");
const min_waiting_mod = @import("../../greedy_methods/minimum_waiting_time.zig");
const fractional_knapsack_mod = @import("../../greedy_methods/fractional_knapsack.zig");

const matrix_mul_mod = @import("../../matrix/matrix_multiply.zig");
const transpose_mod = @import("../../matrix/matrix_transpose.zig");
const rotate_mod = @import("../../matrix/rotate_matrix.zig");
const spiral_mod = @import("../../matrix/spiral_print.zig");
const pascal_mod = @import("../../matrix/pascal_triangle.zig");

fn intToU64(comptime T: type, v: T) u64 {
    const info = @typeInfo(T).int;
    if (info.signedness == .signed) {
        const as_i64: i64 = @intCast(v);
        return @bitCast(as_i64);
    }
    return @as(u64, @intCast(v));
}

fn checksumSlice(comptime T: type, arr: []const T) u64 {
    if (arr.len == 0) return 0;
    return intToU64(T, arr[0]) +
        intToU64(T, arr[arr.len / 2]) +
        intToU64(T, arr[arr.len - 1]) +
        @as(u64, @intCast(arr.len));
}

fn checksumBytes(bytes: []const u8) u64 {
    if (bytes.len == 0) return 0;
    return @as(u64, bytes[0]) + @as(u64, bytes[bytes.len / 2]) + @as(u64, bytes[bytes.len - 1]) + @as(u64, @intCast(bytes.len));
}

fn checksumBool(v: bool) u64 {
    return if (v) 1 else 0;
}

fn valueToU64(value: anytype) u64 {
    const T = @TypeOf(value);
    return switch (@typeInfo(T)) {
        .bool => checksumBool(value),
        .int => intToU64(T, value),
        else => @compileError("Unsupported benchmark return type; expected bool/int/u64-compatible"),
    };
}

fn generateIntData(allocator: Allocator, n: usize) ![]i32 {
    const data = try allocator.alloc(i32, n);
    for (0..n) |i| {
        const idx: u64 = @intCast(i);
        const raw: u64 = (idx * 48_271 + 12_345) % 1_000_003;
        const signed: i64 = @intCast(raw);
        data[i] = @intCast(signed - 500_000);
    }
    return data;
}

fn generateNonNegativeData(allocator: Allocator, n: usize) ![]i32 {
    const data = try allocator.alloc(i32, n);
    for (0..n) |i| {
        const idx: u64 = @intCast(i);
        const raw: u64 = (idx * 48_271 + 12_345) % 100_000;
        data[i] = @intCast(raw);
    }
    return data;
}

fn generateIntData64(allocator: Allocator, n: usize) ![]i64 {
    const data = try allocator.alloc(i64, n);
    for (0..n) |i| {
        const idx: u64 = @intCast(i);
        const raw: u64 = (idx * 48_271 + 12_345) % 1_000_003;
        const signed: i64 = @intCast(raw);
        data[i] = signed - 500_000;
    }
    return data;
}

fn generateSortedData(allocator: Allocator, n: usize) ![]i32 {
    const data = try allocator.alloc(i32, n);
    for (0..n) |i| {
        const value: i64 = @intCast(i * 2);
        data[i] = @intCast(value);
    }
    return data;
}

fn generateSearchQueries(allocator: Allocator, query_count: usize, n: usize) ![]i32 {
    const queries = try allocator.alloc(i32, query_count);
    const n_u64: u64 = @intCast(n);
    for (0..query_count) |i| {
        const idx: u64 = @intCast(i);
        const raw = ((idx * 97) + 31) % n_u64;
        const q: i64 = @intCast(raw * 2);
        queries[i] = @intCast(q);
    }
    return queries;
}

fn generateU64Data(allocator: Allocator, n: usize) ![]u64 {
    const data = try allocator.alloc(u64, n);
    for (0..n) |i| {
        const idx: u64 = @intCast(i);
        data[i] = ((idx * 73) + 19) % 1_000_000;
    }
    return data;
}

fn generateAsciiString(allocator: Allocator, n: usize, mul: u64, add: u64) ![]u8 {
    const s = try allocator.alloc(u8, n);
    for (0..n) |i| {
        const idx: u64 = @intCast(i);
        const raw = ((idx * mul) + add) % 26;
        s[i] = @as(u8, 'a') + @as(u8, @intCast(raw));
    }
    return s;
}

fn generateMatrixData(allocator: Allocator, n: usize, mul: u64, add: u64, mod_val: u64, shift: i64) ![]i64 {
    const data = try allocator.alloc(i64, n);
    for (0..n) |i| {
        const idx: u64 = @intCast(i);
        const raw = ((idx * mul) + add) % mod_val;
        const as_i64: i64 = @intCast(raw);
        data[i] = as_i64 - shift;
    }
    return data;
}

fn repeatString(allocator: Allocator, unit: []const u8, times: usize) ![]u8 {
    const out = try allocator.alloc(u8, unit.len * times);
    var pos: usize = 0;
    for (0..times) |_| {
        @memcpy(out[pos .. pos + unit.len], unit);
        pos += unit.len;
    }
    return out;
}

fn printResult(writer: anytype, algorithm: []const u8, category: []const u8, iterations: usize, total_ns: u64, checksum: u64) !void {
    const avg_ns = total_ns / @as(u64, @intCast(iterations));
    try writer.print("{s},{s},{d},{d},{d},{d}\n", .{ algorithm, category, iterations, total_ns, avg_ns, checksum });
}

fn callU64(comptime Func: anytype, args: anytype) !u64 {
    return switch (@typeInfo(@TypeOf(@call(.auto, Func, args)))) {
        .error_union => blk: {
            const v = try @call(.auto, Func, args);
            break :blk valueToU64(v);
        },
        else => blk: {
            const v = @call(.auto, Func, args);
            break :blk valueToU64(v);
        },
    };
}

fn benchRun(comptime Func: anytype, writer: anytype, name: []const u8, category: []const u8, iterations: usize, args: anytype) !void {
    _ = try callU64(Func, args);
    var checksum: u64 = 0;
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| checksum +%= try callU64(Func, args);
    try printResult(writer, name, category, iterations, timer.read(), checksum);
}

fn benchRunMaybe(filter: ?[]const u8, matched: *bool, comptime Func: anytype, writer: anytype, name: []const u8, category: []const u8, iterations: usize, args: anytype) !void {
    if (filter) |selected| {
        if (!std.mem.eql(u8, selected, name)) return;
        matched.* = true;
    }
    try benchRun(Func, writer, name, category, iterations, args);
}

fn runSortInPlace(comptime SortFn: anytype, allocator: Allocator, base: []const i32) !u64 {
    const working = try allocator.dupe(i32, base);
    defer allocator.free(working);
    SortFn(i32, working);
    return checksumSlice(i32, working);
}

fn runSortAllocated(comptime SortFn: anytype, allocator: Allocator, base: []const i32) !u64 {
    const sorted = try SortFn(allocator, base);
    defer allocator.free(sorted);
    return checksumSlice(i32, sorted);
}

fn runMergeSort(allocator: Allocator, base: []const i32) !u64 {
    const sorted = try merge_sort.mergeSort(i32, allocator, base);
    defer allocator.free(sorted);
    return checksumSlice(i32, sorted);
}

fn runSearch(comptime SearchFn: anytype, data: []const i32, queries: []const i32) !u64 {
    var checksum: u64 = 0;
    for (queries) |q| {
        if (SearchFn(i32, data, q)) |idx| {
            checksum +%= @intCast(idx + 1);
        }
    }
    return checksum;
}

fn runInterpolationSearch(data: []const i32, queries: []const i32) u64 {
    var checksum: u64 = 0;
    for (queries) |q| {
        if (interpolation_search.interpolationSearch(data, q)) |idx| {
            checksum +%= @intCast(idx + 1);
        }
    }
    return checksum;
}

fn runMathGcd(values: []const u64) u64 {
    var sum: u64 = 0;
    var i: usize = 0;
    while (i < 20_000) : (i += 2) {
        const a: i64 = @intCast(values[i]);
        const b: i64 = @intCast(values[i + 1]);
        sum +%= gcd_mod.gcd(a, b);
    }
    return sum;
}

fn runMathLcm(values: []const u64) u64 {
    var sum: u64 = 0;
    var i: usize = 0;
    while (i < 20_000) : (i += 2) {
        const a: i64 = @intCast(values[i] + 1);
        const b: i64 = @intCast(values[i + 1] + 1);
        sum +%= lcm_mod.lcm(a, b);
    }
    return sum;
}

fn runPrimeCheck() u64 {
    var count: u64 = 0;
    var i: u64 = 2;
    while (i < 120_000) : (i += 1) {
        if (prime_check_mod.isPrime(i)) count += 1;
    }
    return count;
}

fn runFibonacciMany(inputs: []const u32) u64 {
    var sum: u64 = 0;
    for (inputs) |n| sum +%= fibonacci_mod.fibonacci(n);
    return sum;
}

fn runFactorialMany(inputs: []const u32) u64 {
    var sum: u64 = 0;
    for (inputs) |n| sum +%= factorial_mod.factorial(n);
    return sum;
}

fn runPowerMany(bases: []const i64, exponents: []const u32) u64 {
    var sum: u64 = 0;
    for (bases, exponents) |b, e| {
        sum +%= intToU64(i64, power_mod.power(b, e));
    }
    return sum;
}

fn runCollatzSteps() !u64 {
    var sum: u64 = 0;
    var i: u64 = 2;
    while (i < 60_000) : (i += 1) {
        sum +%= try collatz_mod.collatzSteps(i);
    }
    return sum;
}

fn runSieve(allocator: Allocator) !u64 {
    const primes = try sieve_mod.primeSieve(allocator, 300_000);
    defer allocator.free(primes);
    return checksumSlice(u64, primes);
}

fn runFibonacciDp(allocator: Allocator) !u64 {
    const seq = try fibonacci_dp_mod.fibonacciDp(allocator, 280);
    defer allocator.free(seq);
    return checksumSlice(u64, seq);
}

fn runCoinChange(allocator: Allocator, coins: []const u32) !u64 {
    return try coin_change_mod.coinChangeWays(allocator, coins, 420);
}

fn runClimbingStairsMany(inputs: []const i32) !u64 {
    var sum: u64 = 0;
    for (inputs) |n| sum +%= try climbing_stairs_mod.climbingStairs(n);
    return sum;
}

fn runMaxSubarray(dp_array: []const i64) u64 {
    return intToU64(i64, max_subarray_mod.maxSubarraySum(dp_array, false));
}

fn runLcs(allocator: Allocator, a: []const u8, b: []const u8) !u64 {
    return try lcs_mod.longestCommonSubsequenceLength(allocator, a, b);
}

fn runEditDistance(allocator: Allocator, a: []const u8, b: []const u8) !u64 {
    return try edit_distance_mod.editDistance(allocator, a, b);
}

fn runKnapsack(allocator: Allocator, weights: []const usize, values: []const usize) !u64 {
    return try knapsack_mod.knapsack(allocator, 800, weights, values);
}

fn buildGraphAdj(allocator: Allocator, n: usize) ![][]usize {
    const adj = try allocator.alloc([]usize, n);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| allocator.free(adj[i]);
        allocator.free(adj);
    }
    for (0..n) |i| {
        var count: usize = 0;
        if (i + 1 < n) count += 1;
        if (i + 2 < n) count += 1;
        if (i % 3 == 0 and i + 17 < n) count += 1;
        adj[i] = try allocator.alloc(usize, count);
        built += 1;
        var idx: usize = 0;
        if (i + 1 < n) {
            adj[i][idx] = i + 1;
            idx += 1;
        }
        if (i + 2 < n) {
            adj[i][idx] = i + 2;
            idx += 1;
        }
        if (i % 3 == 0 and i + 17 < n) {
            adj[i][idx] = i + 17;
            idx += 1;
        }
    }
    return adj;
}

fn freeGraphAdj(allocator: Allocator, adj: [][]usize) void {
    for (adj) |row| allocator.free(row);
    allocator.free(adj);
}

fn buildWeightedGraphAdj(allocator: Allocator, n: usize) ![][]dijkstra_mod.Edge {
    const adj = try allocator.alloc([]dijkstra_mod.Edge, n);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| allocator.free(adj[i]);
        allocator.free(adj);
    }
    for (0..n) |i| {
        var count: usize = 0;
        if (i + 1 < n) count += 1;
        if (i + 2 < n) count += 1;
        if (i % 3 == 0 and i + 17 < n) count += 1;

        adj[i] = try allocator.alloc(dijkstra_mod.Edge, count);
        built += 1;
        var idx: usize = 0;
        const i_u64: u64 = @intCast(i);

        if (i + 1 < n) {
            adj[i][idx] = .{
                .to = i + 1,
                .weight = ((i_u64 * 17) + 3) % 23 + 1,
            };
            idx += 1;
        }
        if (i + 2 < n) {
            adj[i][idx] = .{
                .to = i + 2,
                .weight = ((i_u64 * 31) + 7) % 29 + 1,
            };
            idx += 1;
        }
        if (i % 3 == 0 and i + 17 < n) {
            adj[i][idx] = .{
                .to = i + 17,
                .weight = ((i_u64 * 13) + 11) % 41 + 1,
            };
            idx += 1;
        }
    }
    return adj;
}

fn freeWeightedGraphAdj(allocator: Allocator, adj: [][]dijkstra_mod.Edge) void {
    for (adj) |row| allocator.free(row);
    allocator.free(adj);
}

fn runBfs(allocator: Allocator, adj: []const []const usize) !u64 {
    const order = try bfs_mod.bfs(allocator, adj, 0);
    defer allocator.free(order);
    return checksumSlice(usize, order);
}

fn runDfs(allocator: Allocator, adj: []const []const usize) !u64 {
    const order = try dfs_mod.dfs(allocator, adj, 0);
    defer allocator.free(order);
    return checksumSlice(usize, order);
}

fn runDijkstra(allocator: Allocator, adj: []const []const dijkstra_mod.Edge) !u64 {
    const dist = try dijkstra_mod.dijkstra(allocator, adj, 0);
    defer allocator.free(dist);
    return checksumSlice(u64, dist);
}

fn runIsPowerOfTwo() u64 {
    var count: u64 = 0;
    var i: u64 = 1;
    while (i < 3_000_000) : (i += 1) {
        if (is_power_two_mod.isPowerOfTwo(i)) count += 1;
    }
    return count;
}

fn runCountSetBits() u64 {
    var sum: u64 = 0;
    var i: u64 = 1;
    while (i < 2_000_000) : (i += 1) {
        sum +%= count_set_bits_mod.countSetBits(i);
    }
    return sum;
}

fn runReverseBits() u64 {
    var sum: u64 = 0;
    var i: u32 = 0;
    while (i < 200_000) : (i += 1) {
        sum +%= reverse_bits_mod.reverseBits(i);
    }
    return sum;
}

fn runPowerOfFour() u64 {
    var count: u64 = 0;
    var i: u64 = 1;
    while (i < 3_000_000) : (i += 1) {
        if (power_of_4_mod.isPowerOfFour(i)) count += 1;
    }
    return count;
}

fn runFindUnique() u64 {
    return intToU64(i32, (find_unique_mod.findUniqueNumber(&[_]i32{ 11, 7, 11, 3, 3 }) orelse 0));
}

fn runFindUniqueFromSlice(arr: []const i32) u64 {
    return intToU64(i32, (find_unique_mod.findUniqueNumber(arr) orelse 0));
}

fn runDecimalToBinary(allocator: Allocator) !u64 {
    const s = try dec_to_bin_mod.decimalToBinary(allocator, 987_654_321);
    defer allocator.free(s);
    return checksumBytes(s);
}

fn runBinaryToDecimal(bin_text: []const u8) !u64 {
    const value = try bin_to_dec_mod.binaryToDecimal(bin_text);
    return intToU64(i64, value);
}

fn runBinaryToDecimalMany(samples: []const []const u8) !u64 {
    var sum: u64 = 0;
    for (samples) |s| {
        const value = try bin_to_dec_mod.binaryToDecimal(s);
        sum +%= intToU64(i64, value);
    }
    return sum;
}

fn runDecimalToHex(allocator: Allocator) !u64 {
    const s = try dec_to_hex_mod.decimalToHex(allocator, 987_654_321);
    defer allocator.free(s);
    return checksumBytes(s);
}

fn runBinaryToHex(allocator: Allocator, bin_text: []const u8) !u64 {
    const s = try bin_to_hex_mod.binaryToHex(allocator, bin_text);
    defer allocator.free(s);
    return checksumBytes(s);
}

fn runReverseWords(allocator: Allocator, sentence: []const u8) !u64 {
    const s = try reverse_words_mod.reverseWords(allocator, sentence);
    defer allocator.free(s);
    return checksumBytes(s);
}

fn runPalindrome(text: []const u8) u64 {
    return checksumBool(palindrome_mod.isPalindrome(text));
}

fn runAnagram(a: []const u8, b: []const u8) u64 {
    return checksumBool(anagram_mod.isAnagram(a, b));
}

fn runHammingDistance(a: []const u8, b: []const u8) !u64 {
    return try hamming_mod.hammingDistance(a, b);
}

fn runNaiveSearch(allocator: Allocator, text: []const u8, pattern: []const u8) !u64 {
    const out = try naive_search_mod.naiveSearch(allocator, text, pattern);
    defer allocator.free(out);
    return checksumSlice(usize, out);
}

fn runKmpSearch(allocator: Allocator, text: []const u8, pattern: []const u8) !u64 {
    const idx = try kmp_mod.kmpSearch(allocator, text, pattern);
    return if (idx) |v| @intCast(v) else 0;
}

fn runZFunction(allocator: Allocator, text: []const u8) !u64 {
    const z = try z_function_mod.zFunction(allocator, text);
    defer allocator.free(z);
    return checksumSlice(usize, z);
}

fn runLevenshtein(allocator: Allocator, a: []const u8, b: []const u8) !u64 {
    return try levenshtein_mod.levenshteinDistance(allocator, a, b);
}

fn runRabinKarp(text: []const u8, pattern: []const u8) u64 {
    return checksumBool(rabin_karp_mod.rabinKarp(text, pattern));
}

fn runMinimumCoinChange(allocator: Allocator, coins: []const u64) !u64 {
    const out = try min_coin_mod.minimumCoinChange(allocator, coins, 987);
    defer allocator.free(out);
    return checksumSlice(u64, out);
}

fn runMinimumWaitingTime(allocator: Allocator, queries: []const u64) !u64 {
    return try min_waiting_mod.minimumWaitingTime(allocator, queries);
}

fn runFractionalKnapsack(allocator: Allocator, values: []const f64, weights: []const f64) !u64 {
    const result = try fractional_knapsack_mod.fractionalKnapsack(allocator, values, weights, 50.0);
    return @intFromFloat(result * 1_000_000.0);
}

fn runBestStock(prices: []const i64) u64 {
    return intToU64(i64, best_stock_mod.maxProfit(prices));
}

fn runMatrixMultiply(allocator: Allocator, a: []const i64, b: []const i64, dim: usize) !u64 {
    const c = try matrix_mul_mod.matMul(allocator, a, dim, dim, b, dim, dim);
    defer allocator.free(c);
    return checksumSlice(i64, c);
}

fn runMatrixTranspose(allocator: Allocator, mat: []const i64, rows: usize, cols: usize) !u64 {
    const out = try transpose_mod.transpose(allocator, mat, rows, cols);
    defer allocator.free(out);
    return checksumSlice(i64, out);
}

fn runRotateMatrix(allocator: Allocator, mat: []const i64, n: usize) !u64 {
    const copy = try allocator.dupe(i64, mat);
    defer allocator.free(copy);
    rotate_mod.rotate90(copy, n);
    return checksumSlice(i64, copy);
}

fn runSpiral(allocator: Allocator, mat: []const i64, rows: usize, cols: usize) !u64 {
    const out = try spiral_mod.spiralOrder(allocator, mat, rows, cols);
    defer allocator.free(out);
    return checksumSlice(i64, out);
}

fn runPascal(allocator: Allocator, rows: usize) !u64 {
    const tri = try pascal_mod.pascalTriangle(allocator, rows);
    defer {
        for (tri) |row| allocator.free(row);
        allocator.free(tri);
    }
    if (tri.len == 0) return 0;
    var sum: u64 = 0;
    for (tri[tri.len - 1]) |v| sum +%= v;
    return sum;
}

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    const filter_algorithm = std.process.getEnvVarOwned(allocator, "BENCH_ALGORITHM") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => null,
        else => return err,
    };
    defer if (filter_algorithm) |v| allocator.free(v);
    var filter_matched = false;

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("algorithm,category,iterations,total_ns,avg_ns,checksum\n", .{});

    const bubble_base = try generateIntData(allocator, 1_200);
    defer allocator.free(bubble_base);
    const n2_base = try generateIntData(allocator, 1_300);
    defer allocator.free(n2_base);
    const nlog_base = try generateIntData(allocator, 28_000);
    defer allocator.free(nlog_base);
    const non_neg_base = try generateNonNegativeData(allocator, 30_000);
    defer allocator.free(non_neg_base);
    const search_data = try generateSortedData(allocator, 40_000);
    defer allocator.free(search_data);
    const search_queries = try generateSearchQueries(allocator, 1_200, search_data.len);
    defer allocator.free(search_queries);
    const math_values = try generateU64Data(allocator, 40_000);
    defer allocator.free(math_values);
    const dp_array = try generateIntData64(allocator, 90_000);
    defer allocator.free(dp_array);

    const prices_raw = try generateIntData(allocator, 80_000);
    defer allocator.free(prices_raw);
    const prices = try allocator.alloc(i64, prices_raw.len);
    defer allocator.free(prices);
    for (prices_raw, 0..) |v, i| prices[i] = @as(i64, @intCast(@abs(v) % 1000));

    const waiting_seed = try generateU64Data(allocator, 50_000);
    defer allocator.free(waiting_seed);
    const waiting_queries = try allocator.alloc(u64, waiting_seed.len);
    defer allocator.free(waiting_queries);
    for (waiting_seed, 0..) |v, i| waiting_queries[i] = v % 100;

    const matrix_dim: usize = 70;
    const matrix_a = try generateMatrixData(allocator, matrix_dim * matrix_dim, 31, 7, 41, 20);
    defer allocator.free(matrix_a);
    const matrix_b = try generateMatrixData(allocator, matrix_dim * matrix_dim, 17, 11, 37, 18);
    defer allocator.free(matrix_b);

    const transpose_rows: usize = 180;
    const transpose_cols: usize = 220;
    const transpose_mat = try generateMatrixData(allocator, transpose_rows * transpose_cols, 23, 5, 53, 26);
    defer allocator.free(transpose_mat);

    const rotate_n: usize = 150;
    const rotate_mat = try generateMatrixData(allocator, rotate_n * rotate_n, 29, 3, 71, 35);
    defer allocator.free(rotate_mat);

    const spiral_rows: usize = 160;
    const spiral_cols: usize = 180;
    const spiral_mat = try generateMatrixData(allocator, spiral_rows * spiral_cols, 41, 9, 67, 33);
    defer allocator.free(spiral_mat);

    const text = try generateAsciiString(allocator, 130_000, 7, 3);
    defer allocator.free(text);
    const pattern = text[19_000..19_020];
    const s1 = try generateAsciiString(allocator, 420, 7, 3);
    defer allocator.free(s1);
    const s2 = try generateAsciiString(allocator, 450, 11, 5);
    defer allocator.free(s2);
    const s3 = try generateAsciiString(allocator, 100_000, 5, 1);
    defer allocator.free(s3);
    const s4 = try generateAsciiString(allocator, 100_000, 9, 4);
    defer allocator.free(s4);
    const palindrome_text = try repeatString(allocator, "amanaplanacanalpanama", 5_000);
    defer allocator.free(palindrome_text);
    const pangram_text = try repeatString(allocator, "The quick brown fox jumps over the lazy dog ", 8_000);
    defer allocator.free(pangram_text);
    const reverse_sentence = try repeatString(allocator, "I     Love          Python ", 12_000);
    defer allocator.free(reverse_sentence);
    const anagram_a = try repeatString(allocator, "This is a string ", 4_000);
    defer allocator.free(anagram_a);
    const anagram_b = try repeatString(allocator, "Is this a string ", 4_000);
    defer allocator.free(anagram_b);

    const graph_adj_owned = try buildGraphAdj(allocator, 6_000);
    defer freeGraphAdj(allocator, graph_adj_owned);
    const graph_adj = try allocator.alloc([]const usize, graph_adj_owned.len);
    defer allocator.free(graph_adj);
    for (graph_adj_owned, 0..) |row, i| graph_adj[i] = row;

    const weighted_graph_adj_owned = try buildWeightedGraphAdj(allocator, 2_200);
    defer freeWeightedGraphAdj(allocator, weighted_graph_adj_owned);
    const weighted_graph_adj = try allocator.alloc([]const dijkstra_mod.Edge, weighted_graph_adj_owned.len);
    defer allocator.free(weighted_graph_adj);
    for (weighted_graph_adj_owned, 0..) |row, i| weighted_graph_adj[i] = row;

    const knapsack_weights = try allocator.alloc(usize, 180);
    defer allocator.free(knapsack_weights);
    const knapsack_values = try allocator.alloc(usize, 180);
    defer allocator.free(knapsack_values);
    for (0..180) |i| {
        const idx: u64 = @intCast(i);
        knapsack_weights[i] = @intCast(((idx * 73) + 19) % 40 + 1);
        knapsack_values[i] = @intCast(((idx * 97) + 53) % 500 + 1);
    }

    const fib_inputs = try allocator.alloc(u32, 90);
    defer allocator.free(fib_inputs);
    for (0..90) |i| fib_inputs[i] = @intCast(i + 1);

    const fact_inputs = try allocator.alloc(u32, 20);
    defer allocator.free(fact_inputs);
    for (0..20) |i| fact_inputs[i] = @intCast(i + 1);

    const power_bases = try allocator.alloc(i64, 2_000);
    defer allocator.free(power_bases);
    const power_exps = try allocator.alloc(u32, 2_000);
    defer allocator.free(power_exps);
    for (0..2_000) |i| {
        power_bases[i] = @intCast((i % 19) + 2);
        power_exps[i] = @intCast((i % 10) + 5);
    }

    const stairs_inputs = try allocator.alloc(i32, 2_000);
    defer allocator.free(stairs_inputs);
    for (0..2_000) |i| stairs_inputs[i] = @intCast((i % 45) + 1);

    const unique_arr = try allocator.alloc(i32, 15);
    defer allocator.free(unique_arr);
    const unique_seed = [_]i32{ 11, 7, 11, 3, 3, 5, 5, 9, 9, 13, 13, 17, 17, 19, 19 };
    @memcpy(unique_arr, &unique_seed);

    const bin_samples_owned = try allocator.alloc([]u8, 199);
    defer {
        for (bin_samples_owned) |s| allocator.free(s);
        allocator.free(bin_samples_owned);
    }
    for (0..199) |i| {
        const n: u64 = @intCast((i + 1) * 12_345);
        bin_samples_owned[i] = try dec_to_bin_mod.decimalToBinary(allocator, n);
    }
    const bin_samples = try allocator.alloc([]const u8, 199);
    defer allocator.free(bin_samples);
    for (bin_samples_owned, 0..) |s, i| bin_samples[i] = s;

    const coin_set = [_]u32{ 1, 2, 3, 5, 7, 11, 13 };
    const coins_desc = [_]u64{ 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1 };
    const frac_values = [_]f64{ 60, 100, 120, 140, 30, 20, 80, 75 };
    const frac_weights = [_]f64{ 10, 20, 30, 40, 10, 5, 15, 25 };

    const missing_nums = try allocator.alloc(usize, 50_000);
    defer allocator.free(missing_nums);
    {
        var idx: usize = 0;
        for (0..50_001) |i| {
            if (i == 30_123) continue;
            missing_nums[idx] = i;
            idx += 1;
        }
    }
    const bin_text = "10101111100100101111000011101010";

    // Sorts (12)
    try benchRunMaybe(filter_algorithm, &filter_matched, runSortInPlace, stdout, "bubble_sort", "sorts", 2, .{ bubble_sort.bubbleSort, allocator, bubble_base });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSortInPlace, stdout, "insertion_sort", "sorts", 2, .{ insertion_sort.insertionSort, allocator, n2_base });
    try benchRunMaybe(filter_algorithm, &filter_matched, runMergeSort, stdout, "merge_sort", "sorts", 4, .{ allocator, nlog_base });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSortInPlace, stdout, "quick_sort", "sorts", 4, .{ quick_sort.quickSort, allocator, nlog_base });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSortInPlace, stdout, "heap_sort", "sorts", 4, .{ heap_sort.heapSort, allocator, nlog_base });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSortAllocated, stdout, "radix_sort", "sorts", 4, .{ radix_sort.radixSort, allocator, nlog_base });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSortAllocated, stdout, "bucket_sort", "sorts", 4, .{ bucket_sort.bucketSort, allocator, nlog_base });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSortInPlace, stdout, "selection_sort", "sorts", 2, .{ selection_sort.selectionSort, allocator, n2_base });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSortInPlace, stdout, "shell_sort", "sorts", 4, .{ shell_sort.shellSort, allocator, nlog_base });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSortAllocated, stdout, "counting_sort", "sorts", 4, .{ counting_sort.countingSort, allocator, non_neg_base });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSortInPlace, stdout, "cocktail_shaker_sort", "sorts", 2, .{ cocktail_shaker_sort.cocktailShakerSort, allocator, bubble_base });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSortInPlace, stdout, "gnome_sort", "sorts", 2, .{ gnome_sort.gnomeSort, allocator, bubble_base });

    // Searches (6)
    try benchRunMaybe(filter_algorithm, &filter_matched, runSearch, stdout, "linear_search", "searches", 3, .{ linear_search.linearSearch, search_data, search_queries });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSearch, stdout, "binary_search", "searches", 4, .{ binary_search.binarySearch, search_data, search_queries });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSearch, stdout, "exponential_search", "searches", 4, .{ exponential_search.exponentialSearch, search_data, search_queries });
    try benchRunMaybe(filter_algorithm, &filter_matched, runInterpolationSearch, stdout, "interpolation_search", "searches", 4, .{ search_data, search_queries });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSearch, stdout, "jump_search", "searches", 4, .{ jump_search.jumpSearch, search_data, search_queries });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSearch, stdout, "ternary_search", "searches", 4, .{ ternary_search.ternarySearch, search_data, search_queries });

    // Maths (8)
    try benchRunMaybe(filter_algorithm, &filter_matched, runMathGcd, stdout, "gcd", "maths", 6, .{math_values});
    try benchRunMaybe(filter_algorithm, &filter_matched, runMathLcm, stdout, "lcm", "maths", 6, .{math_values});
    try benchRunMaybe(filter_algorithm, &filter_matched, runFibonacciMany, stdout, "fibonacci", "maths", 200, .{fib_inputs});
    try benchRunMaybe(filter_algorithm, &filter_matched, runFactorialMany, stdout, "factorial", "maths", 200, .{fact_inputs});
    try benchRunMaybe(filter_algorithm, &filter_matched, runPowerMany, stdout, "power", "maths", 300, .{ power_bases, power_exps });
    try benchRunMaybe(filter_algorithm, &filter_matched, runPrimeCheck, stdout, "prime_check", "maths", 20, .{});
    try benchRunMaybe(filter_algorithm, &filter_matched, runSieve, stdout, "sieve_of_eratosthenes", "maths", 6, .{allocator});
    try benchRunMaybe(filter_algorithm, &filter_matched, runCollatzSteps, stdout, "collatz_sequence", "maths", 8, .{});

    // Dynamic Programming (7)
    try benchRunMaybe(filter_algorithm, &filter_matched, runClimbingStairsMany, stdout, "climbing_stairs", "dynamic_programming", 1000, .{stairs_inputs});
    try benchRunMaybe(filter_algorithm, &filter_matched, runFibonacciDp, stdout, "fibonacci_dp", "dynamic_programming", 400, .{allocator});
    try benchRunMaybe(filter_algorithm, &filter_matched, runCoinChange, stdout, "coin_change", "dynamic_programming", 80, .{ allocator, &coin_set });
    try benchRunMaybe(filter_algorithm, &filter_matched, runMaxSubarray, stdout, "max_subarray_sum", "dynamic_programming", 20, .{dp_array});
    try benchRunMaybe(filter_algorithm, &filter_matched, runLcs, stdout, "longest_common_subsequence", "dynamic_programming", 30, .{ allocator, s1, s2 });
    try benchRunMaybe(filter_algorithm, &filter_matched, runEditDistance, stdout, "edit_distance", "dynamic_programming", 30, .{ allocator, s1, s2 });
    try benchRunMaybe(filter_algorithm, &filter_matched, runKnapsack, stdout, "knapsack", "dynamic_programming", 60, .{ allocator, knapsack_weights, knapsack_values });

    // Graphs (3)
    try benchRunMaybe(filter_algorithm, &filter_matched, runBfs, stdout, "bfs", "graphs", 12, .{ allocator, graph_adj });
    try benchRunMaybe(filter_algorithm, &filter_matched, runDfs, stdout, "dfs", "graphs", 12, .{ allocator, graph_adj });
    try benchRunMaybe(filter_algorithm, &filter_matched, runDijkstra, stdout, "dijkstra", "graphs", 8, .{ allocator, weighted_graph_adj });

    // Bit Manipulation (6)
    try benchRunMaybe(filter_algorithm, &filter_matched, runIsPowerOfTwo, stdout, "is_power_of_two", "bit_manipulation", 120, .{});
    try benchRunMaybe(filter_algorithm, &filter_matched, runCountSetBits, stdout, "count_set_bits", "bit_manipulation", 60, .{});
    try benchRunMaybe(filter_algorithm, &filter_matched, runFindUniqueFromSlice, stdout, "find_unique_number", "bit_manipulation", 1000, .{unique_arr});
    try benchRunMaybe(filter_algorithm, &filter_matched, runReverseBits, stdout, "reverse_bits", "bit_manipulation", 80, .{});
    try benchRunMaybe(filter_algorithm, &filter_matched, missing_number_mod.missingNumber, stdout, "missing_number", "bit_manipulation", 220, .{missing_nums});
    try benchRunMaybe(filter_algorithm, &filter_matched, runPowerOfFour, stdout, "power_of_4", "bit_manipulation", 120, .{});

    // Conversions (4)
    try benchRunMaybe(filter_algorithm, &filter_matched, runDecimalToBinary, stdout, "decimal_to_binary", "conversions", 120, .{allocator});
    try benchRunMaybe(filter_algorithm, &filter_matched, runBinaryToDecimalMany, stdout, "binary_to_decimal", "conversions", 120, .{bin_samples});
    try benchRunMaybe(filter_algorithm, &filter_matched, runDecimalToHex, stdout, "decimal_to_hexadecimal", "conversions", 120, .{allocator});
    try benchRunMaybe(filter_algorithm, &filter_matched, runBinaryToHex, stdout, "binary_to_hexadecimal", "conversions", 120, .{ allocator, bin_text });

    // Greedy (4)
    try benchRunMaybe(filter_algorithm, &filter_matched, runBestStock, stdout, "best_time_to_buy_sell_stock", "greedy_methods", 20, .{prices});
    try benchRunMaybe(filter_algorithm, &filter_matched, runMinimumCoinChange, stdout, "minimum_coin_change", "greedy_methods", 200, .{ allocator, &coins_desc });
    try benchRunMaybe(filter_algorithm, &filter_matched, runMinimumWaitingTime, stdout, "minimum_waiting_time", "greedy_methods", 15, .{ allocator, waiting_queries });
    try benchRunMaybe(filter_algorithm, &filter_matched, runFractionalKnapsack, stdout, "fractional_knapsack", "greedy_methods", 600, .{ allocator, &frac_values, &frac_weights });

    // Matrix (5)
    try benchRunMaybe(filter_algorithm, &filter_matched, runMatrixMultiply, stdout, "matrix_multiply", "matrix", 10, .{ allocator, matrix_a, matrix_b, matrix_dim });
    try benchRunMaybe(filter_algorithm, &filter_matched, runMatrixTranspose, stdout, "matrix_transpose", "matrix", 25, .{ allocator, transpose_mat, transpose_rows, transpose_cols });
    try benchRunMaybe(filter_algorithm, &filter_matched, runRotateMatrix, stdout, "rotate_matrix", "matrix", 25, .{ allocator, rotate_mat, rotate_n });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSpiral, stdout, "spiral_print", "matrix", 20, .{ allocator, spiral_mat, spiral_rows, spiral_cols });
    try benchRunMaybe(filter_algorithm, &filter_matched, runPascal, stdout, "pascal_triangle", "matrix", 120, .{ allocator, @as(usize, 180) });

    // Strings (10)
    try benchRunMaybe(filter_algorithm, &filter_matched, runPalindrome, stdout, "palindrome", "strings", 120, .{palindrome_text});
    try benchRunMaybe(filter_algorithm, &filter_matched, runReverseWords, stdout, "reverse_words", "strings", 80, .{ allocator, reverse_sentence });
    try benchRunMaybe(filter_algorithm, &filter_matched, runAnagram, stdout, "anagram", "strings", 600, .{ anagram_a, anagram_b });
    try benchRunMaybe(filter_algorithm, &filter_matched, runHammingDistance, stdout, "hamming_distance", "strings", 120, .{ s3, s4 });
    try benchRunMaybe(filter_algorithm, &filter_matched, runNaiveSearch, stdout, "naive_string_search", "strings", 30, .{ allocator, text, pattern });
    try benchRunMaybe(filter_algorithm, &filter_matched, runKmpSearch, stdout, "knuth_morris_pratt", "strings", 80, .{ allocator, text, pattern });
    try benchRunMaybe(filter_algorithm, &filter_matched, runRabinKarp, stdout, "rabin_karp", "strings", 60, .{ text, pattern });
    try benchRunMaybe(filter_algorithm, &filter_matched, runZFunction, stdout, "z_function", "strings", 80, .{ allocator, text });
    try benchRunMaybe(filter_algorithm, &filter_matched, runLevenshtein, stdout, "levenshtein_distance", "strings", 120, .{ allocator, s1, s2 });
    try benchRunMaybe(filter_algorithm, &filter_matched, is_pangram_mod.isPangram, stdout, "is_pangram", "strings", 120, .{pangram_text});

    if (filter_algorithm != null and !filter_matched) {
        return error.UnknownBenchmarkAlgorithm;
    }
}
