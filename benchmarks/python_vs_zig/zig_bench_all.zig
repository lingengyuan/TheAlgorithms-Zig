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
const ext_gcd_mod = @import("../../maths/extended_euclidean.zig");
const mod_inverse_mod = @import("../../maths/modular_inverse.zig");
const totient_mod = @import("../../maths/eulers_totient.zig");
const crt_mod = @import("../../maths/chinese_remainder_theorem.zig");
const binom_mod = @import("../../maths/binomial_coefficient.zig");
const isqrt_mod = @import("../../maths/integer_square_root.zig");
const miller_rabin_mod = @import("../../maths/miller_rabin.zig");
const matrix_exp_mod = @import("../../maths/matrix_exponentiation.zig");
const stack_mod = @import("../../data_structures/stack.zig");
const queue_mod = @import("../../data_structures/queue.zig");
const singly_linked_list_mod = @import("../../data_structures/singly_linked_list.zig");
const doubly_linked_list_mod = @import("../../data_structures/doubly_linked_list.zig");
const binary_search_tree_mod = @import("../../data_structures/binary_search_tree.zig");
const min_heap_mod = @import("../../data_structures/min_heap.zig");
const trie_mod = @import("../../data_structures/trie.zig");
const disjoint_set_mod = @import("../../data_structures/disjoint_set.zig");
const avl_tree_mod = @import("../../data_structures/avl_tree.zig");
const max_heap_mod = @import("../../data_structures/max_heap.zig");
const priority_queue_mod = @import("../../data_structures/priority_queue.zig");
const hash_map_open_addressing_mod = @import("../../data_structures/hash_map_open_addressing.zig");
const segment_tree_mod = @import("../../data_structures/segment_tree.zig");
const fenwick_tree_mod = @import("../../data_structures/fenwick_tree.zig");
const red_black_tree_mod = @import("../../data_structures/red_black_tree.zig");
const lru_cache_mod = @import("../../data_structures/lru_cache.zig");
const deque_mod = @import("../../data_structures/deque.zig");

const climbing_stairs_mod = @import("../../dynamic_programming/climbing_stairs.zig");
const fibonacci_dp_mod = @import("../../dynamic_programming/fibonacci_dp.zig");
const coin_change_mod = @import("../../dynamic_programming/coin_change.zig");
const max_subarray_mod = @import("../../dynamic_programming/max_subarray_sum.zig");
const lis_mod = @import("../../dynamic_programming/longest_increasing_subsequence.zig");
const rod_cutting_mod = @import("../../dynamic_programming/rod_cutting.zig");
const matrix_chain_mod = @import("../../dynamic_programming/matrix_chain_multiplication.zig");
const palindrome_partition_mod = @import("../../dynamic_programming/palindrome_partitioning.zig");
const word_break_mod = @import("../../dynamic_programming/word_break.zig");
const catalan_mod = @import("../../dynamic_programming/catalan_numbers.zig");
const lcs_mod = @import("../../dynamic_programming/longest_common_subsequence.zig");
const edit_distance_mod = @import("../../dynamic_programming/edit_distance.zig");
const knapsack_mod = @import("../../dynamic_programming/knapsack.zig");
const subset_sum_mod = @import("../../dynamic_programming/subset_sum.zig");
const egg_drop_mod = @import("../../dynamic_programming/egg_drop_problem.zig");
const lps_mod = @import("../../dynamic_programming/longest_palindromic_subsequence.zig");
const max_product_mod = @import("../../dynamic_programming/max_product_subarray.zig");

const bfs_mod = @import("../../graphs/bfs.zig");
const dfs_mod = @import("../../graphs/dfs.zig");
const dijkstra_mod = @import("../../graphs/dijkstra.zig");
const a_star_mod = @import("../../graphs/a_star_search.zig");
const tarjan_mod = @import("../../graphs/tarjan_scc.zig");
const bridges_mod = @import("../../graphs/bridges.zig");
const euler_mod = @import("../../graphs/eulerian_path_circuit_undirected.zig");
const ford_mod = @import("../../graphs/ford_fulkerson.zig");
const bipartite_mod = @import("../../graphs/bipartite_check_bfs.zig");
const bellman_ford_mod = @import("../../graphs/bellman_ford.zig");
const topological_sort_mod = @import("../../graphs/topological_sort.zig");
const floyd_warshall_mod = @import("../../graphs/floyd_warshall.zig");
const detect_cycle_mod = @import("../../graphs/detect_cycle.zig");
const connected_components_mod = @import("../../graphs/connected_components.zig");
const kruskal_mod = @import("../../graphs/kruskal.zig");
const prim_mod = @import("../../graphs/prim.zig");

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
const roman_to_int_mod = @import("../../conversions/roman_to_integer.zig");
const int_to_roman_mod = @import("../../conversions/integer_to_roman.zig");
const temp_conv_mod = @import("../../conversions/temperature_conversion.zig");
const caesar_cipher_mod = @import("../../ciphers/caesar_cipher.zig");
const sha256_mod = @import("../../hashing/sha256.zig");

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
const aho_corasick_mod = @import("../../strings/aho_corasick.zig");
const suffix_array_mod = @import("../../strings/suffix_array.zig");
const run_length_encoding_mod = @import("../../strings/run_length_encoding.zig");

const best_stock_mod = @import("../../greedy_methods/best_time_to_buy_sell_stock.zig");
const min_coin_mod = @import("../../greedy_methods/minimum_coin_change.zig");
const min_waiting_mod = @import("../../greedy_methods/minimum_waiting_time.zig");
const fractional_knapsack_mod = @import("../../greedy_methods/fractional_knapsack.zig");
const activity_selection_mod = @import("../../greedy_methods/activity_selection.zig");
const huffman_coding_mod = @import("../../greedy_methods/huffman_coding.zig");
const job_seq_mod = @import("../../greedy_methods/job_sequencing_with_deadline.zig");

const matrix_mul_mod = @import("../../matrix/matrix_multiply.zig");
const transpose_mod = @import("../../matrix/matrix_transpose.zig");
const rotate_mod = @import("../../matrix/rotate_matrix.zig");
const spiral_mod = @import("../../matrix/spiral_print.zig");
const pascal_mod = @import("../../matrix/pascal_triangle.zig");
const permutations_mod = @import("../../backtracking/permutations.zig");
const combinations_mod = @import("../../backtracking/combinations.zig");
const subsets_mod = @import("../../backtracking/subsets.zig");
const generate_parentheses_mod = @import("../../backtracking/generate_parentheses.zig");
const n_queens_mod = @import("../../backtracking/n_queens.zig");
const sudoku_mod = @import("../../backtracking/sudoku_solver.zig");

const ExtPair = struct {
    a: i64,
    b: i64,
};

const ModInvPair = struct {
    a: i64,
    m: i64,
};

const CrtSystem = struct {
    remainders: [3]i64,
    moduli: [3]i64,
};

const BinomPair = struct {
    n: u64,
    k: u64,
};

const TemperatureCase = struct {
    value: f64,
    from: temp_conv_mod.Scale,
    to: temp_conv_mod.Scale,
};

const EggDropCase = struct {
    eggs: usize,
    floors: usize,
};

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

fn encodeBase26Word(buf: []u8, value: u64) void {
    var x = value;
    var i = buf.len;
    while (i > 0) {
        i -= 1;
        const d: u8 = @intCast(x % 26);
        buf[i] = @as(u8, 'a') + d;
        x /= 26;
    }
}

fn generateBase26Words(allocator: Allocator, count: usize, word_len: usize) ![][]u8 {
    const words = try allocator.alloc([]u8, count);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| allocator.free(words[i]);
        allocator.free(words);
    }
    for (0..count) |i| {
        words[i] = try allocator.alloc(u8, word_len);
        built += 1;
        encodeBase26Word(words[i], @intCast(i));
    }
    return words;
}

fn generateRleText(allocator: Allocator, run_count: usize) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    for (0..run_count) |i| {
        const ch: u8 = @as(u8, 'a') + @as(u8, @intCast(i % 26));
        const run_len: usize = ((i * 7 + 3) % 9) + 1;
        const old_len = out.items.len;
        try out.resize(allocator, old_len + run_len);
        @memset(out.items[old_len .. old_len + run_len], ch);
    }

    return try out.toOwnedSlice(allocator);
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

fn runExtendedEuclidean(pairs: []const ExtPair) u64 {
    var sum: u64 = 0;
    for (pairs) |p| {
        const r = ext_gcd_mod.extendedEuclidean(p.a, p.b);
        sum +%= intToU64(i64, r.gcd);
        sum +%= intToU64(i64, r.x);
        sum +%= intToU64(i64, r.y);
    }
    return sum;
}

fn runModularInverse(pairs: []const ModInvPair) !u64 {
    var sum: u64 = 0;
    for (pairs) |p| {
        sum +%= try mod_inverse_mod.modularInverse(p.a, p.m);
    }
    return sum;
}

fn runTotient(values: []const u64) u64 {
    var sum: u64 = 0;
    for (values) |v| {
        sum +%= totient_mod.eulersTotient(v);
    }
    return sum;
}

fn runCrt(systems: []const CrtSystem) !u64 {
    var sum: u64 = 0;
    for (systems) |s| {
        const x = try crt_mod.chineseRemainderTheorem(&s.remainders, &s.moduli);
        sum +%= intToU64(i64, x);
    }
    return sum;
}

fn runBinomial(pairs: []const BinomPair) !u64 {
    var sum: u64 = 0;
    for (pairs) |p| {
        sum +%= try binom_mod.binomialCoefficient(p.n, p.k);
    }
    return sum;
}

fn runIntegerSquareRoot(values: []const u64) u64 {
    var sum: u64 = 0;
    for (values) |v| {
        sum +%= isqrt_mod.integerSquareRoot(v);
    }
    return sum;
}

fn runMillerRabin(values: []const u64) u64 {
    var prime_count: u64 = 0;
    var signature: u64 = 0;
    for (values) |v| {
        if (miller_rabin_mod.isPrimeMillerRabin(v)) {
            prime_count +%= 1;
            signature +%= v;
        }
    }
    return prime_count +% (signature *% 3);
}

fn runMatrixExponentiation(allocator: Allocator, matrix: []const i64, dim: usize, exponent: u64) !u64 {
    const out = try matrix_exp_mod.matrixPower(allocator, matrix, dim, exponent);
    defer allocator.free(out);
    return checksumSlice(i64, out);
}

fn runStack(values: []const i64, allocator: Allocator) !u64 {
    var stack = stack_mod.Stack(i64).initWithLimit(allocator, values.len + 16);
    defer stack.deinit();

    var total: u64 = 0;
    for (values, 0..) |v, i| {
        try stack.push(v);
        if (i % 3 == 0) {
            if (stack.peek()) |top| total +%= intToU64(i64, top);
        }
        if (i % 5 == 0) {
            if (stack.pop()) |popped| total +%= intToU64(i64, popped) *% 3;
        }
    }

    while (stack.pop()) |v| {
        total +%= intToU64(i64, v);
    }
    return total +% (@as(u64, @intCast(values.len)) *% 11);
}

fn runQueue(values: []const i64, allocator: Allocator) !u64 {
    var queue = queue_mod.Queue(i64).init(allocator);
    defer queue.deinit();

    var total: u64 = 0;
    for (values, 0..) |v, i| {
        try queue.enqueue(v);
        if (i % 4 == 0) {
            if (queue.peek()) |front| total +%= intToU64(i64, front);
        }
        if (i % 6 == 0) {
            if (queue.dequeue()) |x| total +%= intToU64(i64, x) *% 3;
        }
        if (i % 25 == 0) {
            try queue.rotate(1);
        }
    }

    while (queue.dequeue()) |v| {
        total +%= intToU64(i64, v);
    }
    return total +% (@as(u64, @intCast(values.len)) *% 13);
}

fn runSinglyLinkedList(values: []const i64, allocator: Allocator) !u64 {
    var list = singly_linked_list_mod.SinglyLinkedList(i64).init(allocator);
    defer list.deinit();

    var total: u64 = 0;
    for (values, 0..) |v, i| {
        if (i % 2 == 0) {
            try list.insertTail(v);
        } else {
            try list.insertHead(v);
        }

        if (i % 7 == 0) {
            if (list.get(0)) |head| total +%= intToU64(i64, head);
        }
        if (i % 11 == 0 and !list.isEmpty()) {
            if (list.deleteTail()) |x| total +%= intToU64(i64, x) *% 5;
        }
    }

    const probe = @min(list.len, @as(usize, 1024));
    for (0..probe) |idx| {
        if (list.get(idx)) |v| total +%= intToU64(i64, v);
    }

    list.reverse();
    while (!list.isEmpty()) {
        if (list.deleteHead()) |v| total +%= intToU64(i64, v) *% 3;
    }
    return total +% (@as(u64, @intCast(values.len)) *% 17);
}

fn runDoublyLinkedList(values: []const i64, allocator: Allocator) !u64 {
    var list = doubly_linked_list_mod.DoublyLinkedList(i64).init(allocator);
    defer list.deinit();

    var total: u64 = 0;
    for (values, 0..) |v, i| {
        if (i % 3 == 0) {
            try list.insertHead(v);
        } else {
            try list.insertTail(v);
        }

        if (i % 9 == 0 and !list.isEmpty()) {
            if (list.deleteHead()) |x| total +%= intToU64(i64, x) *% 3;
        }
        if (i % 13 == 0 and !list.isEmpty()) {
            if (list.deleteTail()) |x| total +%= intToU64(i64, x) *% 5;
        }
    }

    const probe = @min(list.len, @as(usize, 1024));
    for (0..probe) |idx| {
        if (list.get(idx)) |v| total +%= intToU64(i64, v);
    }

    list.reverse();
    while (!list.isEmpty()) {
        if (list.deleteTail()) |v| total +%= intToU64(i64, v);
    }
    return total +% (@as(u64, @intCast(values.len)) *% 19);
}

fn runBinarySearchTree(values: []const i64, queries: []const i64, removals: []const i64, allocator: Allocator) !u64 {
    var tree = binary_search_tree_mod.BinarySearchTree(i64).init(allocator);
    defer tree.deinit();

    for (values) |v| try tree.insert(v);

    var hits: u64 = 0;
    for (queries) |q| {
        if (tree.search(q)) hits +%= 1;
    }

    const ordered = try tree.inorder(allocator);
    defer allocator.free(ordered);
    const inorder_checksum = checksumSlice(i64, ordered);

    const min_value = tree.getMin() orelse 0;
    const max_value = tree.getMax() orelse 0;

    var removed: u64 = 0;
    for (removals) |v| {
        if (tree.remove(v)) removed +%= 1;
    }

    return inorder_checksum +
        (hits *% 31) +
        (removed *% 17) +
        (intToU64(i64, min_value) *% 3) +
        (intToU64(i64, max_value) *% 5) +
        (@as(u64, @intCast(tree.len)) *% 7);
}

fn runMinHeap(values: []const i64, push_count: usize, allocator: Allocator) !u64 {
    var heap = try min_heap_mod.MinHeap(i64).fromSlice(allocator, values);
    defer heap.deinit();

    var total: u64 = 0;
    for (0..push_count) |i| {
        if (heap.extractMin()) |x| total +%= intToU64(i64, x);

        const idx: i64 = @intCast(i);
        const new_value = @mod(idx * 97 + 31, 100_000) - 50_000;
        try heap.insert(new_value);

        if (i % 4 == 0) {
            if (heap.peek()) |p| total +%= intToU64(i64, p) *% 3;
        }
    }

    while (heap.extractMin()) |x| {
        total +%= intToU64(i64, x);
    }

    return total +
        (@as(u64, @intCast(values.len)) *% 23) +
        (@as(u64, @intCast(push_count)) *% 5);
}

fn runTrie(allocator: Allocator, words: []const []const u8) !u64 {
    var trie = try trie_mod.Trie.init(allocator);
    defer trie.deinit();

    for (words) |word| try trie.insert(word);

    var present: u64 = 0;
    for (words) |word| {
        if (trie.contains(word)) present +%= 1;
    }

    var prefix_hits: u64 = 0;
    for (words) |word| {
        if (trie.startsWith(word[0..3])) prefix_hits +%= 1;
    }

    var removed: u64 = 0;
    var i: usize = 0;
    while (i < words.len) : (i += 5) {
        if (trie.remove(words[i])) removed +%= 1;
    }

    var remain: u64 = 0;
    for (words) |word| {
        if (trie.contains(word)) remain +%= 1;
    }

    return present +% (prefix_hits *% 3) +% (removed *% 5) +% (remain *% 7);
}

fn runDisjointSet(allocator: Allocator, n: usize) !u64 {
    var ds = try disjoint_set_mod.DisjointSet.init(allocator, n);
    defer ds.deinit();

    var i: usize = 0;
    while (i + 1 < n) : (i += 2) {
        _ = try ds.unionSets(i, i + 1);
    }
    i = 0;
    while (i + 3 < n) : (i += 3) {
        _ = try ds.unionSets(i, i + 3);
    }

    var connected_hits: u64 = 0;
    i = 0;
    while (i < 20_000) : (i += 1) {
        const a = (i * 97 + 31) % n;
        const b = (i * 53 + 17) % n;
        if (try ds.connected(a, b)) connected_hits +%= 1;
    }

    var root_sum: u64 = 0;
    i = 0;
    while (i < 10_000) : (i += 1) {
        const idx = (i * 193 + 7) % n;
        root_sum +%= @intCast(try ds.find(idx));
    }

    return connected_hits +% root_sum +% (@as(u64, @intCast(ds.componentCount())) *% 11);
}

fn runAvlTree(allocator: Allocator, values: []const i64, queries: []const i64) !u64 {
    var tree = avl_tree_mod.AvlTree.init(allocator);
    defer tree.deinit();

    for (values) |v| _ = try tree.insert(v);

    var hits_before: u64 = 0;
    for (queries) |q| {
        if (tree.contains(q)) hits_before +%= 1;
    }

    var removed: u64 = 0;
    var i: usize = 0;
    while (i < values.len) : (i += 4) {
        if (tree.remove(values[i])) removed +%= 1;
    }

    var hits_after: u64 = 0;
    for (queries) |q| {
        if (tree.contains(q)) hits_after +%= 1;
    }

    const ordered = try tree.inorder(allocator);
    defer allocator.free(ordered);
    const inorder_checksum = checksumSlice(i64, ordered);

    return hits_before +% (hits_after *% 3) +% (removed *% 5) +% (inorder_checksum *% 7);
}

fn runMaxHeap(allocator: Allocator, values: []const i64) !u64 {
    var heap = try max_heap_mod.MaxHeap(i64).fromSlice(allocator, values);
    defer heap.deinit();

    const out = try allocator.alloc(i64, values.len);
    defer allocator.free(out);

    var idx: usize = 0;
    while (heap.extractMax()) |v| {
        out[idx] = v;
        idx += 1;
    }
    return checksumSlice(i64, out[0..idx]);
}

fn runPriorityQueue(allocator: Allocator, n: usize) !u64 {
    var pq = priority_queue_mod.PriorityQueue.init(allocator);
    defer pq.deinit();

    for (0..n) |i| {
        const value: i64 = @intCast(i);
        const idx: u64 = @intCast(i);
        const priority: i64 = @intCast(((idx * 97) + 31) % 1000);
        try pq.enqueue(value, priority);
    }

    const out = try allocator.alloc(i64, n);
    defer allocator.free(out);
    var idx: usize = 0;
    while (pq.dequeue()) |item| {
        out[idx] = item.value;
        idx += 1;
    }
    return checksumSlice(i64, out[0..idx]);
}

fn runHashMapOpenAddressing(allocator: Allocator, n: usize) !u64 {
    var map = try hash_map_open_addressing_mod.OpenAddressHashMap.init(allocator, n);
    defer map.deinit();

    const keys = try allocator.alloc(i64, n);
    defer allocator.free(keys);

    for (0..n) |i| {
        const idx: i64 = @intCast(i);
        const key = idx * 2 - 80_000;
        const i_u64: u64 = @intCast(i);
        const value: i64 = @intCast(((i_u64 * 131) + 17) % 1_000_003);
        try map.put(key, value - 500_000);
        keys[i] = key;
    }

    var updated: u64 = 0;
    var i: usize = 0;
    while (i < n) : (i += 3) {
        const key = keys[i];
        if (map.get(key)) |value| {
            try map.put(key, value + 11);
            updated +%= 1;
        }
    }

    var removed: u64 = 0;
    i = 0;
    while (i < n) : (i += 5) {
        if (map.remove(keys[i])) removed +%= 1;
    }

    var hits: u64 = 0;
    var lookup_sum: u64 = 0;
    for (keys) |key| {
        if (map.get(key)) |value| {
            hits +%= 1;
            lookup_sum +%= intToU64(i64, value);
        }
    }

    return lookup_sum +% (hits *% 3) +% (removed *% 5) +% (updated *% 7) +% (@as(u64, @intCast(map.count())) *% 11);
}

fn runSegmentTree(allocator: Allocator, values: []const i64) !u64 {
    if (values.len == 0) return 0;

    var st = try segment_tree_mod.SegmentTree.init(allocator, values);
    defer st.deinit();

    const n = values.len;
    var checksum: u64 = 0;

    var i: usize = 0;
    while (i < n) : (i += 97) {
        const left = i;
        const right = @min(n - 1, left + 63);
        const q = try st.query(left, right);
        checksum +%= intToU64(i64, q);
    }

    i = 0;
    while (i < n) : (i += 53) {
        const index = (i * 37 + 11) % n;
        const i_u64: u64 = @intCast(i);
        const value: i64 = @intCast(((i_u64 * 131) + 19) % 1_000_003);
        try st.update(index, value - 500_000);
    }

    i = 0;
    while (i < n) : (i += 89) {
        const left = (i * 17 + 5) % n;
        const span = ((i * 29 + 7) % 64) + 1;
        const right = @min(n - 1, left + span - 1);
        const q = try st.query(left, right);
        checksum +%= intToU64(i64, q);
    }

    return checksum;
}

fn runFenwickTree(allocator: Allocator, values: []const i64) !u64 {
    if (values.len == 0) return 0;

    var fw = try fenwick_tree_mod.FenwickTree.fromSlice(allocator, values);
    defer fw.deinit();

    const n = values.len;
    var checksum: u64 = 0;

    var i: usize = 0;
    while (i < n) : (i += 97) {
        const right = (i * 41 + 23) % (n + 1);
        const sum = try fw.prefixSum(right);
        checksum +%= intToU64(i64, sum);
    }

    i = 0;
    while (i < n) : (i += 53) {
        const index = (i * 31 + 9) % n;
        const delta: i64 = @intCast(((i * 17 + 5) % 201));
        try fw.add(index, delta - 100);
    }

    i = 0;
    while (i < n) : (i += 71) {
        const index = (i * 37 + 11) % n;
        const i_u64: u64 = @intCast(i);
        const value: i64 = @intCast(((i_u64 * 101) + 3) % 1_000_003);
        try fw.set(index, value - 500_000);
    }

    i = 0;
    while (i < n) : (i += 83) {
        const left = (i * 13 + 7) % n;
        const span = ((i * 19 + 5) % 128) + 1;
        const right = @min(n, left + span);
        const sum = try fw.rangeSum(left, right);
        checksum +%= intToU64(i64, sum);
    }

    i = 0;
    while (i < n) : (i += 101) {
        const index = (i * 43 + 29) % n;
        const value = try fw.get(index);
        checksum +%= intToU64(i64, value);
    }

    return checksum;
}

fn runRedBlackTree(allocator: Allocator, values: []const i64, queries: []const i64) !u64 {
    var tree = red_black_tree_mod.RedBlackTree.init(allocator);
    defer tree.deinit();

    var inserted: u64 = 0;
    for (values) |v| {
        if (try tree.insert(v)) inserted +%= 1;
    }

    var hits: u64 = 0;
    for (queries) |q| {
        if (tree.contains(q)) hits +%= 1;
    }

    const ordered = try tree.inorder(allocator);
    defer allocator.free(ordered);
    const inorder_checksum = checksumSlice(i64, ordered);
    const color_props_ok: u64 = if (tree.checkColorProperties()) 1 else 0;

    return inserted +% (hits *% 3) +% (inorder_checksum *% 5) +% (color_props_ok *% 7) +% (@as(u64, @intCast(tree.len())) *% 11);
}

fn runLruCache(allocator: Allocator, capacity: usize, ops: usize) !u64 {
    if (capacity == 0) return 0;

    var cache = try lru_cache_mod.LruCache.init(allocator, capacity);
    defer cache.deinit();

    for (0..capacity) |i| {
        const key: i64 = @intCast(i);
        const value: i64 = @intCast(@as(i64, @intCast(i)) * 2 - 3);
        try cache.put(key, value);
    }

    const key_space = capacity * 3;
    for (0..ops) |i| {
        const key: i64 = @intCast((i * 97 + 31) % key_space);
        if (i % 5 == 0) {
            const i_u64: u64 = @intCast(i);
            const value: i64 = @intCast(((i_u64 * 131) + 17) % 1_000_003);
            try cache.put(key, value - 500_000);
        } else {
            _ = cache.get(key);
        }
    }

    var probe_sum: u64 = 0;
    for (0..2_000) |i| {
        const key: i64 = @intCast((i * 53 + 7) % key_space);
        if (cache.get(key)) |value| {
            probe_sum +%= intToU64(i64, value);
        }
    }

    const info = cache.cacheInfo();
    return probe_sum +% (@as(u64, @intCast(info.hits)) *% 3) +% (@as(u64, @intCast(info.misses)) *% 5) +% (@as(u64, @intCast(info.size)) *% 7);
}

fn runDeque(allocator: Allocator, ops: usize) !u64 {
    var dq = try deque_mod.Deque.init(allocator);
    defer dq.deinit();

    var checksum: u64 = 0;
    for (0..ops) |i| {
        if (i % 4 == 0) {
            const value: i64 = @intCast(@as(i64, @intCast(i)) - 25_000);
            try dq.pushFront(value);
        } else {
            const value: i64 = @intCast(@as(i64, @intCast(i)) * 3 - 12_345);
            try dq.pushBack(value);
        }

        if (i % 7 == 0) {
            if (dq.popFront()) |value| checksum +%= intToU64(i64, value);
        }
        if (i % 11 == 0) {
            if (dq.popBack()) |value| checksum +%= intToU64(i64, value);
        }
    }

    const front = dq.peekFront() orelse 0;
    const back = dq.peekBack() orelse 0;
    return checksum +% (intToU64(i64, front) *% 3) +% (intToU64(i64, back) *% 5) +% (@as(u64, @intCast(dq.len())) *% 7);
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

fn runLis(allocator: Allocator, arr: []const i64) !u64 {
    return @intCast(try lis_mod.longestIncreasingSubsequenceLength(allocator, arr));
}

fn runRodCutting(allocator: Allocator, prices: []const i64, length: usize) !u64 {
    return intToU64(i64, try rod_cutting_mod.rodCutting(allocator, prices, length));
}

fn runMatrixChainMultiplication(allocator: Allocator, dims: []const usize) !u64 {
    return @intCast(try matrix_chain_mod.matrixChainMultiplication(allocator, dims));
}

fn runPalindromePartitioning(allocator: Allocator, text: []const u8) !u64 {
    return @intCast(try palindrome_partition_mod.minPalindromeCuts(allocator, text));
}

fn runWordBreak(allocator: Allocator, text: []const u8, dict: []const []const u8) !u64 {
    return checksumBool(try word_break_mod.wordBreak(allocator, text, dict));
}

fn runCatalanMany(allocator: Allocator, inputs: []const u32) !u64 {
    var sum: u64 = 0;
    for (inputs) |n| {
        sum +%= try catalan_mod.catalanNumber(allocator, n);
    }
    return sum;
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

fn runSubsetSum(allocator: Allocator, numbers: []const i64, targets: []const i64) !u64 {
    var possible: u64 = 0;
    var signature: u64 = 0;
    for (targets) |target| {
        if (try subset_sum_mod.isSubsetSum(allocator, numbers, target)) {
            possible +%= 1;
            signature +%= intToU64(i64, target);
        }
    }
    return possible +% (signature *% 3);
}

fn runEggDropProblem(allocator: Allocator, cases: []const EggDropCase) !u64 {
    var checksum: u64 = 0;
    for (cases) |c| {
        const trials = try egg_drop_mod.eggDropMinTrials(allocator, c.eggs, c.floors);
        checksum +%= @as(u64, @intCast(trials)) +
            (@as(u64, @intCast(c.floors)) *% 3) +
            (@as(u64, @intCast(c.eggs)) *% 5);
    }
    return checksum;
}

fn runLongestPalindromicSubsequence(allocator: Allocator, text: []const u8) !u64 {
    return @intCast(try lps_mod.longestPalindromicSubsequenceLength(allocator, text));
}

fn runMaxProductSubarray(values: []const i64) !u64 {
    const best = try max_product_mod.maxProductSubarray(values);
    return intToU64(i64, best);
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

fn buildWeightedGraphAdj(comptime EdgeType: type, allocator: Allocator, n: usize) ![][]EdgeType {
    const adj = try allocator.alloc([]EdgeType, n);
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

        adj[i] = try allocator.alloc(EdgeType, count);
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

fn freeWeightedGraphAdj(comptime EdgeType: type, allocator: Allocator, adj: [][]EdgeType) void {
    for (adj) |row| allocator.free(row);
    allocator.free(adj);
}

fn buildTarjanAdj(allocator: Allocator, n: usize, block_size: usize) ![][]usize {
    const adj = try allocator.alloc([]usize, n);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| allocator.free(adj[i]);
        allocator.free(adj);
    }

    for (0..n) |i| {
        const block = if (block_size == 0) 1 else block_size;
        const block_start = (i / block) * block;
        const block_len = @min(block, n - block_start);

        var count: usize = 1; // cycle edge inside the block
        if (i % block == 0 and i + block < n) {
            count += 1; // one-way edge to next block leader
        }

        adj[i] = try allocator.alloc(usize, count);
        built += 1;

        const offset = i - block_start;
        const next_in_block = block_start + ((offset + 1) % block_len);

        adj[i][0] = next_in_block;
        if (count == 2) {
            adj[i][1] = i + block;
        }
    }
    return adj;
}

fn buildBridgesAdj(allocator: Allocator, n: usize) ![][]usize {
    const adj = try allocator.alloc([]usize, n);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| allocator.free(adj[i]);
        allocator.free(adj);
    }

    for (0..n) |i| {
        var count: usize = 0;
        const block = i / 3;
        const pos = i % 3;
        const block_start = block * 3;
        const a = block_start;
        const b = block_start + 1;
        const c = block_start + 2;
        const has_full_block = c < n;
        const next_block_a = block_start + 3;

        if (!has_full_block) {
            adj[i] = try allocator.alloc(usize, 0);
            built += 1;
            continue;
        }

        if (pos == 0) {
            count = 2; // a <-> b, a <-> c
        } else if (pos == 1) {
            count = 2; // b <-> a, b <-> c
        } else {
            count = 2; // c <-> a, c <-> b
            if (next_block_a < n) count += 1; // bridge c -> next block a
        }
        if (pos == 0 and block > 0) {
            count += 1; // back bridge from previous block c -> current a
        }

        adj[i] = try allocator.alloc(usize, count);
        built += 1;

        var idx: usize = 0;
        if (pos == 0) {
            adj[i][idx] = b;
            idx += 1;
            adj[i][idx] = c;
            idx += 1;
            if (block > 0) {
                adj[i][idx] = block_start - 1;
                idx += 1;
            }
        } else if (pos == 1) {
            adj[i][idx] = a;
            idx += 1;
            adj[i][idx] = c;
            idx += 1;
        } else {
            adj[i][idx] = a;
            idx += 1;
            adj[i][idx] = b;
            idx += 1;
            if (next_block_a < n) {
                adj[i][idx] = next_block_a;
                idx += 1;
            }
        }
    }
    return adj;
}

fn buildEulerChainAdj(allocator: Allocator, n: usize) ![][]usize {
    const adj = try allocator.alloc([]usize, n);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| allocator.free(adj[i]);
        allocator.free(adj);
    }

    for (0..n) |i| {
        var count: usize = 0;
        if (i > 0) count += 1;
        if (i + 1 < n) count += 1;

        adj[i] = try allocator.alloc(usize, count);
        built += 1;

        var idx: usize = 0;
        if (i > 0) {
            adj[i][idx] = i - 1;
            idx += 1;
        }
        if (i + 1 < n) {
            adj[i][idx] = i + 1;
            idx += 1;
        }
    }
    return adj;
}

fn buildFlowCapacityFlat(allocator: Allocator, n: usize) ![]i64 {
    const mat = try allocator.alloc(i64, n * n);
    @memset(mat, 0);

    for (0..n) |i| {
        const i_u64: u64 = @intCast(i);
        if (i + 1 < n) {
            mat[i * n + (i + 1)] = @intCast(((i_u64 * 17) + 3) % 23 + 1);
        }
        if (i + 2 < n) {
            mat[i * n + (i + 2)] = @intCast(((i_u64 * 31) + 7) % 19 + 1);
        }
        if (i % 5 == 0 and i + 7 < n) {
            mat[i * n + (i + 7)] = @intCast(((i_u64 * 13) + 11) % 29 + 1);
        }
    }
    return mat;
}

fn buildBipartiteAdj(allocator: Allocator, n: usize) ![][]usize {
    const adj = try allocator.alloc([]usize, n);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| allocator.free(adj[i]);
        allocator.free(adj);
    }

    for (0..n) |i| {
        var count: usize = 0;

        if (i % 2 == 0) {
            if (i + 1 < n) count += 1;
            if (i + 3 < n) count += 1;
        } else {
            if (i >= 1) count += 1;
            if (i >= 3) count += 1;
        }

        adj[i] = try allocator.alloc(usize, count);
        built += 1;

        var idx: usize = 0;
        if (i % 2 == 0) {
            if (i + 1 < n) {
                adj[i][idx] = i + 1;
                idx += 1;
            }
            if (i + 3 < n) {
                adj[i][idx] = i + 3;
                idx += 1;
            }
        } else {
            adj[i][idx] = i - 1;
            idx += 1;
            if (i >= 3) {
                adj[i][idx] = i - 3;
                idx += 1;
            }
        }
    }
    return adj;
}

fn buildBellmanEdges(allocator: Allocator, n: usize) ![]bellman_ford_mod.Edge {
    var edge_count: usize = 0;
    for (0..n) |i| {
        if (i + 1 < n) edge_count += 1;
        if (i + 2 < n) edge_count += 1;
        if (i % 3 == 0 and i + 17 < n) edge_count += 1;
    }

    const edges = try allocator.alloc(bellman_ford_mod.Edge, edge_count);
    var idx: usize = 0;
    for (0..n) |i| {
        const i_u64: u64 = @intCast(i);
        if (i + 1 < n) {
            edges[idx] = .{
                .from = i,
                .to = i + 1,
                .weight = @intCast(((i_u64 * 17) + 3) % 23 + 1),
            };
            idx += 1;
        }
        if (i + 2 < n) {
            edges[idx] = .{
                .from = i,
                .to = i + 2,
                .weight = @intCast(((i_u64 * 31) + 7) % 29 + 1),
            };
            idx += 1;
        }
        if (i % 3 == 0 and i + 17 < n) {
            edges[idx] = .{
                .from = i,
                .to = i + 17,
                .weight = @intCast(((i_u64 * 13) + 11) % 41 + 1),
            };
            idx += 1;
        }
    }
    return edges;
}

fn buildFloydMatrix(allocator: Allocator, n: usize, inf: i64) ![]i64 {
    const mat = try allocator.alloc(i64, n * n);
    @memset(mat, inf);

    for (0..n) |i| {
        mat[i * n + i] = 0;
        const i_u64: u64 = @intCast(i);
        if (i + 1 < n) {
            mat[i * n + (i + 1)] = @intCast(((i_u64 * 17) + 3) % 23 + 1);
        }
        if (i + 2 < n) {
            mat[i * n + (i + 2)] = @intCast(((i_u64 * 31) + 7) % 29 + 1);
        }
        if (i % 3 == 0 and i + 17 < n) {
            mat[i * n + (i + 17)] = @intCast(((i_u64 * 13) + 11) % 41 + 1);
        }
    }
    return mat;
}

fn buildCycleGraphAdj(allocator: Allocator, n: usize) ![][]usize {
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
        if (i + 1 == n and n > 0) count += 1; // last node links back to 0

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
        if (i + 1 == n and n > 0) {
            adj[i][idx] = 0;
            idx += 1;
        }
    }
    return adj;
}

fn buildComponentAdj(allocator: Allocator, n: usize) ![][]usize {
    const adj = try allocator.alloc([]usize, n);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| allocator.free(adj[i]);
        allocator.free(adj);
    }

    const split = if (n >= 2) (n / 2) - 1 else 0;
    for (0..n) |i| {
        var count: usize = 0;
        if (i > 0 and i - 1 != split) count += 1;
        if (i + 1 < n and i != split) count += 1;

        adj[i] = try allocator.alloc(usize, count);
        built += 1;

        var idx: usize = 0;
        if (i > 0 and i - 1 != split) {
            adj[i][idx] = i - 1;
            idx += 1;
        }
        if (i + 1 < n and i != split) {
            adj[i][idx] = i + 1;
            idx += 1;
        }
    }
    return adj;
}

fn buildMstEdgesKruskal(allocator: Allocator, n: usize) ![]kruskal_mod.Edge {
    var edge_count: usize = 0;
    for (0..n) |i| {
        if (i + 1 < n) edge_count += 1;
        if (i + 2 < n) edge_count += 1;
        if (i % 5 == 0 and i + 11 < n) edge_count += 1;
    }

    const edges = try allocator.alloc(kruskal_mod.Edge, edge_count);
    var idx: usize = 0;
    for (0..n) |i| {
        const i_u64: u64 = @intCast(i);
        if (i + 1 < n) {
            edges[idx] = .{
                .u = i,
                .v = i + 1,
                .weight = @intCast(((i_u64 * 19) + 5) % 37 + 1),
            };
            idx += 1;
        }
        if (i + 2 < n) {
            edges[idx] = .{
                .u = i,
                .v = i + 2,
                .weight = @intCast(((i_u64 * 23) + 7) % 43 + 1),
            };
            idx += 1;
        }
        if (i % 5 == 0 and i + 11 < n) {
            edges[idx] = .{
                .u = i,
                .v = i + 11,
                .weight = @intCast(((i_u64 * 29) + 13) % 53 + 1),
            };
            idx += 1;
        }
    }
    return edges;
}

fn buildMstAdjPrim(allocator: Allocator, n: usize) ![][]prim_mod.Edge {
    const adj = try allocator.alloc([]prim_mod.Edge, n);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| allocator.free(adj[i]);
        allocator.free(adj);
    }

    for (0..n) |i| {
        var count: usize = 0;
        if (i > 0) count += 1;
        if (i + 1 < n) count += 1;
        if (i >= 2) count += 1;
        if (i + 2 < n) count += 1;
        if (i >= 11 and (i - 11) % 5 == 0) count += 1;
        if (i % 5 == 0 and i + 11 < n) count += 1;

        adj[i] = try allocator.alloc(prim_mod.Edge, count);
        built += 1;

        var idx: usize = 0;
        const i_u64: u64 = @intCast(i);

        if (i > 0) {
            const j = i - 1;
            const j_u64: u64 = @intCast(j);
            adj[i][idx] = .{ .to = j, .weight = @intCast(((j_u64 * 19) + 5) % 37 + 1) };
            idx += 1;
        }
        if (i + 1 < n) {
            adj[i][idx] = .{ .to = i + 1, .weight = @intCast(((i_u64 * 19) + 5) % 37 + 1) };
            idx += 1;
        }
        if (i >= 2) {
            const j = i - 2;
            const j_u64: u64 = @intCast(j);
            adj[i][idx] = .{ .to = j, .weight = @intCast(((j_u64 * 23) + 7) % 43 + 1) };
            idx += 1;
        }
        if (i + 2 < n) {
            adj[i][idx] = .{ .to = i + 2, .weight = @intCast(((i_u64 * 23) + 7) % 43 + 1) };
            idx += 1;
        }
        if (i >= 11 and (i - 11) % 5 == 0) {
            const j = i - 11;
            const j_u64: u64 = @intCast(j);
            adj[i][idx] = .{ .to = j, .weight = @intCast(((j_u64 * 29) + 13) % 53 + 1) };
            idx += 1;
        }
        if (i % 5 == 0 and i + 11 < n) {
            adj[i][idx] = .{ .to = i + 11, .weight = @intCast(((i_u64 * 29) + 13) % 53 + 1) };
            idx += 1;
        }
    }
    return adj;
}

fn freePrimAdj(allocator: Allocator, adj: [][]prim_mod.Edge) void {
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

fn runAStar(allocator: Allocator, adj: []const []const a_star_mod.Edge, heuristics: []const u64, goal: usize) !u64 {
    const result = try a_star_mod.aStarSearch(allocator, adj, heuristics, 0, goal);
    defer allocator.free(result.path);
    return result.cost +% @as(u64, @intCast(result.path.len));
}

fn runTarjanScc(allocator: Allocator, adj: []const []const usize) !u64 {
    var result = try tarjan_mod.tarjanScc(allocator, adj);
    defer result.deinit(allocator);

    var checksum: u64 = @intCast(result.components.len);
    for (result.components) |component| {
        checksum +%= @as(u64, @intCast(component.len));
    }
    return checksum;
}

fn runBridges(allocator: Allocator, adj: []const []const usize) !u64 {
    const bridges = try bridges_mod.findBridges(allocator, adj);
    defer allocator.free(bridges);

    var checksum: u64 = @intCast(bridges.len);
    for (bridges) |bridge| {
        const u64_u: u64 = @intCast(bridge.u);
        const u64_v: u64 = @intCast(bridge.v);
        checksum +%= u64_u *% 1_315_423_911 +% u64_v;
    }
    return checksum;
}

fn runEulerianPathUndirected(allocator: Allocator, adj: []const []const usize) !u64 {
    const result = try euler_mod.findEulerianPathOrCircuit(allocator, adj);
    defer allocator.free(result.path);

    const kind_value: u64 = switch (result.kind) {
        .circuit => 1,
        .path => 2,
    };

    if (result.path.len == 0) return kind_value;
    return kind_value +% checksumSlice(usize, result.path);
}

fn runFordFulkerson(allocator: Allocator, capacity: []const []const i64) !u64 {
    const source: usize = 0;
    const sink: usize = if (capacity.len == 0) 0 else capacity.len - 1;
    const flow = try ford_mod.fordFulkersonMaxFlow(allocator, capacity, source, sink);
    return intToU64(i64, flow);
}

fn runBipartiteCheck(allocator: Allocator, adj: []const []const usize) !u64 {
    return checksumBool(try bipartite_mod.isBipartiteBfs(allocator, adj));
}

fn runBellmanFord(allocator: Allocator, vertex_count: usize, edges: []const bellman_ford_mod.Edge) !u64 {
    const dist = try bellman_ford_mod.bellmanFord(allocator, vertex_count, edges, 0);
    defer allocator.free(dist);
    return checksumSlice(i64, dist);
}

fn runTopologicalSort(allocator: Allocator, adj: []const []const usize) !u64 {
    const order = try topological_sort_mod.topologicalSort(allocator, adj);
    defer allocator.free(order);
    return checksumSlice(usize, order);
}

fn runFloydWarshall(allocator: Allocator, matrix: []const i64, n: usize, inf: i64) !u64 {
    const out = try floyd_warshall_mod.floydWarshall(allocator, matrix, n, inf);
    defer allocator.free(out);
    return checksumSlice(i64, out);
}

fn runDetectCycle(allocator: Allocator, adj: []const []const usize) !u64 {
    return checksumBool(try detect_cycle_mod.hasCycle(allocator, adj));
}

fn runConnectedComponents(allocator: Allocator, adj: []const []const usize) !u64 {
    return @intCast(try connected_components_mod.countConnectedComponents(allocator, adj));
}

fn runKruskal(allocator: Allocator, vertex_count: usize, edges: []const kruskal_mod.Edge) !u64 {
    const weight = try kruskal_mod.kruskalMstWeight(allocator, vertex_count, edges);
    return intToU64(i64, weight);
}

fn runPrim(allocator: Allocator, adj: []const []const prim_mod.Edge) !u64 {
    const weight = try prim_mod.primMstWeight(allocator, adj, 0);
    return intToU64(i64, weight);
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

fn runRomanToIntegerMany(romans: []const []const u8) !u64 {
    var checksum: u64 = 0;
    for (romans) |roman| {
        const value = try roman_to_int_mod.romanToInteger(roman);
        checksum +%= (@as(u64, value) *% 3) +% @as(u64, @intCast(roman.len));
    }
    return checksum;
}

fn runIntegerToRomanMany(allocator: Allocator, numbers: []const u32) !u64 {
    var checksum: u64 = 0;
    for (numbers) |n| {
        const roman = try int_to_roman_mod.integerToRoman(allocator, n);
        checksum +%= checksumBytes(roman) +% @as(u64, n);
        allocator.free(roman);
    }
    return checksum;
}

fn runTemperatureConversion(cases: []const TemperatureCase) !u64 {
    var checksum: u64 = 0;
    for (cases) |c| {
        const converted = try temp_conv_mod.convertTemperature(c.value, c.from, c.to);
        const scaled = converted * 1_000_000.0;
        const quantized: i64 = @intFromFloat(@round(scaled));
        checksum +%= intToU64(i64, quantized);
    }
    return checksum;
}

fn runCaesarCipher(allocator: Allocator, text: []const u8, key: i64) !u64 {
    const encrypted = try caesar_cipher_mod.encrypt(allocator, text, key, null);
    defer allocator.free(encrypted);

    const decrypted = try caesar_cipher_mod.decrypt(allocator, encrypted, key, null);
    defer allocator.free(decrypted);

    if (!std.mem.eql(u8, decrypted, text)) return error.RoundTripMismatch;

    return checksumBytes(encrypted) +
        (checksumBytes(decrypted) *% 3) +
        @as(u64, @intCast(text.len));
}

fn runSha256(allocator: Allocator, payload: []const u8) !u64 {
    const digest_hex = try sha256_mod.sha256Hex(allocator, payload);
    defer allocator.free(digest_hex);

    const first: u64 = if (payload.len == 0) 0 else payload[0];
    const mid: u64 = if (payload.len == 0) 0 else payload[payload.len / 2];
    const last: u64 = if (payload.len == 0) 0 else payload[payload.len - 1];

    return checksumBytes(digest_hex) +
        first +
        (mid *% 3) +
        (last *% 5) +
        @as(u64, @intCast(payload.len));
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

fn runAhoCorasick(allocator: Allocator, patterns: []const []const u8, text: []const u8) !u64 {
    const matches = try aho_corasick_mod.findMatches(allocator, patterns, text);
    defer allocator.free(matches);

    const counts = try allocator.alloc(usize, patterns.len);
    defer allocator.free(counts);
    @memset(counts, 0);

    var pos_sum: u64 = 0;
    for (matches) |m| {
        if (m.pattern_index < counts.len) counts[m.pattern_index] += 1;
        pos_sum +%= (@as(u64, @intCast(m.position + 1)) *% 1_315_423_911) +% @as(u64, @intCast(m.pattern_index));
    }

    var total: u64 = 0;
    for (counts, 0..) |count, i| {
        total +%= @as(u64, @intCast(i + 1)) *% @as(u64, @intCast(count));
    }
    return total +% pos_sum;
}

fn runSuffixArray(allocator: Allocator, text: []const u8) !u64 {
    const sa = try suffix_array_mod.suffixArray(allocator, text);
    defer allocator.free(sa);
    const lcp = try suffix_array_mod.lcpArray(allocator, text, sa);
    defer allocator.free(lcp);

    return checksumSlice(usize, sa) +% (checksumSlice(usize, lcp) *% 3) +% (@as(u64, @intCast(text.len)) *% 7);
}

fn runRunLengthEncoding(allocator: Allocator, text: []const u8) !u64 {
    const encoded = try run_length_encoding_mod.runLengthEncode(allocator, text);
    defer allocator.free(encoded);
    const decoded = try run_length_encoding_mod.runLengthDecode(allocator, encoded);
    defer allocator.free(decoded);
    if (!std.mem.eql(u8, text, decoded)) return error.RoundTripMismatch;

    const first_count: usize = if (encoded.len == 0) 0 else encoded[0].count;
    const last_count: usize = if (encoded.len == 0) 0 else encoded[encoded.len - 1].count;
    return checksumBytes(decoded) +
        (@as(u64, @intCast(encoded.len)) *% 3) +
        (@as(u64, @intCast(first_count)) *% 5) +
        (@as(u64, @intCast(last_count)) *% 7);
}

fn runActivitySelection(allocator: Allocator, start: []const i64, finish: []const i64) !u64 {
    const selected = try activity_selection_mod.activitySelection(allocator, start, finish);
    defer allocator.free(selected);
    return checksumSlice(usize, selected);
}

fn runHuffmanCoding(allocator: Allocator, text: []const u8) !u64 {
    const codes = try huffman_coding_mod.buildHuffmanCodes(allocator, text);
    defer huffman_coding_mod.freeHuffmanCodes(allocator, codes);

    const encoded = try huffman_coding_mod.encodeText(allocator, text, codes);
    defer allocator.free(encoded);
    const decoded = try huffman_coding_mod.decodeBits(allocator, encoded, codes);
    defer allocator.free(decoded);

    if (!std.mem.eql(u8, decoded, text)) return error.RoundTripMismatch;

    const first_bit: u64 = if (encoded.len > 0 and encoded[0] == '1') 1 else 0;
    const last_bit: u64 = if (encoded.len > 0 and encoded[encoded.len - 1] == '1') 1 else 0;

    return checksumBytes(decoded) +
        (@as(u64, @intCast(encoded.len)) *% 3) +
        (@as(u64, @intCast(codes.len)) *% 5) +
        (first_bit *% 7) +
        (last_bit *% 11);
}

fn runJobSequencingWithDeadline(allocator: Allocator, jobs: []const job_seq_mod.Job) !u64 {
    const result = try job_seq_mod.jobSequencingWithDeadlines(allocator, jobs);
    defer allocator.free(result.slots);
    return @as(u64, @intCast(result.count)) +% (intToU64(i64, result.max_profit) *% 3);
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

fn runPermutations(allocator: Allocator, items_base: []const i32) !u64 {
    const items = try allocator.dupe(i32, items_base);
    defer allocator.free(items);

    var result = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result.items) |perm| allocator.free(perm);
        result.deinit(allocator);
    }
    try permutations_mod.permutations(allocator, items, 0, &result);

    var total: u64 = 0;
    for (result.items) |perm| {
        var sig: u64 = 0;
        for (perm, 0..) |v, idx| {
            sig +%= (@as(u64, @intCast(idx + 1)) *% intToU64(i32, v));
        }
        total +%= sig;
    }

    return total +
        (@as(u64, @intCast(result.items.len)) *% 17) +
        @as(u64, @intCast(items_base.len));
}

fn runCombinations(allocator: Allocator, n: usize, k: usize) !u64 {
    var result = std.ArrayListUnmanaged([]usize){};
    defer {
        for (result.items) |combo| allocator.free(combo);
        result.deinit(allocator);
    }
    try combinations_mod.combinations(allocator, n, k, &result);

    var total: u64 = 0;
    for (result.items) |combo| {
        var sig = @as(u64, @intCast(combo.len)) *% 11;
        for (combo, 0..) |v, idx| {
            sig +%= @as(u64, @intCast(idx + 1)) *% @as(u64, @intCast(v));
        }
        total +%= sig;
    }

    return total +
        (@as(u64, @intCast(result.items.len)) *% 13) +
        (@as(u64, @intCast(n)) *% 5) +
        @as(u64, @intCast(k));
}

fn runSubsets(allocator: Allocator, items: []const i32) !u64 {
    var result = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result.items) |subset| allocator.free(subset);
        result.deinit(allocator);
    }
    try subsets_mod.allSubsets(allocator, items, &result);

    var total: u64 = 0;
    for (result.items) |subset| {
        var sig = @as(u64, @intCast(subset.len)) *% 19;
        for (subset, 0..) |v, idx| {
            const adjusted: i64 = @as(i64, v) + 37;
            sig +%= @as(u64, @intCast(idx + 1)) *% intToU64(i64, adjusted);
        }
        total +%= sig;
    }

    return total +
        (@as(u64, @intCast(result.items.len)) *% 7) +
        (@as(u64, @intCast(items.len)) *% 3);
}

fn runGenerateParentheses(allocator: Allocator, n: usize) !u64 {
    var result = std.ArrayListUnmanaged([]u8){};
    defer {
        for (result.items) |s| allocator.free(s);
        result.deinit(allocator);
    }
    try generate_parentheses_mod.generateParentheses(allocator, n, &result);

    var total: u64 = 0;
    for (result.items) |s| total +%= checksumBytes(s);

    return total +
        (@as(u64, @intCast(result.items.len)) *% 23) +
        @as(u64, @intCast(n));
}

fn runNQueens(allocator: Allocator, n: usize) !u64 {
    const count = try n_queens_mod.nQueensCount(allocator, n);
    return (@as(u64, @intCast(count)) *% 97) +% @as(u64, @intCast(n));
}

fn runSudoku(solvable: [9][9]u8, unsolvable: [9][9]u8) !u64 {
    var grid = solvable;
    if (!sudoku_mod.solve(&grid)) return error.SudokuExpectedSolved;

    var flat: [81]u8 = undefined;
    var weighted: u64 = 0;
    var idx: usize = 0;
    for (grid) |row| {
        for (row) |cell| {
            flat[idx] = cell;
            weighted +%= (@as(u64, @intCast(idx + 1)) *% @as(u64, cell));
            idx += 1;
        }
    }

    var impossible = unsolvable;
    const unsolved_ok = !sudoku_mod.solve(&impossible);
    return checksumSlice(u8, flat[0..]) +
        weighted +
        (if (unsolved_ok) @as(u64, 97) else 0);
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
    const ext_pairs = try allocator.alloc(ExtPair, 4_000);
    defer allocator.free(ext_pairs);
    for (0..ext_pairs.len) |i| {
        const idx = i * 2;
        var a: i64 = @intCast(math_values[idx] % 200_000);
        var b: i64 = @intCast(math_values[idx + 1] % 200_000);
        a -= 100_000;
        b -= 100_000;
        if (a == 0 and b == 0) b = 1;
        ext_pairs[i] = .{ .a = a, .b = b };
    }

    const modinv_pairs = try allocator.alloc(ModInvPair, 2_000);
    defer allocator.free(modinv_pairs);
    for (0..modinv_pairs.len) |i| {
        const idx: u64 = @intCast(i);
        var m: i64 = @intCast(((idx * 37) + 101) % 50_000 + 3);
        if (@mod(m, 2) == 0) m += 1;

        const m_u64: u64 = @intCast(m);
        var a: i64 = @intCast(((idx * 97) + 31) % m_u64);
        if (a == 0) a = 1;
        while (gcd_mod.gcd(a, m) != 1) {
            a = @mod(a + 1, m);
            if (a == 0) a = 1;
        }
        modinv_pairs[i] = .{ .a = a, .m = m };
    }

    const totient_inputs = try allocator.alloc(u64, 20_000);
    defer allocator.free(totient_inputs);
    for (0..totient_inputs.len) |i| {
        totient_inputs[i] = (math_values[i] % 1_000_000) + 1;
    }

    const crt_systems = try allocator.alloc(CrtSystem, 2_500);
    defer allocator.free(crt_systems);
    for (0..crt_systems.len) |i| {
        const i_u64: u64 = @intCast(i);
        crt_systems[i] = .{
            .remainders = .{
                @intCast(i_u64 % 3),
                @intCast((i_u64 * 2 + 1) % 5),
                @intCast((i_u64 * 3 + 2) % 7),
            },
            .moduli = .{ 3, 5, 7 },
        };
    }

    const binom_pairs = try allocator.alloc(BinomPair, 3_000);
    defer allocator.free(binom_pairs);
    for (0..binom_pairs.len) |i| {
        const idx: u64 = @intCast(i);
        binom_pairs[i] = .{
            .n = ((idx * 19) + 20) % 47 + 20,
            .k = ((idx * 11) + 7) % 20 + 1,
        };
    }

    const isqrt_inputs = try allocator.alloc(u64, 40_000);
    defer allocator.free(isqrt_inputs);
    const isqrt_modulus: u64 = (@as(u64, 1) << 63) - 1;
    for (0..isqrt_inputs.len) |i| {
        const idx: u64 = @intCast(i);
        isqrt_inputs[i] = ((idx * 9_999_991) + 1_234_567_891) % isqrt_modulus;
    }
    const trie_words_owned = try generateBase26Words(allocator, 12_000, 6);
    defer {
        for (trie_words_owned) |word| allocator.free(word);
        allocator.free(trie_words_owned);
    }
    const trie_words = try allocator.alloc([]const u8, trie_words_owned.len);
    defer allocator.free(trie_words);
    for (trie_words_owned, 0..) |word, idx| trie_words[idx] = word;
    const disjoint_set_n: usize = 50_000;
    const avl_values = try allocator.alloc(i64, 50_000);
    defer allocator.free(avl_values);
    for (0..avl_values.len) |i| {
        const idx: u64 = @intCast(i);
        avl_values[i] = @intCast(((idx * 73) + 19) % 50_000);
    }
    const avl_queries = try allocator.alloc(i64, 20_000);
    defer allocator.free(avl_queries);
    for (0..avl_queries.len) |i| {
        const idx: u64 = @intCast(i);
        avl_queries[i] = @intCast(((idx * 97) + 31) % 50_000);
    }
    const max_heap_values = try generateIntData64(allocator, 50_000);
    defer allocator.free(max_heap_values);
    const priority_queue_n: usize = 60_000;
    const hash_map_n: usize = 60_000;
    const segment_values = try generateIntData64(allocator, 60_000);
    defer allocator.free(segment_values);
    const fenwick_values = try generateIntData64(allocator, 60_000);
    defer allocator.free(fenwick_values);
    const rb_values = try allocator.alloc(i64, 70_000);
    defer allocator.free(rb_values);
    for (0..rb_values.len) |i| {
        const idx: u64 = @intCast(i);
        const value: i64 = @intCast(((idx * 73) + 19) % 50_000);
        rb_values[i] = value - 25_000;
    }
    const rb_queries = try allocator.alloc(i64, 30_000);
    defer allocator.free(rb_queries);
    for (0..rb_queries.len) |i| {
        const idx: u64 = @intCast(i);
        const value: i64 = @intCast(((idx * 97) + 31) % 80_000);
        rb_queries[i] = value - 40_000;
    }
    const lru_capacity: usize = 4_096;
    const lru_ops: usize = 80_000;
    const deque_ops: usize = 90_000;
    const stack_values = try generateIntData64(allocator, 70_000);
    defer allocator.free(stack_values);
    const queue_values = try generateIntData64(allocator, 70_000);
    defer allocator.free(queue_values);
    const singly_list_values = try generateIntData64(allocator, 24_000);
    defer allocator.free(singly_list_values);
    const doubly_list_values = try generateIntData64(allocator, 24_000);
    defer allocator.free(doubly_list_values);
    const bst_values = try allocator.alloc(i64, 20_000);
    defer allocator.free(bst_values);
    for (0..bst_values.len) |i| {
        bst_values[i] = @intCast((i * 73) + 19);
    }
    const bst_queries = try allocator.alloc(i64, 12_000);
    defer allocator.free(bst_queries);
    for (0..bst_queries.len) |i| {
        const base = bst_values[i % bst_values.len];
        bst_queries[i] = if (i % 2 == 0) base else base + 1;
    }
    const bst_removals = try allocator.alloc(i64, 4_000);
    defer allocator.free(bst_removals);
    for (0..bst_removals.len) |i| {
        bst_removals[i] = bst_values[(i * 5) % bst_values.len];
    }
    const min_heap_bench_values = try generateIntData64(allocator, 20_000);
    defer allocator.free(min_heap_bench_values);
    const min_heap_push_count: usize = 8_000;

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
    const aho_patterns_owned = try generateBase26Words(allocator, 600, 4);
    defer {
        for (aho_patterns_owned) |word| allocator.free(word);
        allocator.free(aho_patterns_owned);
    }
    const aho_patterns = try allocator.alloc([]const u8, aho_patterns_owned.len);
    defer allocator.free(aho_patterns);
    for (aho_patterns_owned, 0..) |word, idx| aho_patterns[idx] = word;

    var aho_text_builder = std.ArrayListUnmanaged(u8){};
    defer aho_text_builder.deinit(allocator);
    for (0..18_000) |i| {
        const pattern_idx = (i * 37 + 11) % aho_patterns.len;
        try aho_text_builder.appendSlice(allocator, aho_patterns[pattern_idx]);
        try aho_text_builder.append(allocator, 'x');
    }
    const aho_text = try aho_text_builder.toOwnedSlice(allocator);
    defer allocator.free(aho_text);

    const suffix_text = try generateAsciiString(allocator, 12_000, 7, 3);
    defer allocator.free(suffix_text);
    const rle_text = try generateRleText(allocator, 20_000);
    defer allocator.free(rle_text);
    const caesar_text = try repeatString(allocator, "The quick brown fox jumps over the lazy dog 0123456789! ", 6_000);
    defer allocator.free(caesar_text);
    const caesar_key: i64 = 8_000;
    const sha_payload = try generateAsciiString(allocator, 220_000, 17, 5);
    defer allocator.free(sha_payload);
    const permutations_items = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const combinations_n: usize = 16;
    const combinations_k: usize = 8;
    const subset_items = [_]i32{ 3, 8, 13, 18, 23, 28, 2, 7, 12, 17, 22, 27, 1, 6 };
    const parentheses_n: usize = 9;
    const n_queens_n: usize = 10;
    const sudoku_solvable = [9][9]u8{
        [_]u8{ 3, 0, 6, 5, 0, 8, 4, 0, 0 },
        [_]u8{ 5, 2, 0, 0, 0, 0, 0, 0, 0 },
        [_]u8{ 0, 8, 7, 0, 0, 0, 0, 3, 1 },
        [_]u8{ 0, 0, 3, 0, 1, 0, 0, 8, 0 },
        [_]u8{ 9, 0, 0, 8, 6, 3, 0, 0, 5 },
        [_]u8{ 0, 5, 0, 0, 9, 0, 6, 0, 0 },
        [_]u8{ 1, 3, 0, 0, 0, 0, 2, 5, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 7, 4 },
        [_]u8{ 0, 0, 5, 2, 0, 6, 3, 0, 0 },
    };
    const sudoku_unsolvable = [9][9]u8{
        [_]u8{ 5, 0, 6, 5, 0, 8, 4, 0, 3 },
        [_]u8{ 5, 2, 0, 0, 0, 0, 0, 0, 2 },
        [_]u8{ 1, 8, 7, 0, 0, 0, 0, 3, 1 },
        [_]u8{ 0, 0, 3, 0, 1, 0, 0, 8, 0 },
        [_]u8{ 9, 0, 0, 8, 6, 3, 0, 0, 5 },
        [_]u8{ 0, 5, 0, 0, 9, 0, 6, 0, 0 },
        [_]u8{ 1, 3, 0, 0, 0, 0, 2, 5, 0 },
        [_]u8{ 0, 0, 0, 0, 0, 0, 0, 7, 4 },
        [_]u8{ 0, 0, 5, 2, 0, 6, 3, 0, 0 },
    };

    const graph_adj_owned = try buildGraphAdj(allocator, 6_000);
    defer freeGraphAdj(allocator, graph_adj_owned);
    const graph_adj = try allocator.alloc([]const usize, graph_adj_owned.len);
    defer allocator.free(graph_adj);
    for (graph_adj_owned, 0..) |row, i| graph_adj[i] = row;

    const weighted_graph_adj_owned = try buildWeightedGraphAdj(dijkstra_mod.Edge, allocator, 2_200);
    defer freeWeightedGraphAdj(dijkstra_mod.Edge, allocator, weighted_graph_adj_owned);
    const weighted_graph_adj = try allocator.alloc([]const dijkstra_mod.Edge, weighted_graph_adj_owned.len);
    defer allocator.free(weighted_graph_adj);
    for (weighted_graph_adj_owned, 0..) |row, i| weighted_graph_adj[i] = row;

    const weighted_graph_astar_owned = try buildWeightedGraphAdj(a_star_mod.Edge, allocator, 2_200);
    defer freeWeightedGraphAdj(a_star_mod.Edge, allocator, weighted_graph_astar_owned);
    const weighted_graph_astar = try allocator.alloc([]const a_star_mod.Edge, weighted_graph_astar_owned.len);
    defer allocator.free(weighted_graph_astar);
    for (weighted_graph_astar_owned, 0..) |row, i| weighted_graph_astar[i] = row;
    const weighted_graph_heuristics = try allocator.alloc(u64, weighted_graph_astar.len);
    defer allocator.free(weighted_graph_heuristics);
    @memset(weighted_graph_heuristics, 0);
    const weighted_graph_goal = weighted_graph_astar.len - 1;

    const tarjan_adj_owned = try buildTarjanAdj(allocator, 700, 5);
    defer freeGraphAdj(allocator, tarjan_adj_owned);
    const tarjan_adj = try allocator.alloc([]const usize, tarjan_adj_owned.len);
    defer allocator.free(tarjan_adj);
    for (tarjan_adj_owned, 0..) |row, i| tarjan_adj[i] = row;

    const bridges_adj_owned = try buildBridgesAdj(allocator, 600);
    defer freeGraphAdj(allocator, bridges_adj_owned);
    const bridges_adj = try allocator.alloc([]const usize, bridges_adj_owned.len);
    defer allocator.free(bridges_adj);
    for (bridges_adj_owned, 0..) |row, i| bridges_adj[i] = row;

    const euler_adj_owned = try buildEulerChainAdj(allocator, 4_000);
    defer freeGraphAdj(allocator, euler_adj_owned);
    const euler_adj = try allocator.alloc([]const usize, euler_adj_owned.len);
    defer allocator.free(euler_adj);
    for (euler_adj_owned, 0..) |row, i| euler_adj[i] = row;

    const flow_n: usize = 120;
    const flow_capacity_flat = try buildFlowCapacityFlat(allocator, flow_n);
    defer allocator.free(flow_capacity_flat);
    const flow_capacity_rows = try allocator.alloc([]const i64, flow_n);
    defer allocator.free(flow_capacity_rows);
    for (0..flow_n) |i| {
        flow_capacity_rows[i] = flow_capacity_flat[i * flow_n .. (i + 1) * flow_n];
    }

    const bipartite_adj_owned = try buildBipartiteAdj(allocator, 6_000);
    defer freeGraphAdj(allocator, bipartite_adj_owned);
    const bipartite_adj = try allocator.alloc([]const usize, bipartite_adj_owned.len);
    defer allocator.free(bipartite_adj);
    for (bipartite_adj_owned, 0..) |row, i| bipartite_adj[i] = row;

    const bellman_n: usize = 1_800;
    const bellman_edges = try buildBellmanEdges(allocator, bellman_n);
    defer allocator.free(bellman_edges);

    const floyd_n: usize = 120;
    const floyd_inf: i64 = 1_000_000_000_000;
    const floyd_mat = try buildFloydMatrix(allocator, floyd_n, floyd_inf);
    defer allocator.free(floyd_mat);

    const cycle_graph_adj_owned = try buildCycleGraphAdj(allocator, 600);
    defer freeGraphAdj(allocator, cycle_graph_adj_owned);
    const cycle_graph_adj = try allocator.alloc([]const usize, cycle_graph_adj_owned.len);
    defer allocator.free(cycle_graph_adj);
    for (cycle_graph_adj_owned, 0..) |row, i| cycle_graph_adj[i] = row;

    const component_adj_owned = try buildComponentAdj(allocator, 6_000);
    defer freeGraphAdj(allocator, component_adj_owned);
    const component_adj = try allocator.alloc([]const usize, component_adj_owned.len);
    defer allocator.free(component_adj);
    for (component_adj_owned, 0..) |row, i| component_adj[i] = row;

    const mst_n: usize = 1_800;
    const mst_edges = try buildMstEdgesKruskal(allocator, mst_n);
    defer allocator.free(mst_edges);
    const mst_adj_owned = try buildMstAdjPrim(allocator, mst_n);
    defer freePrimAdj(allocator, mst_adj_owned);
    const mst_adj = try allocator.alloc([]const prim_mod.Edge, mst_adj_owned.len);
    defer allocator.free(mst_adj);
    for (mst_adj_owned, 0..) |row, i| mst_adj[i] = row;

    const knapsack_weights = try allocator.alloc(usize, 180);
    defer allocator.free(knapsack_weights);
    const knapsack_values = try allocator.alloc(usize, 180);
    defer allocator.free(knapsack_values);
    for (0..180) |i| {
        const idx: u64 = @intCast(i);
        knapsack_weights[i] = @intCast(((idx * 73) + 19) % 40 + 1);
        knapsack_values[i] = @intCast(((idx * 97) + 53) % 500 + 1);
    }

    const rod_prices = try allocator.alloc(i64, 220);
    defer allocator.free(rod_prices);
    for (0..rod_prices.len) |i| {
        const idx: u64 = @intCast(i);
        rod_prices[i] = @intCast(((idx * 37) + 11) % 600 + 1);
    }
    const rod_length: usize = 200;

    const mcm_dims = try allocator.alloc(usize, 71);
    defer allocator.free(mcm_dims);
    for (0..mcm_dims.len) |i| {
        const idx: u64 = @intCast(i);
        mcm_dims[i] = @intCast(((idx * 13) + 7) % 50 + 5);
    }

    const pal_part_text = try repeatString(allocator, "abacdcaba", 80);
    defer allocator.free(pal_part_text);

    const word_break_text = try repeatString(allocator, "zigisfastandsafe", 3_000);
    defer allocator.free(word_break_text);
    const word_break_dict = [_][]const u8{ "zig", "is", "fast", "and", "safe" };

    const catalan_inputs = try allocator.alloc(u32, 30);
    defer allocator.free(catalan_inputs);
    for (0..30) |i| catalan_inputs[i] = @intCast(i + 1);

    const subset_numbers = try allocator.alloc(i64, 72);
    defer allocator.free(subset_numbers);
    for (0..subset_numbers.len) |i| {
        subset_numbers[i] = @intCast(((i * 17) + 5) % 50 + 1);
    }

    const subset_targets = try allocator.alloc(i64, 64);
    defer allocator.free(subset_targets);
    for (0..subset_targets.len) |i| {
        subset_targets[i] = @intCast(((i * 97) + 31) % 1200);
    }

    const egg_drop_cases = try allocator.alloc(EggDropCase, 220);
    defer allocator.free(egg_drop_cases);
    for (0..egg_drop_cases.len) |i| {
        egg_drop_cases[i] = .{
            .eggs = (i % 10) + 2,
            .floors = ((i * 131) + 17) % 5000 + 1,
        };
    }

    const lps_text = try generateAsciiString(allocator, 700, 7, 3);
    defer allocator.free(lps_text);

    const max_product_array = try allocator.alloc(i64, 90_000);
    defer allocator.free(max_product_array);
    for (0..max_product_array.len) |i| {
        if (i % 8 == 0) {
            max_product_array[i] = 0;
        } else {
            const value: i64 = @intCast(((i * 73) + 19) % 7);
            max_product_array[i] = value - 3;
        }
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

    const activity_n: usize = 60_000;
    const activity_start = try allocator.alloc(i64, activity_n);
    defer allocator.free(activity_start);
    const activity_finish = try allocator.alloc(i64, activity_n);
    defer allocator.free(activity_finish);
    for (0..activity_n) |i| {
        const base: i64 = @intCast(i);
        activity_finish[i] = base + 1;
        activity_start[i] = if (i % 4 == 0 and i > 0) base - 1 else base;
    }
    if (activity_n > 0) activity_start[0] = 0;

    var huffman_text_builder = std.ArrayListUnmanaged(u8){};
    defer huffman_text_builder.deinit(allocator);
    for (0..12) |i| {
        const ch: u8 = @as(u8, 'a') + @as(u8, @intCast(i));
        const repeat: usize = 4_000 - i * 250;
        const old_len = huffman_text_builder.items.len;
        try huffman_text_builder.resize(allocator, old_len + repeat);
        @memset(huffman_text_builder.items[old_len .. old_len + repeat], ch);
    }
    const huffman_text = try huffman_text_builder.toOwnedSlice(allocator);
    defer allocator.free(huffman_text);

    const job_data = try allocator.alloc(job_seq_mod.Job, 30_000);
    defer allocator.free(job_data);
    for (0..job_data.len) |i| {
        job_data[i] = .{
            .id = @intCast(i + 1),
            .deadline = ((i * 7) % 600) + 1,
            .profit = @intCast(((i * 97) % 900) + 50),
        };
    }

    const roman_numbers = try allocator.alloc(u32, 10_000);
    defer allocator.free(roman_numbers);
    for (0..roman_numbers.len) |i| {
        roman_numbers[i] = @intCast(((i * 37) % 3999) + 1);
    }
    const roman_samples_owned = try allocator.alloc([]u8, roman_numbers.len);
    defer {
        for (roman_samples_owned) |s| allocator.free(s);
        allocator.free(roman_samples_owned);
    }
    for (roman_numbers, 0..) |n, i| {
        roman_samples_owned[i] = try int_to_roman_mod.integerToRoman(allocator, n);
    }
    const roman_samples = try allocator.alloc([]const u8, roman_numbers.len);
    defer allocator.free(roman_samples);
    for (roman_samples_owned, 0..) |s, i| roman_samples[i] = s;

    const temperature_cases = try allocator.alloc(TemperatureCase, 24_000);
    defer allocator.free(temperature_cases);
    for (0..temperature_cases.len) |i| {
        const from_scale: temp_conv_mod.Scale = switch (i % 4) {
            0 => .celsius,
            1 => .fahrenheit,
            2 => .kelvin,
            else => .rankine,
        };
        const to_scale: temp_conv_mod.Scale = switch ((i * 3 + 1) % 4) {
            0 => .celsius,
            1 => .fahrenheit,
            2 => .kelvin,
            else => .rankine,
        };
        const value: f64 = switch (from_scale) {
            .celsius => @as(f64, @floatFromInt((i * 17) % 7000)) / 10.0 - 273.0,
            .fahrenheit => @as(f64, @floatFromInt((i * 13) % 8000)) / 10.0 - 459.0,
            .kelvin => @as(f64, @floatFromInt((i * 11) % 9000)) / 10.0,
            .rankine => @as(f64, @floatFromInt((i * 19) % 9000)) / 10.0,
        };
        temperature_cases[i] = .{
            .value = value,
            .from = from_scale,
            .to = to_scale,
        };
    }

    const miller_rabin_values = try allocator.alloc(u64, 45_008);
    defer allocator.free(miller_rabin_values);
    for (0..45_000) |i| {
        const idx: u64 = @intCast(i);
        miller_rabin_values[i] = ((idx * 48_271 + 12_345) % 2_000_000) + 2;
    }
    const miller_extras = [_]u64{
        561,
        563,
        838_201,
        838_207,
        3_078_386_641,
        3_078_386_653,
        18_446_744_073_709_551_556,
        18_446_744_073_709_551_557,
    };
    for (miller_extras, 0..) |v, i| {
        miller_rabin_values[45_000 + i] = v;
    }

    const matrix_exp_dim: usize = 12;
    const matrix_exp_base = try generateMatrixData(allocator, matrix_exp_dim * matrix_exp_dim, 43, 17, 31, 15);
    defer allocator.free(matrix_exp_base);
    const matrix_exp_exponent: u64 = 17;

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

    // Maths (16)
    try benchRunMaybe(filter_algorithm, &filter_matched, runMathGcd, stdout, "gcd", "maths", 6, .{math_values});
    try benchRunMaybe(filter_algorithm, &filter_matched, runMathLcm, stdout, "lcm", "maths", 6, .{math_values});
    try benchRunMaybe(filter_algorithm, &filter_matched, runFibonacciMany, stdout, "fibonacci", "maths", 200, .{fib_inputs});
    try benchRunMaybe(filter_algorithm, &filter_matched, runFactorialMany, stdout, "factorial", "maths", 200, .{fact_inputs});
    try benchRunMaybe(filter_algorithm, &filter_matched, runPowerMany, stdout, "power", "maths", 300, .{ power_bases, power_exps });
    try benchRunMaybe(filter_algorithm, &filter_matched, runPrimeCheck, stdout, "prime_check", "maths", 20, .{});
    try benchRunMaybe(filter_algorithm, &filter_matched, runSieve, stdout, "sieve_of_eratosthenes", "maths", 6, .{allocator});
    try benchRunMaybe(filter_algorithm, &filter_matched, runCollatzSteps, stdout, "collatz_sequence", "maths", 8, .{});
    try benchRunMaybe(filter_algorithm, &filter_matched, runExtendedEuclidean, stdout, "extended_euclidean", "maths", 20, .{ext_pairs});
    try benchRunMaybe(filter_algorithm, &filter_matched, runModularInverse, stdout, "modular_inverse", "maths", 20, .{modinv_pairs});
    try benchRunMaybe(filter_algorithm, &filter_matched, runTotient, stdout, "eulers_totient", "maths", 12, .{totient_inputs});
    try benchRunMaybe(filter_algorithm, &filter_matched, runCrt, stdout, "chinese_remainder_theorem", "maths", 25, .{crt_systems});
    try benchRunMaybe(filter_algorithm, &filter_matched, runBinomial, stdout, "binomial_coefficient", "maths", 20, .{binom_pairs});
    try benchRunMaybe(filter_algorithm, &filter_matched, runIntegerSquareRoot, stdout, "integer_square_root", "maths", 25, .{isqrt_inputs});
    try benchRunMaybe(filter_algorithm, &filter_matched, runMillerRabin, stdout, "miller_rabin", "maths", 12, .{miller_rabin_values});
    try benchRunMaybe(filter_algorithm, &filter_matched, runMatrixExponentiation, stdout, "matrix_exponentiation", "maths", 80, .{ allocator, matrix_exp_base, matrix_exp_dim, matrix_exp_exponent });

    // Data Structures (17)
    try benchRunMaybe(filter_algorithm, &filter_matched, runStack, stdout, "stack", "data_structures", 10, .{ stack_values, allocator });
    try benchRunMaybe(filter_algorithm, &filter_matched, runQueue, stdout, "queue", "data_structures", 10, .{ queue_values, allocator });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSinglyLinkedList, stdout, "singly_linked_list", "data_structures", 8, .{ singly_list_values, allocator });
    try benchRunMaybe(filter_algorithm, &filter_matched, runDoublyLinkedList, stdout, "doubly_linked_list", "data_structures", 8, .{ doubly_list_values, allocator });
    try benchRunMaybe(filter_algorithm, &filter_matched, runBinarySearchTree, stdout, "binary_search_tree", "data_structures", 6, .{ bst_values, bst_queries, bst_removals, allocator });
    try benchRunMaybe(filter_algorithm, &filter_matched, runMinHeap, stdout, "min_heap", "data_structures", 8, .{ min_heap_bench_values, min_heap_push_count, allocator });
    try benchRunMaybe(filter_algorithm, &filter_matched, runTrie, stdout, "trie", "data_structures", 8, .{ allocator, trie_words });
    try benchRunMaybe(filter_algorithm, &filter_matched, runDisjointSet, stdout, "disjoint_set", "data_structures", 10, .{ allocator, disjoint_set_n });
    try benchRunMaybe(filter_algorithm, &filter_matched, runAvlTree, stdout, "avl_tree", "data_structures", 4, .{ allocator, avl_values, avl_queries });
    try benchRunMaybe(filter_algorithm, &filter_matched, runMaxHeap, stdout, "max_heap", "data_structures", 8, .{ allocator, max_heap_values });
    try benchRunMaybe(filter_algorithm, &filter_matched, runPriorityQueue, stdout, "priority_queue", "data_structures", 8, .{ allocator, priority_queue_n });
    try benchRunMaybe(filter_algorithm, &filter_matched, runHashMapOpenAddressing, stdout, "hash_map_open_addressing", "data_structures", 6, .{ allocator, hash_map_n });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSegmentTree, stdout, "segment_tree", "data_structures", 8, .{ allocator, segment_values });
    try benchRunMaybe(filter_algorithm, &filter_matched, runFenwickTree, stdout, "fenwick_tree", "data_structures", 10, .{ allocator, fenwick_values });
    try benchRunMaybe(filter_algorithm, &filter_matched, runRedBlackTree, stdout, "red_black_tree", "data_structures", 6, .{ allocator, rb_values, rb_queries });
    try benchRunMaybe(filter_algorithm, &filter_matched, runLruCache, stdout, "lru_cache", "data_structures", 8, .{ allocator, lru_capacity, lru_ops });
    try benchRunMaybe(filter_algorithm, &filter_matched, runDeque, stdout, "deque", "data_structures", 20, .{ allocator, deque_ops });

    // Dynamic Programming (17)
    try benchRunMaybe(filter_algorithm, &filter_matched, runClimbingStairsMany, stdout, "climbing_stairs", "dynamic_programming", 1000, .{stairs_inputs});
    try benchRunMaybe(filter_algorithm, &filter_matched, runFibonacciDp, stdout, "fibonacci_dp", "dynamic_programming", 400, .{allocator});
    try benchRunMaybe(filter_algorithm, &filter_matched, runCoinChange, stdout, "coin_change", "dynamic_programming", 80, .{ allocator, &coin_set });
    try benchRunMaybe(filter_algorithm, &filter_matched, runMaxSubarray, stdout, "max_subarray_sum", "dynamic_programming", 20, .{dp_array});
    try benchRunMaybe(filter_algorithm, &filter_matched, runLis, stdout, "longest_increasing_subsequence", "dynamic_programming", 12, .{ allocator, dp_array });
    try benchRunMaybe(filter_algorithm, &filter_matched, runRodCutting, stdout, "rod_cutting", "dynamic_programming", 80, .{ allocator, rod_prices, rod_length });
    try benchRunMaybe(filter_algorithm, &filter_matched, runMatrixChainMultiplication, stdout, "matrix_chain_multiplication", "dynamic_programming", 30, .{ allocator, mcm_dims });
    try benchRunMaybe(filter_algorithm, &filter_matched, runPalindromePartitioning, stdout, "palindrome_partitioning", "dynamic_programming", 60, .{ allocator, pal_part_text });
    try benchRunMaybe(filter_algorithm, &filter_matched, runWordBreak, stdout, "word_break", "dynamic_programming", 60, .{ allocator, word_break_text, &word_break_dict });
    try benchRunMaybe(filter_algorithm, &filter_matched, runCatalanMany, stdout, "catalan_numbers", "dynamic_programming", 120, .{ allocator, catalan_inputs });
    try benchRunMaybe(filter_algorithm, &filter_matched, runLcs, stdout, "longest_common_subsequence", "dynamic_programming", 30, .{ allocator, s1, s2 });
    try benchRunMaybe(filter_algorithm, &filter_matched, runEditDistance, stdout, "edit_distance", "dynamic_programming", 30, .{ allocator, s1, s2 });
    try benchRunMaybe(filter_algorithm, &filter_matched, runKnapsack, stdout, "knapsack", "dynamic_programming", 60, .{ allocator, knapsack_weights, knapsack_values });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSubsetSum, stdout, "subset_sum", "dynamic_programming", 8, .{ allocator, subset_numbers, subset_targets });
    try benchRunMaybe(filter_algorithm, &filter_matched, runEggDropProblem, stdout, "egg_drop_problem", "dynamic_programming", 120, .{ allocator, egg_drop_cases });
    try benchRunMaybe(filter_algorithm, &filter_matched, runLongestPalindromicSubsequence, stdout, "longest_palindromic_subsequence", "dynamic_programming", 25, .{ allocator, lps_text });
    try benchRunMaybe(filter_algorithm, &filter_matched, runMaxProductSubarray, stdout, "max_product_subarray", "dynamic_programming", 20, .{max_product_array});

    // Graphs (16)
    try benchRunMaybe(filter_algorithm, &filter_matched, runBfs, stdout, "bfs", "graphs", 12, .{ allocator, graph_adj });
    try benchRunMaybe(filter_algorithm, &filter_matched, runDfs, stdout, "dfs", "graphs", 12, .{ allocator, graph_adj });
    try benchRunMaybe(filter_algorithm, &filter_matched, runDijkstra, stdout, "dijkstra", "graphs", 8, .{ allocator, weighted_graph_adj });
    try benchRunMaybe(filter_algorithm, &filter_matched, runAStar, stdout, "a_star_search", "graphs", 8, .{ allocator, weighted_graph_astar, weighted_graph_heuristics, weighted_graph_goal });
    try benchRunMaybe(filter_algorithm, &filter_matched, runTarjanScc, stdout, "tarjan_scc", "graphs", 10, .{ allocator, tarjan_adj });
    try benchRunMaybe(filter_algorithm, &filter_matched, runBridges, stdout, "bridges", "graphs", 8, .{ allocator, bridges_adj });
    try benchRunMaybe(filter_algorithm, &filter_matched, runEulerianPathUndirected, stdout, "eulerian_path_circuit_undirected", "graphs", 20, .{ allocator, euler_adj });
    try benchRunMaybe(filter_algorithm, &filter_matched, runFordFulkerson, stdout, "ford_fulkerson", "graphs", 4, .{ allocator, flow_capacity_rows });
    try benchRunMaybe(filter_algorithm, &filter_matched, runBipartiteCheck, stdout, "bipartite_check_bfs", "graphs", 16, .{ allocator, bipartite_adj });
    try benchRunMaybe(filter_algorithm, &filter_matched, runBellmanFord, stdout, "bellman_ford", "graphs", 4, .{ allocator, bellman_n, bellman_edges });
    try benchRunMaybe(filter_algorithm, &filter_matched, runTopologicalSort, stdout, "topological_sort", "graphs", 12, .{ allocator, graph_adj });
    try benchRunMaybe(filter_algorithm, &filter_matched, runFloydWarshall, stdout, "floyd_warshall", "graphs", 2, .{ allocator, floyd_mat, floyd_n, floyd_inf });
    try benchRunMaybe(filter_algorithm, &filter_matched, runDetectCycle, stdout, "detect_cycle", "graphs", 20, .{ allocator, cycle_graph_adj });
    try benchRunMaybe(filter_algorithm, &filter_matched, runConnectedComponents, stdout, "connected_components", "graphs", 8, .{ allocator, component_adj });
    try benchRunMaybe(filter_algorithm, &filter_matched, runKruskal, stdout, "kruskal", "graphs", 6, .{ allocator, mst_n, mst_edges });
    try benchRunMaybe(filter_algorithm, &filter_matched, runPrim, stdout, "prim", "graphs", 6, .{ allocator, mst_adj });

    // Bit Manipulation (6)
    try benchRunMaybe(filter_algorithm, &filter_matched, runIsPowerOfTwo, stdout, "is_power_of_two", "bit_manipulation", 120, .{});
    try benchRunMaybe(filter_algorithm, &filter_matched, runCountSetBits, stdout, "count_set_bits", "bit_manipulation", 60, .{});
    try benchRunMaybe(filter_algorithm, &filter_matched, runFindUniqueFromSlice, stdout, "find_unique_number", "bit_manipulation", 1000, .{unique_arr});
    try benchRunMaybe(filter_algorithm, &filter_matched, runReverseBits, stdout, "reverse_bits", "bit_manipulation", 80, .{});
    try benchRunMaybe(filter_algorithm, &filter_matched, missing_number_mod.missingNumber, stdout, "missing_number", "bit_manipulation", 220, .{missing_nums});
    try benchRunMaybe(filter_algorithm, &filter_matched, runPowerOfFour, stdout, "power_of_4", "bit_manipulation", 120, .{});

    // Conversions (7)
    try benchRunMaybe(filter_algorithm, &filter_matched, runDecimalToBinary, stdout, "decimal_to_binary", "conversions", 120, .{allocator});
    try benchRunMaybe(filter_algorithm, &filter_matched, runBinaryToDecimalMany, stdout, "binary_to_decimal", "conversions", 120, .{bin_samples});
    try benchRunMaybe(filter_algorithm, &filter_matched, runDecimalToHex, stdout, "decimal_to_hexadecimal", "conversions", 120, .{allocator});
    try benchRunMaybe(filter_algorithm, &filter_matched, runBinaryToHex, stdout, "binary_to_hexadecimal", "conversions", 120, .{ allocator, bin_text });
    try benchRunMaybe(filter_algorithm, &filter_matched, runRomanToIntegerMany, stdout, "roman_to_integer", "conversions", 120, .{roman_samples});
    try benchRunMaybe(filter_algorithm, &filter_matched, runIntegerToRomanMany, stdout, "integer_to_roman", "conversions", 120, .{ allocator, roman_numbers });
    try benchRunMaybe(filter_algorithm, &filter_matched, runTemperatureConversion, stdout, "temperature_conversion", "conversions", 120, .{temperature_cases});

    // Ciphers (1)
    try benchRunMaybe(filter_algorithm, &filter_matched, runCaesarCipher, stdout, "caesar_cipher", "ciphers", 80, .{ allocator, caesar_text, caesar_key });

    // Hashing (1)
    try benchRunMaybe(filter_algorithm, &filter_matched, runSha256, stdout, "sha256", "hashing", 10, .{ allocator, sha_payload });

    // Greedy (7)
    try benchRunMaybe(filter_algorithm, &filter_matched, runBestStock, stdout, "best_time_to_buy_sell_stock", "greedy_methods", 20, .{prices});
    try benchRunMaybe(filter_algorithm, &filter_matched, runMinimumCoinChange, stdout, "minimum_coin_change", "greedy_methods", 200, .{ allocator, &coins_desc });
    try benchRunMaybe(filter_algorithm, &filter_matched, runMinimumWaitingTime, stdout, "minimum_waiting_time", "greedy_methods", 15, .{ allocator, waiting_queries });
    try benchRunMaybe(filter_algorithm, &filter_matched, runFractionalKnapsack, stdout, "fractional_knapsack", "greedy_methods", 600, .{ allocator, &frac_values, &frac_weights });
    try benchRunMaybe(filter_algorithm, &filter_matched, runActivitySelection, stdout, "activity_selection", "greedy_methods", 120, .{ allocator, activity_start, activity_finish });
    try benchRunMaybe(filter_algorithm, &filter_matched, runHuffmanCoding, stdout, "huffman_coding", "greedy_methods", 60, .{ allocator, huffman_text });
    try benchRunMaybe(filter_algorithm, &filter_matched, runJobSequencingWithDeadline, stdout, "job_sequencing_with_deadline", "greedy_methods", 15, .{ allocator, job_data });

    // Matrix (5)
    try benchRunMaybe(filter_algorithm, &filter_matched, runMatrixMultiply, stdout, "matrix_multiply", "matrix", 10, .{ allocator, matrix_a, matrix_b, matrix_dim });
    try benchRunMaybe(filter_algorithm, &filter_matched, runMatrixTranspose, stdout, "matrix_transpose", "matrix", 25, .{ allocator, transpose_mat, transpose_rows, transpose_cols });
    try benchRunMaybe(filter_algorithm, &filter_matched, runRotateMatrix, stdout, "rotate_matrix", "matrix", 25, .{ allocator, rotate_mat, rotate_n });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSpiral, stdout, "spiral_print", "matrix", 20, .{ allocator, spiral_mat, spiral_rows, spiral_cols });
    try benchRunMaybe(filter_algorithm, &filter_matched, runPascal, stdout, "pascal_triangle", "matrix", 120, .{ allocator, @as(usize, 180) });

    // Backtracking (6)
    try benchRunMaybe(filter_algorithm, &filter_matched, runPermutations, stdout, "permutations", "backtracking", 3, .{ allocator, &permutations_items });
    try benchRunMaybe(filter_algorithm, &filter_matched, runCombinations, stdout, "combinations", "backtracking", 6, .{ allocator, combinations_n, combinations_k });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSubsets, stdout, "subsets", "backtracking", 6, .{ allocator, &subset_items });
    try benchRunMaybe(filter_algorithm, &filter_matched, runGenerateParentheses, stdout, "generate_parentheses", "backtracking", 12, .{ allocator, parentheses_n });
    try benchRunMaybe(filter_algorithm, &filter_matched, runNQueens, stdout, "n_queens", "backtracking", 60, .{ allocator, n_queens_n });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSudoku, stdout, "sudoku_solver", "backtracking", 40, .{ sudoku_solvable, sudoku_unsolvable });

    // Strings (13)
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
    try benchRunMaybe(filter_algorithm, &filter_matched, runAhoCorasick, stdout, "aho_corasick", "strings", 20, .{ allocator, aho_patterns, aho_text });
    try benchRunMaybe(filter_algorithm, &filter_matched, runSuffixArray, stdout, "suffix_array", "strings", 6, .{ allocator, suffix_text });
    try benchRunMaybe(filter_algorithm, &filter_matched, runRunLengthEncoding, stdout, "run_length_encoding", "strings", 40, .{ allocator, rle_text });

    if (filter_algorithm != null and !filter_matched) {
        return error.UnknownBenchmarkAlgorithm;
    }
}
