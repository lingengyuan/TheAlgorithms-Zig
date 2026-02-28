const bench = @import("benchmarks/python_vs_zig/zig_bench_all.zig");

pub fn main() !void {
    try bench.main();
}
