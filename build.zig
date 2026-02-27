const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // 在循环外创建一次 test step，避免重复注册 panic
    const test_step = b.step("test", "Run all algorithm tests");

    // 每个算法文件注册为独立测试单元
    const test_files = [_][]const u8{
        "sorts/bubble_sort.zig",
        "sorts/insertion_sort.zig",
        "sorts/merge_sort.zig",
        "searches/linear_search.zig",
        "searches/binary_search.zig",
    };

    for (test_files) |file| {
        // Zig 0.15+ API：addTest 使用 root_module + createModule 模式
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(file),
                .target = target,
                .optimize = optimize,
            }),
        });
        const run = b.addRunArtifact(t);
        test_step.dependOn(&run.step);
    }
}
