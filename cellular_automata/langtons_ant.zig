//! Langton's Ant - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/cellular_automata/langtons_ant.py

const std = @import("std");
const testing = std.testing;

pub const LangtonsAntError = error{
    InvalidBoardSize,
    AntPositionOutOfBounds,
};

pub const Position = struct {
    x: i32,
    y: i32,
};

pub const LangtonsAnt = struct {
    allocator: std.mem.Allocator,
    board: [][]bool,
    width: usize,
    height: usize,
    ant_position: Position,
    ant_direction: u8,

    /// Initializes white board and ant state.
    ///
    /// Time complexity: O(width * height)
    /// Space complexity: O(width * height)
    pub fn init(allocator: std.mem.Allocator, width: usize, height: usize) !LangtonsAnt {
        if (width == 0 or height == 0) {
            return LangtonsAntError.InvalidBoardSize;
        }

        const board = try allocator.alloc([]bool, height);
        var built_rows: usize = 0;
        errdefer {
            for (0..built_rows) |i| allocator.free(board[i]);
            allocator.free(board);
        }

        for (0..height) |row| {
            board[row] = try allocator.alloc(bool, width);
            built_rows += 1;
            @memset(board[row], true);
        }

        return LangtonsAnt{
            .allocator = allocator,
            .board = board,
            .width = width,
            .height = height,
            .ant_position = .{ .x = @intCast(width / 2), .y = @intCast(height / 2) },
            .ant_direction = 3,
        };
    }

    pub fn deinit(self: *LangtonsAnt) void {
        for (self.board) |row| {
            self.allocator.free(row);
        }
        self.allocator.free(self.board);
    }

    fn inBounds(self: *const LangtonsAnt, pos: Position) bool {
        return pos.x >= 0 and pos.y >= 0 and
            @as(usize, @intCast(pos.x)) < self.height and
            @as(usize, @intCast(pos.y)) < self.width;
    }

    /// Performs one Langton ant move.
    ///
    /// Time complexity: O(1)
    /// Space complexity: O(1)
    pub fn moveAnt(self: *LangtonsAnt) LangtonsAntError!void {
        if (!self.inBounds(self.ant_position)) {
            return LangtonsAntError.AntPositionOutOfBounds;
        }

        const x: usize = @intCast(self.ant_position.x);
        const y: usize = @intCast(self.ant_position.y);

        if (self.board[x][y]) {
            self.ant_direction = (self.ant_direction + 1) % 4;
        } else {
            self.ant_direction = (self.ant_direction + 3) % 4;
        }

        const delta = switch (self.ant_direction) {
            0 => Position{ .x = -1, .y = 0 },
            1 => Position{ .x = 0, .y = 1 },
            2 => Position{ .x = 1, .y = 0 },
            else => Position{ .x = 0, .y = -1 },
        };

        self.board[x][y] = !self.board[x][y];
        self.ant_position = .{ .x = self.ant_position.x + delta.x, .y = self.ant_position.y + delta.y };
    }
};

test "langtons ant: python examples" {
    var ant = try LangtonsAnt.init(testing.allocator, 2, 2);
    defer ant.deinit();

    try testing.expect(ant.board[0][0] and ant.board[0][1] and ant.board[1][0] and ant.board[1][1]);
    try testing.expectEqual(@as(i32, 1), ant.ant_position.x);
    try testing.expectEqual(@as(i32, 1), ant.ant_position.y);

    try ant.moveAnt();
    try testing.expectEqualSlices(bool, &[_]bool{ true, true }, ant.board[0]);
    try testing.expectEqualSlices(bool, &[_]bool{ true, false }, ant.board[1]);

    try ant.moveAnt();
    try testing.expectEqualSlices(bool, &[_]bool{ true, false }, ant.board[0]);
    try testing.expectEqualSlices(bool, &[_]bool{ true, false }, ant.board[1]);
}

test "langtons ant: boundary and extreme cases" {
    var ant = try LangtonsAnt.init(testing.allocator, 2, 2);
    defer ant.deinit();

    try ant.moveAnt();
    try ant.moveAnt();
    try testing.expectError(LangtonsAntError.AntPositionOutOfBounds, ant.moveAnt());

    var large = try LangtonsAnt.init(testing.allocator, 200, 200);
    defer large.deinit();
    for (0..1000) |_| {
        try large.moveAnt();
    }

    try testing.expectError(LangtonsAntError.InvalidBoardSize, LangtonsAnt.init(testing.allocator, 0, 80));
}
