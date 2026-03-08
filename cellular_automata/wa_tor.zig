//! Wa-Tor Simulation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/cellular_automata/wa_tor.py

const std = @import("std");
const testing = std.testing;

pub const PREY_INITIAL_COUNT: usize = 30;
pub const PREY_REPRODUCTION_TIME: i32 = 5;

pub const PREDATOR_INITIAL_COUNT: usize = 50;
pub const PREDATOR_INITIAL_ENERGY_VALUE: i32 = 15;
pub const PREDATOR_FOOD_VALUE: i32 = 5;
pub const PREDATOR_REPRODUCTION_TIME: i32 = 20;

pub const MAX_ENTITIES: usize = 500;
pub const DELETE_UNBALANCED_ENTITIES: usize = 50;

pub const WaTorError = error{
    InvalidDimensions,
    OutOfBounds,
    CellOccupied,
    PlanetFull,
    NotPredator,
};

pub const Coords = struct {
    row: usize,
    col: usize,
};

pub const Direction = enum {
    N,
    E,
    S,
    W,
};

pub const Entity = struct {
    prey: bool,
    coords: Coords,
    remaining_reproduction_time: i32,
    energy_value: ?i32,
    alive: bool,

    pub fn init(prey: bool, coords: Coords) Entity {
        return .{
            .prey = prey,
            .coords = coords,
            .remaining_reproduction_time = if (prey) PREY_REPRODUCTION_TIME else PREDATOR_REPRODUCTION_TIME,
            .energy_value = if (prey) null else PREDATOR_INITIAL_ENERGY_VALUE,
            .alive = true,
        };
    }

    pub fn resetReproductionTime(self: *Entity) void {
        self.remaining_reproduction_time = if (self.prey) PREY_REPRODUCTION_TIME else PREDATOR_REPRODUCTION_TIME;
    }
};

pub const WaTor = struct {
    allocator: std.mem.Allocator,
    width: usize,
    height: usize,
    planet: [][]?*Entity,
    rng: std.Random.DefaultPrng,
    all_allocated_entities: std.ArrayListUnmanaged(*Entity),

    pub fn initEmpty(allocator: std.mem.Allocator, width: usize, height: usize, seed: u64) !WaTor {
        if (width == 0 or height == 0) return WaTorError.InvalidDimensions;

        const planet = try allocator.alloc([]?*Entity, height);
        errdefer allocator.free(planet);

        var built_rows: usize = 0;
        errdefer {
            for (0..built_rows) |i| allocator.free(planet[i]);
        }

        for (0..height) |r| {
            planet[r] = try allocator.alloc(?*Entity, width);
            built_rows += 1;
            @memset(planet[r], null);
        }

        return .{
            .allocator = allocator,
            .width = width,
            .height = height,
            .planet = planet,
            .rng = std.Random.DefaultPrng.init(seed),
            .all_allocated_entities = .{},
        };
    }

    pub fn init(allocator: std.mem.Allocator, width: usize, height: usize, seed: u64) !WaTor {
        var world = try WaTor.initEmpty(allocator, width, height, seed);
        errdefer world.deinit();

        for (0..PREY_INITIAL_COUNT) |_| {
            try world.addEntity(true);
        }
        for (0..PREDATOR_INITIAL_COUNT) |_| {
            try world.addEntity(false);
        }
        return world;
    }

    pub fn deinit(self: *WaTor) void {
        for (self.all_allocated_entities.items) |entity| {
            self.allocator.destroy(entity);
        }
        self.all_allocated_entities.deinit(self.allocator);

        for (self.planet) |row| {
            self.allocator.free(row);
        }
        self.allocator.free(self.planet);
    }

    pub fn clearPlanet(self: *WaTor) void {
        for (self.planet) |row| {
            @memset(row, null);
        }
        for (self.all_allocated_entities.items) |entity| {
            entity.alive = false;
        }
    }

    fn randomIndex(self: *WaTor, upper: usize) usize {
        return self.rng.random().intRangeLessThan(usize, 0, upper);
    }

    fn inBounds(self: *const WaTor, row: usize, col: usize) bool {
        return row < self.height and col < self.width;
    }

    fn countEntities(self: *const WaTor) usize {
        var count: usize = 0;
        for (self.planet) |row| {
            for (row) |cell| {
                if (cell != null) count += 1;
            }
        }
        return count;
    }

    fn createEntity(self: *WaTor, prey: bool, coords: Coords) !*Entity {
        const entity = try self.allocator.create(Entity);
        entity.* = Entity.init(prey, coords);
        try self.all_allocated_entities.append(self.allocator, entity);
        return entity;
    }

    pub fn placeEntity(self: *WaTor, prey: bool, row: usize, col: usize) !*Entity {
        if (!self.inBounds(row, col)) return WaTorError.OutOfBounds;
        if (self.planet[row][col] != null) return WaTorError.CellOccupied;

        const entity = try self.createEntity(prey, .{ .row = row, .col = col });
        self.planet[row][col] = entity;
        return entity;
    }

    pub fn addEntity(self: *WaTor, prey: bool) !void {
        if (self.countEntities() >= self.width * self.height) {
            return WaTorError.PlanetFull;
        }

        while (true) {
            const row = self.randomIndex(self.height);
            const col = self.randomIndex(self.width);
            if (self.planet[row][col] == null) {
                _ = try self.placeEntity(prey, row, col);
                return;
            }
        }
    }

    pub fn getEntities(self: *WaTor, allocator: std.mem.Allocator) ![]*Entity {
        var entities = std.ArrayListUnmanaged(*Entity){};
        defer entities.deinit(allocator);

        for (self.planet) |row| {
            for (row) |cell| {
                if (cell) |entity| {
                    try entities.append(allocator, entity);
                }
            }
        }

        return entities.toOwnedSlice(allocator);
    }

    pub fn getSurroundingPrey(self: *WaTor, allocator: std.mem.Allocator, entity: *const Entity) ![]*Entity {
        var prey_entities = std.ArrayListUnmanaged(*Entity){};
        defer prey_entities.deinit(allocator);

        const neighbors = [_]struct { dr: isize, dc: isize }{
            .{ .dr = -1, .dc = 0 },
            .{ .dr = 1, .dc = 0 },
            .{ .dr = 0, .dc = -1 },
            .{ .dr = 0, .dc = 1 },
        };

        for (neighbors) |delta| {
            const nr_signed = @as(isize, @intCast(entity.coords.row)) + delta.dr;
            const nc_signed = @as(isize, @intCast(entity.coords.col)) + delta.dc;
            if (nr_signed < 0 or nc_signed < 0) continue;

            const nr: usize = @intCast(nr_signed);
            const nc: usize = @intCast(nc_signed);
            if (!self.inBounds(nr, nc)) continue;

            if (self.planet[nr][nc]) |adjacent| {
                if (adjacent.prey) {
                    try prey_entities.append(allocator, adjacent);
                }
            }
        }

        return prey_entities.toOwnedSlice(allocator);
    }

    fn shuffledDirections(self: *WaTor) [4]Direction {
        var directions = [_]Direction{ .N, .E, .S, .W };
        var i = directions.len;
        while (i > 1) {
            i -= 1;
            const j = self.randomIndex(i + 1);
            std.mem.swap(Direction, &directions[i], &directions[j]);
        }
        return directions;
    }

    fn destination(self: *const WaTor, coords: Coords, direction: Direction) ?Coords {
        const r: isize = @intCast(coords.row);
        const c: isize = @intCast(coords.col);
        const Delta = struct { dr: isize, dc: isize };

        const d: Delta = switch (direction) {
            .N => .{ .dr = -1, .dc = 0 },
            .S => .{ .dr = 1, .dc = 0 },
            .W => .{ .dr = 0, .dc = -1 },
            .E => .{ .dr = 0, .dc = 1 },
        };

        const nr = r + d.dr;
        const nc = c + d.dc;
        if (nr < 0 or nc < 0) return null;

        const row: usize = @intCast(nr);
        const col: usize = @intCast(nc);
        if (!self.inBounds(row, col)) return null;

        return .{ .row = row, .col = col };
    }

    pub fn moveAndReproduce(self: *WaTor, entity: *Entity, direction_orders: []const Direction) !void {
        if (!entity.alive) return;

        const original = entity.coords;
        var moved = false;

        for (direction_orders) |direction| {
            const next = self.destination(entity.coords, direction) orelse continue;
            if (self.planet[next.row][next.col] == null) {
                self.planet[next.row][next.col] = entity;
                self.planet[original.row][original.col] = null;
                entity.coords = next;
                moved = true;
                break;
            }
        }

        if (moved and entity.remaining_reproduction_time <= 0) {
            if (self.countEntities() < MAX_ENTITIES) {
                _ = try self.placeEntity(entity.prey, original.row, original.col);
                entity.resetReproductionTime();
            }
        } else {
            entity.remaining_reproduction_time -= 1;
        }
    }

    pub fn performPreyActions(self: *WaTor, entity: *Entity, direction_orders: []const Direction) !void {
        try self.moveAndReproduce(entity, direction_orders);
    }

    pub fn performPredatorActions(self: *WaTor, entity: *Entity, occupied_by_prey_coords: ?Coords, direction_orders: []const Direction) !void {
        if (entity.energy_value == null) return WaTorError.NotPredator;

        if (entity.energy_value.? == 0) {
            self.planet[entity.coords.row][entity.coords.col] = null;
            entity.alive = false;
            return;
        }

        if (occupied_by_prey_coords) |target| {
            if (!self.inBounds(target.row, target.col)) return WaTorError.OutOfBounds;

            if (self.planet[target.row][target.col]) |prey| {
                if (prey.prey) {
                    prey.alive = false;
                    self.planet[target.row][target.col] = entity;
                    self.planet[entity.coords.row][entity.coords.col] = null;
                    entity.coords = target;
                    entity.energy_value.? += PREDATOR_FOOD_VALUE;
                }
            }
        } else {
            try self.moveAndReproduce(entity, direction_orders);
        }

        entity.energy_value.? -= 1;
    }

    pub fn balancePredatorsAndPrey(self: *WaTor) !void {
        var entities = try self.getEntities(self.allocator);
        defer self.allocator.free(entities);

        if (entities.len >= MAX_ENTITIES - (MAX_ENTITIES / 10)) {
            // Shuffle entities first, matching Python's randomness.
            var i = entities.len;
            while (i > 1) {
                i -= 1;
                const j = self.randomIndex(i + 1);
                std.mem.swap(*Entity, &entities[i], &entities[j]);
            }

            var prey = std.ArrayListUnmanaged(*Entity){};
            defer prey.deinit(self.allocator);
            var predators = std.ArrayListUnmanaged(*Entity){};
            defer predators.deinit(self.allocator);

            for (entities) |entity| {
                if (entity.prey) {
                    try prey.append(self.allocator, entity);
                } else {
                    try predators.append(self.allocator, entity);
                }
            }

            const purge = if (prey.items.len > predators.items.len) prey.items else predators.items;
            const purge_count = @min(DELETE_UNBALANCED_ENTITIES, purge.len);

            for (purge[0..purge_count]) |entity| {
                if (self.planet[entity.coords.row][entity.coords.col] != null) {
                    self.planet[entity.coords.row][entity.coords.col] = null;
                    entity.alive = false;
                }
            }
        }
    }

    /// Runs Wa-Tor simulation for fixed iterations.
    ///
    /// Time complexity: O(iterations * entities)
    /// Space complexity: O(entities)
    pub fn run(self: *WaTor, iteration_count: usize) !void {
        for (0..iteration_count) |_| {
            var all_entities = try self.getEntities(self.allocator);
            defer self.allocator.free(all_entities);

            var remaining = all_entities.len;
            while (remaining > 0) {
                const pick = self.randomIndex(remaining);
                remaining -= 1;
                std.mem.swap(*Entity, &all_entities[pick], &all_entities[remaining]);
                const entity = all_entities[remaining];

                if (!entity.alive) continue;

                const directions = self.shuffledDirections();
                if (entity.prey) {
                    try self.performPreyActions(entity, &directions);
                } else {
                    const surrounding = try self.getSurroundingPrey(self.allocator, entity);
                    defer self.allocator.free(surrounding);

                    var target: ?Coords = null;
                    if (surrounding.len > 0) {
                        const prey = surrounding[self.randomIndex(surrounding.len)];
                        target = prey.coords;
                    }

                    try self.performPredatorActions(entity, target, &directions);
                }
            }

            try self.balancePredatorsAndPrey();
        }
    }
};

test "wa tor: entity defaults and reset" {
    const prey = Entity.init(true, .{ .row = 0, .col = 0 });
    try testing.expect(prey.prey);
    try testing.expectEqual(@as(i32, PREY_REPRODUCTION_TIME), prey.remaining_reproduction_time);
    try testing.expectEqual(@as(?i32, null), prey.energy_value);

    var predator = Entity.init(false, .{ .row = 2, .col = 1 });
    try testing.expect(!predator.prey);
    try testing.expectEqual(@as(i32, PREDATOR_REPRODUCTION_TIME), predator.remaining_reproduction_time);
    try testing.expectEqual(@as(?i32, PREDATOR_INITIAL_ENERGY_VALUE), predator.energy_value);

    predator.remaining_reproduction_time = 0;
    predator.resetReproductionTime();
    try testing.expectEqual(@as(i32, PREDATOR_REPRODUCTION_TIME), predator.remaining_reproduction_time);
}

test "wa tor: surrounding prey and movement/reproduction" {
    var world = try WaTor.initEmpty(testing.allocator, 3, 3, 1234);
    defer world.deinit();

    _ = try world.placeEntity(true, 0, 1);
    const predator = try world.placeEntity(false, 1, 1);
    _ = try world.placeEntity(true, 2, 1);

    const surrounding = try world.getSurroundingPrey(testing.allocator, predator);
    defer testing.allocator.free(surrounding);
    try testing.expectEqual(@as(usize, 2), surrounding.len);

    world.clearPlanet();
    const prey = try world.placeEntity(true, 1, 1);
    try world.moveAndReproduce(prey, &[_]Direction{.N});

    try testing.expect(world.planet[0][1] != null);
    try testing.expect(world.planet[1][1] == null);
    try testing.expectEqual(@as(i32, PREY_REPRODUCTION_TIME - 1), world.planet[0][1].?.remaining_reproduction_time);

    world.clearPlanet();
    const reproducible_predator = try world.placeEntity(false, 0, 1);
    reproducible_predator.remaining_reproduction_time = 0;
    try world.moveAndReproduce(reproducible_predator, &[_]Direction{.W});

    try testing.expect(world.planet[0][0] != null);
    try testing.expect(world.planet[0][1] != null);
    try testing.expectEqual(@as(i32, PREDATOR_REPRODUCTION_TIME), world.planet[0][0].?.remaining_reproduction_time);
    try testing.expectEqual(@as(i32, PREDATOR_REPRODUCTION_TIME), world.planet[0][1].?.remaining_reproduction_time);
}

test "wa tor: predator action and simulation extreme" {
    var world = try WaTor.initEmpty(testing.allocator, 20, 20, 20260308);
    defer world.deinit();

    const prey = try world.placeEntity(true, 0, 0);
    _ = prey;
    const predator = try world.placeEntity(false, 0, 1);

    try world.performPredatorActions(predator, .{ .row = 0, .col = 0 }, &[_]Direction{});
    try testing.expect(world.planet[0][0] == predator);
    try testing.expect(world.planet[0][1] == null);
    try testing.expectEqual(@as(?i32, 19), predator.energy_value);

    world.clearPlanet();
    for (0..50) |_| try world.addEntity(true);
    for (0..80) |_| try world.addEntity(false);

    try world.run(200);

    const entities = try world.getEntities(testing.allocator);
    defer testing.allocator.free(entities);
    try testing.expect(entities.len <= MAX_ENTITIES);

    for (entities) |entity| {
        try testing.expect(entity.coords.row < world.height);
        try testing.expect(entity.coords.col < world.width);
        if (!entity.prey) {
            try testing.expect(entity.energy_value != null);
            try testing.expect(entity.energy_value.? >= 0);
        }
    }

    var tiny = try WaTor.initEmpty(testing.allocator, 1, 1, 9);
    defer tiny.deinit();
    _ = try tiny.placeEntity(true, 0, 0);
    try testing.expectError(WaTorError.PlanetFull, tiny.addEntity(false));
}
