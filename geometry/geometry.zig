//! Geometry - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/geometry/geometry.py

const std = @import("std");
const testing = std.testing;

pub const GeometryError = error{
    InvalidAngle,
    InvalidLength,
    InvalidRadius,
    InvalidCuts,
    IndexOutOfBounds,
};

pub const Angle = struct {
    degrees: f64 = 90.0,

    /// Creates an angle in degrees in the inclusive range [0, 360].
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn init(degrees: f64) GeometryError!Angle {
        if (!std.math.isFinite(degrees) or degrees < 0.0 or degrees > 360.0) return error.InvalidAngle;
        return .{ .degrees = degrees };
    }
};

pub const Side = struct {
    length: f64,
    angle: Angle,
    next_side: ?*const Side = null,

    /// Creates a polygon side with positive length.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn init(length: f64, angle: Angle, next_side: ?*const Side) GeometryError!Side {
        if (!std.math.isFinite(length) or length <= 0.0) return error.InvalidLength;
        return .{ .length = length, .angle = angle, .next_side = next_side };
    }
};

pub const Ellipse = struct {
    major_radius: f64,
    minor_radius: f64,

    /// Creates an ellipse with positive radii.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn init(major_radius: f64, minor_radius: f64) GeometryError!Ellipse {
        if (!std.math.isFinite(major_radius) or major_radius <= 0.0) return error.InvalidRadius;
        if (!std.math.isFinite(minor_radius) or minor_radius <= 0.0) return error.InvalidRadius;
        return .{ .major_radius = major_radius, .minor_radius = minor_radius };
    }

    pub fn area(self: Ellipse) f64 {
        return std.math.pi * self.major_radius * self.minor_radius;
    }

    pub fn perimeter(self: Ellipse) f64 {
        return std.math.pi * (self.major_radius + self.minor_radius);
    }
};

pub const Circle = struct {
    radius: f64,

    /// Creates a circle with positive radius.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn init(radius: f64) GeometryError!Circle {
        if (!std.math.isFinite(radius) or radius <= 0.0) return error.InvalidRadius;
        return .{ .radius = radius };
    }

    pub fn area(self: Circle) f64 {
        return std.math.pi * self.radius * self.radius;
    }

    pub fn perimeter(self: Circle) f64 {
        return 2.0 * std.math.pi * self.radius;
    }

    pub fn diameter(self: Circle) f64 {
        return self.radius * 2.0;
    }

    /// Returns the maximum number of parts after `num_cuts` cuts.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn maxParts(self: Circle, num_cuts: f64) GeometryError!f64 {
        _ = self;
        if (!std.math.isFinite(num_cuts) or num_cuts < 0.0) return error.InvalidCuts;
        return (num_cuts + 2.0 + num_cuts * num_cuts) * 0.5;
    }
};

pub const Polygon = struct {
    allocator: std.mem.Allocator,
    sides: std.ArrayListUnmanaged(Side) = .{},

    pub fn init(allocator: std.mem.Allocator) Polygon {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Polygon) void {
        self.sides.deinit(self.allocator);
    }

    /// Adds a side to the polygon.
    /// Time complexity: amortized O(1), Space complexity: O(1)
    pub fn addSide(self: *Polygon, side: Side) !void {
        try self.sides.append(self.allocator, side);
    }

    /// Returns the side at `index`.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn getSide(self: *const Polygon, index: usize) GeometryError!Side {
        if (index >= self.sides.items.len) return error.IndexOutOfBounds;
        return self.sides.items[index];
    }

    /// Replaces the side at `index`.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn setSide(self: *Polygon, index: usize, side: Side) GeometryError!void {
        if (index >= self.sides.items.len) return error.IndexOutOfBounds;
        self.sides.items[index] = side;
    }
};

pub const Rectangle = struct {
    short_side: Side,
    long_side: Side,

    /// Creates a rectangle from its short and long sides.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn init(short_side_length: f64, long_side_length: f64) GeometryError!Rectangle {
        const right_angle = try Angle.init(90.0);
        return .{
            .short_side = try Side.init(short_side_length, right_angle, null),
            .long_side = try Side.init(long_side_length, right_angle, null),
        };
    }

    pub fn perimeter(self: Rectangle) f64 {
        return (self.short_side.length + self.long_side.length) * 2.0;
    }

    pub fn area(self: Rectangle) f64 {
        return self.short_side.length * self.long_side.length;
    }
};

pub const Square = struct {
    rectangle: Rectangle,

    /// Creates a square from one side length.
    /// Time complexity: O(1), Space complexity: O(1)
    pub fn init(side_length: f64) GeometryError!Square {
        return .{ .rectangle = try Rectangle.init(side_length, side_length) };
    }

    pub fn perimeter(self: Square) f64 {
        return self.rectangle.perimeter();
    }

    pub fn area(self: Square) f64 {
        return self.rectangle.area();
    }
};

test "geometry: angle validation" {
    const default_angle = Angle{};
    try testing.expectApproxEqAbs(90.0, default_angle.degrees, 1e-12);
    try testing.expectApproxEqAbs(45.5, (try Angle.init(45.5)).degrees, 1e-12);
    try testing.expectError(error.InvalidAngle, Angle.init(-1.0));
    try testing.expectError(error.InvalidAngle, Angle.init(361.0));
}

test "geometry: side validation and linkage" {
    const angle = try Angle.init(45.6);
    const next = try Side.init(1.0, try Angle.init(2.0), null);
    const side = try Side.init(5.0, angle, &next);
    try testing.expectApproxEqAbs(5.0, side.length, 1e-12);
    try testing.expect(side.next_side != null);
    try testing.expectError(error.InvalidLength, Side.init(-1.0, angle, null));
}

test "geometry: ellipse and circle" {
    const ellipse = try Ellipse.init(5.0, 10.0);
    try testing.expectApproxEqAbs(157.07963267948966, ellipse.area(), 1e-12);
    try testing.expectApproxEqAbs(47.12388980384689, ellipse.perimeter(), 1e-12);

    const circle = try Circle.init(5.0);
    try testing.expectApproxEqAbs(78.53981633974483, circle.area(), 1e-12);
    try testing.expectApproxEqAbs(31.41592653589793, circle.perimeter(), 1e-12);
    try testing.expectApproxEqAbs(10.0, circle.diameter(), 1e-12);
    try testing.expectApproxEqAbs(1.0, try circle.maxParts(0.0), 1e-12);
    try testing.expectApproxEqAbs(29.0, try circle.maxParts(7.0), 1e-12);
    try testing.expectApproxEqAbs(1486.0, try circle.maxParts(54.0), 1e-12);
    try testing.expectApproxEqAbs(265.375, try circle.maxParts(22.5), 1e-12);
    try testing.expectError(error.InvalidCuts, circle.maxParts(-222.0));
}

test "geometry: polygon add get set and bounds" {
    var polygon = Polygon.init(testing.allocator);
    defer polygon.deinit();

    try testing.expectError(error.IndexOutOfBounds, polygon.getSide(0));

    const side5 = try Side.init(5.0, Angle{}, null);
    try polygon.addSide(side5);
    try testing.expectApproxEqAbs(5.0, (try polygon.getSide(0)).length, 1e-12);

    const side10 = try Side.init(10.0, Angle{}, null);
    try polygon.setSide(0, side10);
    try testing.expectApproxEqAbs(10.0, (try polygon.getSide(0)).length, 1e-12);
    try testing.expectError(error.IndexOutOfBounds, polygon.setSide(1, side10));
}

test "geometry: rectangle and square" {
    const rectangle = try Rectangle.init(5.0, 10.0);
    try testing.expectApproxEqAbs(30.0, rectangle.perimeter(), 1e-12);
    try testing.expectApproxEqAbs(50.0, rectangle.area(), 1e-12);
    try testing.expectError(error.InvalidLength, Rectangle.init(-5.0, 10.0));

    const square = try Square.init(5.0);
    try testing.expectApproxEqAbs(20.0, square.perimeter(), 1e-12);
    try testing.expectApproxEqAbs(25.0, square.area(), 1e-12);
}

test "geometry: extreme finite boundaries" {
    const huge = try Circle.init(1e150);
    try testing.expect(std.math.isFinite(huge.area()));
    try testing.expect(std.math.isFinite(huge.perimeter()));

    const tiny_angle = try Angle.init(0.0);
    try testing.expectApproxEqAbs(0.0, tiny_angle.degrees, 1e-12);
}
