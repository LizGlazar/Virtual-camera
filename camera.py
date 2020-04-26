# -*- coding: utf-8 -*-
import numpy as np
import pygame

screen_width = 800
screen_height = 600

BLACK = 0, 0, 0
WHITE = 255, 255, 255
BLUE = 0, 0, 255
GREEN = 0, 255, 0
RED = 255, 0, 0
GOLD = 221, 166, 0
SEA_BLUE = 0, 255, 255
PINK = 255, 0, 191
GREY = 128, 128, 128
YELLOW = 255, 255, 0

z_step_delta = 2

drawn = set()


class Polygon:
    def __init__(self, points, z, color):
        self.points = points
        self.z = z
        self.color = color

    def get_points(self):
        return self.points

    def get_z(self):
        return self.z

    def get_color(self):
        return self.color

    def __str__(self):
        return "points=" + str(self.points) + ";color=" + str(self.color) + ";z=" + str(self.z)


class Cuboid:
    def __init__(self, cornerstone, length_x, length_y, length_z, color=None):
        self.cornerstone = np.array(cornerstone, dtype="f")
        self.length_x = length_x
        self.x_vector = np.array([length_x, 0, 0], dtype="f")
        self.length_y = length_y
        self.y_vector = np.array([0, length_y, 0], dtype="f")
        self.length_z = length_z
        self.z_vector = np.array([0, 0, length_z], dtype="f")
        self.color = color

    def points(self):
        # lower base; cornerstone is bottom_left
        bottom_right = self.cornerstone + self.x_vector
        bottom_left_back = self.cornerstone + self.z_vector
        bottom_right_back = bottom_right + self.z_vector

        upper_left = self.cornerstone + self.y_vector
        upper_right = bottom_right + self.y_vector
        upper_left_back = bottom_left_back + self.y_vector
        upper_right_back = bottom_right_back + self.y_vector

        return [self.cornerstone, bottom_right, bottom_right_back, bottom_left_back,
                upper_left, upper_right, upper_right_back, upper_left_back]
        
    def get_color(self):
        return self.color


class KeyEventHandler:
    def __init__(self, key, msg, handler, argProvider=None):
        self.key = key
        self.handler = handler
        self.argProvider = argProvider
        self.msg = msg

    def get_key(self):
        return self.key

    def run_handler(self):
        print(self.msg)
        if self.argProvider is not None:
            self.handler(self.argProvider())
        else:
            self.handler()


class EventHandler:
    def __init__(self, quit_handler, key_handlers):
        self.quit_handler = quit_handler
        self.key_handlers = key_handlers

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            print("Quit")
            self.quit_handler()
        elif event.type == pygame.KEYDOWN:
            if event.key in self.key_handlers:
                self.key_handlers[event.key].run_handler()
            else:
                print("No registered handler for ", event.key)
                self.quit_handler()


class Camera:
    def __init__(self):
        self.reset()

    def move_plane(self, shift):
        next_value = self.plane + shift
        if next_value > 0:
            self.plane += shift

    def zoom_in(self, shift):
        self.move_plane(shift)

    def zoom_out(self, shift):
        self.move_plane(-shift)

    def get_projection_matrix(self):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1 / self.plane, 0]
        ], dtype="f")

    def get_transformation_matrix(self):
        return self.transformation_matrix

    def reset(self):
        self.position = np.array([0, 0, 0], dtype="f")
        self.plane = 50.0
        self.transformation_matrix = np.eye(4)

    def apply_transformation(self, m):
        self.transformation_matrix = np.matmul(m, self.transformation_matrix)

    def z_axis_move(self, z):
        movement_matrix = np.eye(4)
        movement_matrix[2][3] = z
        self.apply_transformation(movement_matrix)

    # the observer stays, the scene moves in the other direction
    def move_forward(self, z):
        self.z_axis_move(-z)

    def move_backward(self, z):
        self.z_axis_move(z)

    def x_axis_move(self, x):
        movement_matrix = np.eye(4)
        movement_matrix[0][3] = x
        self.apply_transformation(movement_matrix)

    def move_left(self, x):
        self.x_axis_move(x)

    def move_right(self, x):
        self.x_axis_move(-x)

    def y_axis_move(self, y):
        movement_matrix = np.eye(4)
        movement_matrix[1][3] = y
        self.apply_transformation(movement_matrix)

    def move_up(self, y):
        self.y_axis_move(-y)

    def move_down(self, y):
        self.y_axis_move(y)

    def y_axis_rotation(self, fi):
        rotation_matrix = np.eye(4, dtype="f")
        rotation_matrix[0][0] = np.cos(fi)
        rotation_matrix[0][2] = np.sin(fi)
        rotation_matrix[2][0] = -rotation_matrix[0][2]
        rotation_matrix[2][2] = rotation_matrix[0][0]
        self.apply_transformation(rotation_matrix)

    def rotate_left(self, fi):
        self.y_axis_rotation(fi)

    def rotate_right(self, fi):
        self.y_axis_rotation(-fi)

    def x_axis_rotation(self, fi):
        rotation_matrix = np.eye(4, dtype="f")
        rotation_matrix[1][1] = np.cos(fi)
        rotation_matrix[1][2] = -np.sin(fi)
        rotation_matrix[2][1] = -rotation_matrix[1][2]
        rotation_matrix[2][2] = rotation_matrix[1][1]
        self.apply_transformation(rotation_matrix)

    def rotate_up(self, fi):
        self.x_axis_rotation(fi)

    def rotate_down(self, fi):
        self.x_axis_rotation(-fi)

    def z_axis_rotation(self, fi):
        rotation_matrix = np.eye(4, dtype="f")
        rotation_matrix[0][0] = np.cos(fi)
        rotation_matrix[1][1] = rotation_matrix[0][0]
        rotation_matrix[1][0] = np.sin(fi)
        rotation_matrix[0][1] = -rotation_matrix[1][0]
        self.apply_transformation(rotation_matrix)

    def tilt_left(self, fi):
        self.z_axis_rotation(fi)

    def tilt_right(self, fi):
        self.z_axis_rotation(-fi)


def increase_dimension(vector):
    new_vector = np.resize(vector, len(vector) + 1)
    new_vector[-1] = 1
    return new_vector


def normalize_with_z_hint(p, d):
    (v, z) = p
    return v * (d / v[2]) if v[2] != 0 else v, z


def calculate_projection_with_z_hint(projection_matrix, points, d):
    positioned = map(lambda p: (np.matmul(projection_matrix, p), p[2]), points)
    normalized = map(lambda v: normalize_with_z_hint(v, d), positioned)
    return list(normalized)


def calculate_projection(projection_matrix, points, d):
    positioned = map(lambda p: np.matmul(projection_matrix, p), points)
    normalized = map(lambda v: v * (d / v[2]) if v[2] != 0 else v, positioned)
    return list(normalized)


# assuming that point is a 4D numpy array
def fit_to_screen(point):
    new_point = np.copy(point)
    new_point[0] += screen_width / 2.
    new_point[1] = (screen_height / 2.) - new_point[1]
    return new_point


# assuming that point is a 4D numpy array
def extract_2d(point):
    return np.resize(point, 2)


def draw_cuboid(screen, projection_matrix, transformation_matrix, d, cuboid):
    points = cuboid.points()
    points_with_dimension = map(increase_dimension, points)
    transformed_points = map(lambda p: np.matmul(transformation_matrix, p),
                             points_with_dimension)
    projected = calculate_projection(projection_matrix, transformed_points, d)

    if cuboid not in drawn:
        drawn.add(cuboid)
        print(projected)
        print()
    on_screen = list(map(extract_2d, map(fit_to_screen, projected)))

    # draw bottom
    pygame.draw.line(screen, BLACK, on_screen[0], on_screen[1])
    pygame.draw.line(screen, BLACK, on_screen[1], on_screen[2])
    pygame.draw.line(screen, BLACK, on_screen[2], on_screen[3])
    pygame.draw.line(screen, BLACK, on_screen[3], on_screen[0])

    # draw top
    pygame.draw.line(screen, BLACK, on_screen[4], on_screen[5])
    pygame.draw.line(screen, BLACK, on_screen[5], on_screen[6])
    pygame.draw.line(screen, BLACK, on_screen[6], on_screen[7])
    pygame.draw.line(screen, BLACK, on_screen[7], on_screen[4])

    # connect top and bottom
    pygame.draw.line(screen, BLACK, on_screen[0], on_screen[4])
    pygame.draw.line(screen, BLACK, on_screen[1], on_screen[5])
    pygame.draw.line(screen, BLACK, on_screen[2], on_screen[6])
    pygame.draw.line(screen, BLACK, on_screen[3], on_screen[7])


# assuming that a scene is just a list of cuboids
def draw_scene(screen, projection_matrix, transformation_matrix, d, scene):
    for c in scene:
        draw_cuboid(screen, projection_matrix, transformation_matrix, d, c)


# points is a list of cuboid points
def transform_points(projection_matrix, transformation_matrix, d, points):
    all_points_with_dimension = map(increase_dimension, points)
    all_points_transformed = map(lambda p: np.matmul(transformation_matrix, p), all_points_with_dimension)
    projected = calculate_projection_with_z_hint(projection_matrix, all_points_transformed, d)
    on_screen = map(lambda p: (extract_2d(fit_to_screen(p[0])), p[1]), projected)
    return list(on_screen)


def transform_cuboid(cuboid, projection_matrix, transformation_matrix, d):
    return transform_points(projection_matrix, transformation_matrix, d, cuboid.points()), cuboid.get_color()


def build_polygon(points_with_z_hint, color):
    points = list(map(lambda p: p[0], points_with_z_hint))
    z_values = list(map(lambda p: p[1], points_with_z_hint))
    z_avg = np.mean(z_values)
    return Polygon(points, z_avg, color)


def build_polygons(points, color):
    return [
            # bottom
            build_polygon([points[0], points[1], points[2], points[3]], color),
            # top
            build_polygon([points[4], points[5], points[6], points[7]], color),
            # front
            build_polygon([points[0], points[1], points[5], points[4]], color),
            # back
            build_polygon([points[3], points[2], points[6], points[7]], color),
            # left
            build_polygon([points[0], points[3], points[7], points[4]], color),
            # right
            build_polygon([points[1], points[2], points[6], points[5]], color)
            ]


def prepare_polygons_for_drawing(projection_matrix, transformation_matrix, d, scene):
    # list of (points, color)
    transformed_cuboids = [transform_cuboid(c, projection_matrix, transformation_matrix, d) for c in scene]
    colored_polygons = [build_polygons(points, color) for points, color in transformed_cuboids]
    import itertools
    polygons = list(itertools.chain(*colored_polygons))
    polygons.sort(key=lambda p: p.get_z(), reverse=True)

    return polygons


# assuming that a scene is just a list of cuboids
def draw_scene_with_painter_algorithm(screen, projection_matrix, transformation_matrix, d, scene):
    polygons = prepare_polygons_for_drawing(projection_matrix, transformation_matrix, d, scene)

    for polygon in polygons:
        pygame.draw.polygon(screen, polygon.get_color(), polygon.get_points())


class State:
    def __init__(self):
        self.is_running = True
        self.draw_lines_only = True
        self.z_step = 2
        self.fi_step = 0.0875  # approximately 5 degress

    def stop(self):
        self.is_running = False

    def should_run(self):
        return self.is_running
    
    def should_draw_lines_only(self):
        return self.draw_lines_only

    def flip_drawing_mode(self):
        self.draw_lines_only = not self.draw_lines_only

    def get_z_step(self):
        return self.z_step

    def get_fi_step(self):
        return self.fi_step

    def increase_speed(self):
        self.z_step = min(self.z_step + z_step_delta, 30)

    def decrease_speed(self):
        self.z_step = max(0, self.z_step - z_step_delta)


if __name__ == "__main__":
    scene = [
        # left side
        Cuboid([-350, -100, 75], 150, 300, 75, BLUE),
        Cuboid([-350, -100, 200], 150, 300, 75, GREEN),
        Cuboid([-350, -100, 300], 150, 300, 75, RED),
        # right side
        Cuboid([275, -100, 75], 150, 300, 75, GOLD),
        Cuboid([275, -100, 200], 150, 300, 75, SEA_BLUE),
        Cuboid([275, -100, 300], 150, 300, 75, GREY)
    ]
    camera = Camera()
    state = State()
    handlers = [KeyEventHandler(pygame.K_z, "Zoom in", camera.zoom_in, state.get_z_step),
                KeyEventHandler(pygame.K_x, "Zoom out", camera.zoom_out, state.get_z_step),
                KeyEventHandler(pygame.K_r, "Reset", camera.reset),
                KeyEventHandler(pygame.K_UP, "Going forward", camera.move_forward, state.get_z_step),
                KeyEventHandler(pygame.K_DOWN, "Going backward", camera.move_backward, state.get_z_step),
                KeyEventHandler(pygame.K_a, "Rotating left", camera.rotate_left, state.get_fi_step),
                KeyEventHandler(pygame.K_s, "Rotating right", camera.rotate_right, state.get_fi_step),
                KeyEventHandler(pygame.K_LEFT, "Going left", camera.move_left, state.get_z_step),
                KeyEventHandler(pygame.K_RIGHT, "Going right", camera.move_right, state.get_z_step),
                KeyEventHandler(pygame.K_d, "Going up", camera.move_up, state.get_z_step),
                KeyEventHandler(pygame.K_c, "Going down", camera.move_down, state.get_z_step),
                KeyEventHandler(pygame.K_f, "Rotating up", camera.rotate_up, state.get_fi_step),
                KeyEventHandler(pygame.K_v, "Rotating down", camera.rotate_down, state.get_fi_step),
                KeyEventHandler(pygame.K_g, "Tilting left", camera.tilt_left, state.get_fi_step),
                KeyEventHandler(pygame.K_b, "Tilting right", camera.tilt_right, state.get_fi_step),
                KeyEventHandler(pygame.K_p, "Changing drawing mode", state.flip_drawing_mode),
                KeyEventHandler(pygame.K_q, "Quit alias", state.stop),
                KeyEventHandler(pygame.K_EQUALS, "Increasing speed", state.increase_speed),
                KeyEventHandler(pygame.K_MINUS, "Decreasing speed", state.decrease_speed)
                ]
    handlers_by_key = {e.get_key(): e for e in handlers}

    screen = pygame.display.set_mode((screen_width, screen_height))
   
    event_handler = EventHandler(lambda: state.stop(), handlers_by_key)

    clock = pygame.time.Clock()
    while state.should_run():
        clock.tick(10)

        event = pygame.event.poll()
        event_handler.handle_event(event)

        screen.fill(WHITE)
        if state.should_draw_lines_only():
            draw_scene(screen, camera.get_projection_matrix(),
                   camera.get_transformation_matrix(), camera.plane, scene)
        else:
            draw_scene_with_painter_algorithm(screen, camera.get_projection_matrix(),
                   camera.get_transformation_matrix(), camera.plane, scene)
        pygame.display.flip()

    pygame.quit()
