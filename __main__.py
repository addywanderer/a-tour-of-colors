import pygame
from level import LEVELS, DEFAULT_BLOCK_HEALTH
from colors import COLORS
from guns import GUN_STATS, BULLET_RADII
from os import listdir
from os.path import isfile, join
from random import randint
from math import floor, ceil, sin, pi, sqrt
from collections import Counter
from copy import deepcopy

pygame.init()

CAPTION = "A Tour of Colors"
ICON = "goal"

WIDTH = 1000
HEIGHT = 800
DIMS = [WIDTH, HEIGHT]
FPS = 60

MSPEED = 13  # max ground speed
AGILE = 4  # ability to change direction
JUMP = 20
FRICTION = 0.5
STOP = 1
PERCEPTION = 0.3
SCROLL = [WIDTH // 3, HEIGHT // 3]  # distance from side of screen to scroll x, y
RESP_BUFFER = 0.1  # secs before player goes back to start after dying
BOUNCE_STRENGTH = 1.14  # amount bouncepads bounce
GRAV_COOLDOWN = 30
SEARCH_RADIUS = 100  # radius in blocks to search for valid block placement
# precompute candidate offsets (bx, by) within SEARCH_RADIUS sorted by distance
search_radius_range = range(-SEARCH_RADIUS, SEARCH_RADIUS + 1)
SEARCH_OFFSETS = [
    (bx, by)
    for bx in search_radius_range
    for by in search_radius_range
    if max(abs(bx), abs(by)) <= SEARCH_RADIUS
]
SEARCH_OFFSETS.sort(key=lambda t: (t[0] * t[0] + t[1] * t[1], abs(t[0]) + abs(t[1])))
# coral = (255, 96, 96)
# lime = (196, 255, 14)
BGCOLOR = "random"

CHARACTER = "troll"
PLAYER_SIZE = 32  # size of player sprite
INVINCIBLE = True

# Toolbar / UI for equipping guns and blocks
TOOLBAR_HEIGHT = 64
TOOLBAR_ICON_SIZE = 48
TOOLBAR_PADDING = 16
TOOLBAR_BORDER = 4

# level_num = randint(0, len(LEVELS) - 1)
level_num = 1

# x_vel is velocity to the right
# y_vel is velocity down

# pygame.Surface needs SRCALPHA as 2nd param

PATH = "assets"
TILES = [
    f"bg_tile_lvl{i + 1}.png" for i in range(len(listdir(join(PATH, "background"))))
]

ICON = join("objects", ICON + ".png")

wd = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(CAPTION)
pygame.display.set_icon(pygame.image.load(join(PATH, ICON)))


# ======= IMAGE / ROTATION CACHES & HELPERS =======
IMAGE_CACHE = {}
ROTATED_CACHE = {}


def load_image_cached(path):
    """Load image once and cache Surface."""
    key = path
    if key in IMAGE_CACHE:
        return IMAGE_CACHE[key]
    real_path = path
    try:
        if not isfile(real_path):
            real_path = join(PATH, path)
    except Exception:
        real_path = join(PATH, path)
    surf = pygame.image.load(real_path).convert_alpha()
    IMAGE_CACHE[key] = surf
    return surf


def get_rotated_cached(base_key, base_surf, angle, snap_deg=1):
    """Return rotated surface, caching by (base_key, snapped_angle)."""
    a = int(round(angle / snap_deg) * snap_deg) % 360
    key = (base_key, a)
    surf = ROTATED_CACHE.get(key)
    if surf is None:
        surf = pygame.transform.rotate(base_surf, a)
        ROTATED_CACHE[key] = surf
    return surf


def blit_rotated(rotated_surface, world_center) -> None:
    """Blit a pygame.rotate() surface, keeping its center at world_center."""
    screen_center = (
        floor(world_center[0] - t_offset[0]),
        floor(world_center[1] - t_offset[1]),
    )
    wd.blit(rotated_surface, rotated_surface.get_rect(center=screen_center))


# returns rotation of sprite by angle clockwise for each obj in sprites
def rotate_image(sprites, angle) -> list[pygame.Surface]:
    return [pygame.transform.rotate(sprite, angle) for sprite in sprites]


# returns list of flipped sprite for each obj in sprites (list of surfaces)
def flip_image(sprites) -> list[pygame.Surface]:
    return [pygame.transform.flip(sprite, True, False) for sprite in sprites]


def load_sprite_sheets(
    path, width, height, flip=False
) -> dict[str : list[pygame.Surface]]:
    images = [f for f in listdir(path) if isfile(join(path, f))]
    allsprites = {}
    for image in images:
        spritesheet = pygame.image.load(join(path, image)).convert_alpha()
        sprites = []
        for i in range(spritesheet.get_width() // width):
            surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
            rect = pygame.Rect(i * width, 0, width, height)
            surface.blit(spritesheet, (0, 0), rect)
            sprites.append(surface)
        if flip:
            allsprites[image.replace(".png", "") + "_right"] = sprites
            allsprites[image.replace(".png", "") + "_left"] = flip_image(sprites)
        else:
            allsprites[image.replace(".png", "")] = sprites
    return allsprites


# ====================================


def random_color():
    return tuple([randint(color_range[i][0], color_range[i][1]) for i in range(3)])


def process_levels(level, color):
    wd.fill([randint(0, 255) for _ in range(3)])
    pos = (WIDTH // 2 - 128, HEIGHT // 2 - 32)
    wd.blit(load_image_cached(join(PATH, "load.png")), pos)
    pygame.display.update()

    global color_range
    color_range, looking_for_color = COLORS[0], True
    while looking_for_color:
        option = COLORS[randint(0, len(COLORS) - 1)]
        if color_range != option:
            color_range = option
            looking_for_color = False
    start, bounds, gravity_val = level[0]
    data = [list(start), list(bounds), gravity_val]

    level_with_objs = []
    for obj in level[1:]:
        if obj[1] in ALL_OBJECTS.keys():
            level_with_objs.append(ALL_OBJECTS[obj[1]](*obj))
        else:
            level_with_objs.append(Object(*obj))
    return data, level_with_objs, random_color() if color == "random" else color


# ====================================


def normalize(n):
    if n == 0:
        return 0
    elif n > 0:
        return 1
    return -1


def valid_obj_coll(obj):
    return (
        obj.name != "layer"
        and not (obj.name == "unstable" and obj.count_to_break <= 0)
        and not (obj.name == "item")
        and not (obj.name == "lock" and obj.unlocked)
    )


def collision_quick_test(obj1, obj2):  # checks if collision is impossible
    return (obj1.rect.centerx - obj2.rect.centerx) ** 2 + (
        obj1.rect.centery - obj2.rect.centery
    ) ** 2 > obj1.squared_max_diag_over_four + obj2.squared_max_diag_over_four + obj2.root_max_diag * obj1.root_max_diag / 2


def circle_rect_collides(center, radius, rect) -> bool:
    """Check whether a circle intersects a rectangle using the closest point."""
    cx, cy = center
    closest_x = min(max(cx, rect.x), rect.right)
    closest_y = min(max(cy, rect.y), rect.bottom)
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy <= radius * radius


# ====================================


class Object(pygame.sprite.Sprite):
    def __init__(self, rect, name, angle, health=100, path=None) -> None:
        super().__init__()
        self.float_rect = [float(i) for i in rect]
        self.rect = pygame.Rect(rect)
        self.name = name
        self.health = health
        self.max_health = self.health
        self.angle = angle
        self.path = self.name if path is None else path

        self.squared_max_diag = self.rect.width**2 + self.rect.height**2
        self.squared_max_diag_over_four = self.squared_max_diag / 4
        self.root_max_diag = sqrt(self.squared_max_diag)
        self.root_max_diag_two = sqrt(2 * self.squared_max_diag)
        self.color, self.cooldown = random_color(), 50
        self.bord = tuple(abs(i - randint(0, 35)) for i in self.color)
        if self.name not in ["block", "player", "bullet", "item", "lock", "layer"]:
            self.image = load_image_cached(join(PATH, "objects", self.path + ".png"))
            self.image = rotate_image([self.image], self.angle)[0]
        self.dead = False

    def maybe_die(self) -> None:
        if self.dead:
            return
        if self.health is not None and self.health <= 0:
            self.handle_death()

    def handle_death(self) -> None:
        if self.name == "player":
            player.respawn(data[0])
            return
        self.dead = True
        if self in level:
            level.remove(self)
        if self in onscreen_objects:
            onscreen_objects.remove(self)

    def draw(self) -> None:
        pos = [self.rect.x, self.rect.y]
        wd.blit(self.image, tuple(pos[i] - t_offset[i] for i in [0, 1]))
        health_ratio = max(0, min(1, self.health / self.max_health))
        bar_color = [int(self.color[i] * health_ratio) for i in range(3)]
        overlay = pygame.Surface((self.rect.w - 8, self.rect.h - 8), pygame.SRCALPHA)
        overlay.fill((*bar_color, 150))
        wd.blit(overlay, (self.rect.x - t_offset[0] + 4, self.rect.y - t_offset[1] + 4))
        rect = (
            self.rect.x - t_offset[0] + 4,
            self.rect.y - t_offset[1] + 4,
            self.rect.w - 8,
            self.rect.h - 8,
        )
        pygame.draw.rect(wd, (*bar_color, 160), rect, 2)

    def add_increment(self, increment):
        self.float_rect[0] += increment[0]
        self.float_rect[1] += increment[1]

    def update_rect_from_float(self):
        self.rect.x = int(round(self.float_rect[0]))
        self.rect.y = int(round(self.float_rect[1]))

    def handle_collision(self, collision_targets=None) -> None:
        """
        Generic collision handler for any moving object.
        Performs step-by-step movement with collision detection.
        Requires object to have: float_rect, xvel, yvel, rect, squared_max_diag
        """

        if collision_targets is None:
            collision_targets = onscreen_objects + [player]
        fx, fy = ceil(abs(self.xvel)), ceil(abs(self.yvel))
        max_speed = fx if fx > fy else fy
        if max_speed == 0:
            return None
        increment = [self.xvel / max_speed, self.yvel / max_speed]
        for _ in range(max_speed):
            self.add_increment(increment)
            self.update_rect_from_float()
            for obj in collision_targets:
                if obj is self:
                    continue
                if not valid_obj_coll(obj):
                    continue
                if collision_quick_test(self, obj):
                    continue
                if not obj.rect.colliderect(self.rect):
                    continue
                self.add_increment([-incr for incr in increment])
                self.update_rect_from_float()
                if hasattr(self, "fallcount") and self.yvel > 0:
                    self.fallcount = 0
                self.xvel = 0
                self.yvel = 0
                return None


class Player(Object):
    def __init__(self, size) -> None:
        self.image = load_image_cached(join(PATH, "characters", CHARACTER + ".png"))
        super().__init__(
            pygame.Rect(*data[0], size, size),
            "player",
            0,
            path=None,
            health=1,
        )
        self.size = size
        self.images = [flip_image([self.image])[0], self.image]
        self.xvel = self.yvel = self.stationary_xvel = self.stationary_yvel = 0
        self.direction, self.walking, self.fallcount = 1, False, 0
        self.inventory, self.gravity_switch, self.friction = [], 0, FRICTION
        self.mspeed, self.agile, self.stop = MSPEED, AGILE, STOP
        self.hit_count = self.loaded = 0
        self.collide, self.last_block_on = [None] * 4, None
        self.block_spawn_timer = 0.0
        self.angle = 0
        self.short_jump = False
        self.bullets = []
        self.toolbar = [
            [
                GUN_STATS["debug_gun"],
                GUN_STATS["pistol"],
                GUN_STATS["pistol"],
                GUN_STATS["pistol"],
                GUN_STATS["pistol"],
            ],
            [
                [[6, 7], "block", 0, None, DEFAULT_BLOCK_HEALTH[4]],
                [[64, 4], "block", 0, None, DEFAULT_BLOCK_HEALTH[8]],
                [[4, 64], "block", 0, None, DEFAULT_BLOCK_HEALTH[16]],
                [[64, 64], "bouncepad", 90, None, DEFAULT_BLOCK_HEALTH["bouncepad"]],
                [[64, 64], "ice", 0, None, DEFAULT_BLOCK_HEALTH["ice"]],
            ],
        ]
        self.selected = [self.toolbar[0][0], self.toolbar[1][0]]
        self.select_num = [0, 0]
        self.update_sprite()
        self.respawn(data[0])
        self.update_gun()

    def update_sprite(self) -> None:
        direction = (normalize(self.xvel) + 1) // 2
        if direction:
            self.image = self.images[direction]
        self.update()

    def update(self) -> None:
        self.rect = self.image.get_rect(
            topleft=tuple([round(self.float_rect[i]) for i in [0, 1]])
        )

    def update_aim(self) -> None:
        mouse_pos_diffs = [
            (mouse[i] + t_offset[i] - self.float_rect[i] - self.float_rect[2 + i] / 2)
            for i in [0, 1]
        ]
        mouse_pos_diff_mag = sqrt(mouse_pos_diffs[0] ** 2 + mouse_pos_diffs[1] ** 2)

        self.angle = -pygame.Vector2(*mouse_pos_diffs).as_polar()[1]
        self.aim_dir = [mouse_pos_diffs[i] / mouse_pos_diff_mag for i in [0, 1]]

    def update_gun(self) -> None:
        self.update_aim()

        if pygame.mouse.get_pressed(num_buttons=3)[0] and self.loaded >= 0:
            rect = (self.rect.centerx - 32, self.rect.centery - 32, 64, 64)
            bullet = Bullet(self, rect, self.selected[0]["bullet_image"])
            self.bullets.append(bullet)
            recoil = self.selected[0]["recoil"]
            if recoil:
                self.xvel -= self.aim_dir[0] * recoil
                self.yvel -= self.aim_dir[1] * recoil
            self.loaded = -self.selected[0]["reload"]
        elif self.loaded < 0:
            self.loaded += 1

        for bullet in self.bullets[:]:
            if bullet.dead:
                self.bullets.remove(bullet)
            else:
                bullet.loop(self)

        # use cached image + cached rotation to avoid repeated loads/transforms
        gun_img_path = join(PATH, "guns", self.selected[0].get("gun_image") + ".png")
        base = load_image_cached(gun_img_path)
        if base.get_size() != (self.size, self.size):
            base = pygame.transform.scale(base, (self.size, self.size))
        rotated = get_rotated_cached(gun_img_path, base, self.angle)
        if getattr(self, "_gun_img_id", None) != id(rotated):
            self.rotated_gun = rotated
            self._gun_img_id = id(rotated)

    def draw_toolbar(self) -> None:
        """Draw toolbar with guns and block-size selectors at top-left."""
        icon_dist = TOOLBAR_ICON_SIZE + TOOLBAR_PADDING
        x = WIDTH // 2 - (len(self.toolbar[0]) * (icon_dist) + TOOLBAR_PADDING)
        y = HEIGHT - TOOLBAR_HEIGHT + (TOOLBAR_HEIGHT - TOOLBAR_ICON_SIZE) // 2

        for j in [0, 1]:
            for i, item in enumerate(self.toolbar[j]):
                box_rect = pygame.Rect(x, y, TOOLBAR_ICON_SIZE, TOOLBAR_ICON_SIZE)
                if j == 1 and item[1] == "block":
                    max_size = max(max(*item[0]) for item in self.toolbar[1])
                    inner_dims = [int((item[0][i] / max_size) * (TOOLBAR_ICON_SIZE - 8)) for i in [0, 1]]
                    inner_pos = [box_rect[i] + (TOOLBAR_ICON_SIZE - inner_dims[i]) // 2 for i in [0, 1]]
                    pygame.draw.rect(wd, (0, 0, 0), box_rect)
                    pygame.draw.rect(wd, (200, 200, 200), (*inner_pos, *inner_dims))
                    x += icon_dist
                    continue
                if j == 0:
                    path = join(PATH, "guns", item["gun_image"] + ".png")
                else: 
                    path = join(PATH, "objects", item[1] + ".png")
                img = load_image_cached(path)
                surf = pygame.Surface([TOOLBAR_ICON_SIZE] * 2, pygame.SRCALPHA)
                surf.blit(img, (0, 0))
                wd.blit(surf, box_rect.topleft)
                color = (255, 255, 0) if i == self.select_num[j] else (0, 0, 0)
                pygame.draw.rect(wd, color, box_rect, TOOLBAR_BORDER)
                x += icon_dist
            x += TOOLBAR_PADDING

    def draw(self) -> None:
        for bullet in self.bullets:
            if not bullet.dead:
                blit_rotated(bullet.rotated_image, bullet.rect.center)
        image_pos = [self.float_rect[i] - t_offset[i] for i in [0, 1]]
        wd.blit(self.image, [floor(image_pos[i]) for i in [0, 1]])
        blit_rotated(self.rotated_gun, self.rect.center)
        self.draw_toolbar()

    def respawn(self, start) -> list[float, float]:
        global gravity, level, color, data
        self.float_rect[:2], self.collide, self.fallcount = start, [None] * 4, 1
        self.xvel, self.yvel = 0, 0
        gravity, self.last_block_on, self.inventory = data[2], None, []
        self.bullets, self.loaded = [], 0
        data, level, color = process_levels(LEVELS[level_num - 1], BGCOLOR)
        self.update()

    def loop(self) -> list[float, float]:
        self.adjust_speed()
        self.collision()
        self.end_collision()
        self.update_sprite()
        self.update_gun()

        if self.gravity_switch > 0:
            self.gravity_switch -= 1

        if self.hit_count:
            self.hit_count += 1
        if self.hit_count > FPS * RESP_BUFFER and not INVINCIBLE:
            self.hit_count = 0
            self.respawn(data[0])
        elif not (data[1][0] <= self.float_rect[1] <= data[1][1]):
            self.respawn(data[0])

    def adjust_speed(self) -> None:
        if not self.walking:
            if self.xvel != 0:
                self.xvel *= self.friction
            if -self.stop <= self.xvel <= self.stop:
                self.xvel = 0
        self.walking = False

        self.fallcount += 1
        if not ((self.collide[2] and gravity < 0) or (self.collide[3] and gravity > 0)):
            self.yvel += (self.fallcount / FPS) * gravity  # gravity

    def add_incr(self, x, y) -> None:
        self.float_rect[0] += x
        self.float_rect[1] += y
        self.update()

    def try_direction(self, direction, obj):
        self.add_incr(direction[0], direction[1])
        collided = self.has_collided(obj)
        self.add_incr(-direction[0], -direction[1])
        return obj if collided else None

    def has_collided(self, obj) -> bool:
        colliding = self.rect.colliderect(obj.rect)
        if colliding and obj.name == "item":
            self.inventory.append(obj)
            obj.collected = True
        return colliding and valid_obj_coll(obj)

    def collision(self) -> None:
        self.xvel_before_coll, self.yvel_before_coll = self.xvel, self.yvel
        axes = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for i, axis in enumerate(axes):
            iterator = (
                obj for obj in onscreen_objects if self.try_direction(axis, obj)
            )
            self.collide[i] = next(iterator, None)
        self.xvel += self.stationary_xvel
        self.yvel += self.stationary_yvel
        fx, fy = ceil(abs(self.xvel)), ceil(abs(self.yvel))
        max_speed = fx if fx > fy else fy
        if max_speed == 0:
            return None
        increment = [self.xvel / max_speed, self.yvel / max_speed]
        direction = [abs(i) / i if i else 0 for i in [self.xvel, self.yvel]]
        for _ in range(max_speed):
            self.add_incr(increment[0], increment[1])
            for obj in onscreen_objects:
                if collision_quick_test(self, obj):
                    continue
                if not self.has_collided(obj):
                    continue
                self.add_incr(-increment[0], -increment[1])
                coll = [self.try_direction(i, obj) for i in axes]
                self.collide = [
                    obj if (coll[i] and (direction[i // 2] in axes[i % 2])) else None
                    for i in range(4)
                ]
                return None

    def end_collision(self) -> None:
        self.xvel -= self.stationary_xvel
        self.yvel -= self.stationary_yvel
        self.float_rect = [round(self.float_rect[i]) for i in range(4)]
        angles, zero_set = [[270, 90], [180, 0]], [self.xvel, self.yvel]
        for i in range(4):
            if self.collide[i] and not (
                self.collide[i].name == "bouncepad"
                and self.collide[i].angle == angles[i // 2][i % 2]
            ):
                zero_set[i // 2] = 0
        self.xvel, self.yvel = zero_set
        if (self.collide[2] and gravity < 0) or (self.collide[3] and gravity > 0):
            self.fallcount = 0


class Bullet(Object):
    def __init__(self, player, rect, path) -> None:
        super().__init__(rect, "bullet", player.angle)
        self.path = path
        self.speed = player.selected[0]["bullet_speed"]
        self.damage = player.selected[0].get("damage", 0)
        self.radius = BULLET_RADII.get(path, BULLET_RADII.get("ammo", 8))
        self.movement = [
            player.aim_dir[0] * self.speed,
            player.aim_dir[1] * self.speed,
        ]
        # use cached load/rotation for bullet image
        base_path = (
            join("bullets", self.path + ".png")
            if isinstance(self.path, str)
            else self.path
        )
        base_surf = (
            load_image_cached(base_path) if isinstance(base_path, str) else base_path
        )
        # scale to bullet rect size if necessary
        size_w, size_h = int(self.float_rect[2]), int(self.float_rect[3])
        if base_surf.get_size() != (size_w, size_h):
            base_surf = pygame.transform.scale(base_surf, (size_w, size_h))
        rotated = get_rotated_cached(base_path, base_surf, self.angle)
        if getattr(self, "_rot_img_id", None) != id(rotated):
            self.rotated_image = rotated
            self._rot_img_id = id(rotated)
        self.loop(player)

    def loop(self, player) -> None:
        self.float_rect[0] += self.movement[0]
        self.float_rect[1] += self.movement[1]
        self.rect.x, self.rect.y = round(self.float_rect[0]), round(self.float_rect[1])

        # bounding-box prefilter before overlap
        if (
            self.float_rect[0] < player.float_rect[0] - WIDTH
            or self.float_rect[1] < player.float_rect[1] - HEIGHT
            or self.float_rect[0] > player.float_rect[0] + WIDTH
            or self.float_rect[1] > player.float_rect[1] + HEIGHT
        ):
            self.dead = True
            return None

        candidates = [
            o for o in level if hasattr(o, "rect") and o.rect.colliderect(self.rect)
        ]
        for object in candidates:
            if not circle_rect_collides(self.rect.center, self.radius, object.rect):
                continue
            if object.health > 0:
                object.health -= self.damage
                if object.health <= 0:
                    object.handle_death()
            self.dead = True
            break


class Block(Object):
    def __init__(self, rect, name, angle, health=100, path=None):
        super().__init__(rect, name, angle, health=100, path=None)

    def draw(self):
        pos = [self.rect.x, self.rect.y]
        if self.cooldown <= 0:
            self.cooldown = 50
            self.color = random_color()
            self.bord = tuple(abs(i - 10) for i in self.color)
        pygame.draw.rect(
            wd,
            self.bord,
            [pos[i] - t_offset[i] for i in [0, 1]] + [self.rect.w, self.rect.h],
        )
        pygame.draw.rect(
            wd,
            self.color,
            [pos[i] - t_offset[i] + 4 for i in [0, 1]]
            + [self.rect.w - 8, self.rect.h - 8],
        )

        # draw a persistent health bar (no regeneration)
        max_h = getattr(self, "max_health", 100) or 100
        health_ratio = max(0, min(1, self.health / max_h))
        bar_total_w = self.rect.w - 8
        bar_w = floor(health_ratio * bar_total_w)
        bar_x = self.rect.x - t_offset[0] + 4
        bar_y = self.rect.y - t_offset[1] + 4
        bar_color = [int(self.color[i] * health_ratio) for i in range(3)]
        if bar_w > 0:
            pygame.draw.rect(wd, bar_color, (bar_x, bar_y, bar_w, self.rect.h - 8))


class Bouncepad(Object):
    def __init__(self, rect, name, angle, health=100, path=None) -> None:
        super().__init__(rect, name, angle, health=100, path=None)
        self.path = self.name
        sprite_sheet = load_sprite_sheets(join(PATH, "objects"), rect[2], rect[3])
        self.sprites = rotate_image(sprite_sheet[self.path], angle)
        if health is not None:
            self.health = health
            self.max_health = health
        self.anim, self.bounced, self.angle = 0, 0, angle

    def loop(self) -> None:
        if 0 < self.bounced <= 2 * len(self.sprites):
            self.bounced += 1
            self.image = self.sprites[(self.anim // 2) % len(self.sprites)]
            self.anim = 0 if self.anim // 2 > len(self.sprites) else self.anim + 1
        else:
            self.anim, self.bounced = 0, 0
            self.image = self.sprites[-1]


class Unstable(Object):
    def __init__(self, rect, name, angle, health=100, path=None) -> None:
        super().__init__(rect, name, angle, health=100, path=None)
        count_to_break = 60
        respawn_buffer = 120
        self.count_to_break, self.break_len, self.respawn_buffer = (
            count_to_break,
            count_to_break,
            respawn_buffer,
        )
        self.ghost_image = self.image.copy()
        self.ghost_image.set_alpha(100)

    def draw(self) -> None:
        img = self.image if self.count_to_break > 0 else self.ghost_image
        wd.blit(img, tuple(self.rect[i] - t_offset[i] for i in range(2)))
        w = floor(max(0, self.count_to_break) / self.break_len * self.rect.w)
        pygame.draw.rect(
            wd,
            (0, 0, 0),
            (
                self.rect.x - t_offset[0] + self.rect.w - w + 4,
                self.rect.y - t_offset[1] + 4,
                w - 8,
                self.rect.h - 8,
            ),
        )

    def loop(self):
        if self.count_to_break <= 0:
            self.count_to_break -= 1
            if self.count_to_break <= -self.respawn_buffer and not (
                self.rect.colliderect(player.rect)
                or any(
                    o.name == "moving" and self.rect.colliderect(o.rect)
                    for o in level
                    if o is not self
                )
            ):
                self.count_to_break = self.break_len


class Moving(Object):
    def __init__(self, rect, name, angle, health=100, path=None):
        super().__init__(rect, name, angle, health=100, path=None)
        move_axis = 0
        move_range = 10 * 64
        speed = 4
        self.angle = move_axis * 90
        self.sprite_sheet = load_sprite_sheets(join(PATH, "objects"), rect[2], rect[3])
        self.sprites = rotate_image(self.sprite_sheet[self.path], self.angle)
        wire = [64, 0] if move_axis == 0 else [0, 64]
        args = ["layer", "wire", self.angle]
        self.wires = [
            Object([rect[0] + wire[0] * i, rect[1] + wire[1] * i, 64, 64], *args)
            for i in range((move_range // 64) + 1)
        ]
        self.anim, self.count, self.anim_delay = 0, 0, 1
        self.change_direction = 0
        self.start_pos = [rect[0], rect[1]]
        self.move_axis = move_axis  # 0 for x, 1 for y
        self.move_range = move_range
        self.speed = speed

    def loop(self):
        self.rect[self.move_axis] += self.speed
        self.change_direction %= 2
        condition = (
            not self.start_pos[self.move_axis]
            <= self.rect[self.move_axis]
            <= self.start_pos[self.move_axis] + self.move_range
        )
        if condition:
            self.speed *= -1
        if condition or self.change_direction:
            self.change_direction += 1
        self.count = (self.count - abs(self.speed) // self.speed) % (
            self.anim_delay * len(self.sprites)
        )
        self.anim = (self.count // self.anim_delay) % len(self.sprites)
        self.image = self.sprites[self.anim]

        if self.change_direction:
            return None

        colliding = self.rect.colliderect(player.rect)
        move_direction = abs(self.speed) // self.speed
        axis = self.move_axis == 1
        while colliding:
            colliding = self.rect.colliderect(player.rect)
            player.float_rect[axis] += move_direction
            player.update()


class Item(Object):
    def __init__(self, rect, name, angle, key_name, health=100, path=None):
        super().__init__(rect, name, angle, path=key_name, health=health)
        self.image = load_image_cached(join(PATH, "items", self.path + ".png"))
        self.anim_delay = 30
        self.item_name, self.anim_delay, self.count = self.path, self.anim_delay, 0
        self.collected, self.orig_y = False, self.rect.y

    def draw(self):
        if not self.collected:
            super().draw()

    def loop(self):
        self.rect.y = self.orig_y + 2 * sin((tick % 60) / 30 * pi)


class Lock(Object):
    def __init__(self, rect, name, angle, key_name, health=100, path=None):
        super().__init__(rect, name, angle, health=100, path=None)
        self.images = [
            load_image_cached(join(PATH, "objects", p + ".png"))
            for p in ["lock_" + key_name[-1], "open_" + key_name[-1]]
        ]
        self.image, self.key_name, self.unlocked = self.images[0], key_name, False

    def loop(self):
        self.unlocked = any(i.item_name == self.key_name for i in player.inventory)
        self.image = self.images[self.unlocked]


def resolve_enemy_spawn(target_center, size=64) -> tuple[int, int] | None:
    anchor = pygame.Rect(
        target_center[0] - size // 2,
        target_center[1] - size // 2,
        size,
        size,
    )
    if not block_placement_blocked(anchor):
        return anchor.topleft

    for bx, by in SEARCH_OFFSETS:
        candidate = anchor.move(bx * size, by * size)
        if not block_placement_blocked(candidate):
            return candidate.topleft
    return None


class Enemy(Object):
    def __init__(self, rect, name, angle, health=100, path=None):
        super().__init__(rect, name, angle, health=100, path=None)

    def can_see_player(enemy, player) -> bool:
        start = enemy.rect.center
        end = player.rect.center
        for obj in level:
            if obj.name != "enemy" and obj.rect.clipline(start, end):
                return False
        return True

    def update(self) -> None:
        """Update rect position from float_rect"""
        self.rect.x = int(round(self.float_rect[0]))
        self.rect.y = int(round(self.float_rect[1]))


class FlyingEnemy(Enemy):
    def __init__(self, rect, name, angle, health=100, path=None):
        super().__init__(rect, name, angle, health=100, path=None)
        self.speed = 3

    def loop(self):
        if not self.can_see_player(self, player):
            return None

        # move toward the player in small steps and stop on collision
        dx = player.rect.centerx - self.rect.centerx
        dy = player.rect.centery - self.rect.centery
        mag = sqrt(dx * dx + dy * dy) or 1
        move_x = (dx / mag) * self.speed
        move_y = (dy / mag) * self.speed

        steps = max(1, ceil(max(abs(move_x), abs(move_y))))
        step_x = move_x / steps
        step_y = move_y / steps
        stopped = False
        for _ in range(steps):
            self.float_rect[0] += step_x
            self.float_rect[1] += step_y
            self.update()

            # check collisions with onscreen_objects (cheap rect check first)
            for o in onscreen_objects:
                if o is self:
                    continue
                if not getattr(o, "rect", None) or not o.rect.colliderect(self.rect):
                    continue
                # skip non-solid objects
                if not valid_obj_coll(o):
                    continue
                # collided: revert last micro-step and stop moving
                self.float_rect[0] -= step_x
                self.float_rect[1] -= step_y
                self.update()
                stopped = True
                break
            if stopped:
                break


class FallingEnemy(Enemy):
    def __init__(self, rect, name, angle, health=100, path=None):
        super().__init__(rect, name, angle, health=100, path=None)
        self.speed = 20
        self.jump_power = 20
        self.jump_cooldown = 30
        # Physics
        self.yvel = 0
        self.xvel = 0
        self.fallcount = 0

    def loop(self):
        if not (data[1][0] <= self.float_rect[1] <= data[1][1]):
            self.handle_death()
            return None

        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
        self.fallcount += 1
        self.yvel += (self.fallcount / FPS) * gravity

        self.xvel *= 0.9

        self.handle_collision()
        self.update()

        dx = player.rect.centerx - self.rect.centerx
        dy = player.rect.centery - self.rect.centery

        if (
            self.fallcount == 0
            and self.can_see_player(self, player)
            and self.jump_cooldown <= 0
        ):
            mag = sqrt(dx * dx + dy * dy) or 1
            self.xvel = (dx / mag) * self.speed
            self.yvel = -self.jump_power
            self.jump_cooldown = 30


def obj_loop() -> None:
    for obj in list(level):
        if obj.dead:
            obj.handle_death()
        obj.maybe_die()
        if obj.dead:
            continue
        if hasattr(obj, "loop"):
            obj.loop()


def obj_interaction():
    global data, level_num, level, color
    player.short_jump = False
    player.friction, last = FRICTION, player.last_block_on
    # print([coll.rect for coll in player.collide if coll])
    if data[2] > 0:
        if player.collide[3]:
            last = player.collide[3]
    elif player.collide[2]:
        last = player.collide[2]
    if not (
        last
        and last.name == "moving"
        and not last.change_direction
        and last in player.collide
    ):
        player.stationary_xvel = player.stationary_yvel = 0
    non_none_objs = [o for o in player.collide if o]
    if non_none_objs:
        obj, count = Counter(non_none_objs).most_common(1)[0]
        if count >= 3 and not (obj.name == "moving" and obj.change_direction):
            return player.respawn(data[0])

    for obj in player.collide:
        if not obj:
            continue
        if obj.name == "block":
            obj.cooldown -= 1
        elif obj.name == "goal":
            level_num = (level_num + 1) % len(LEVELS)
            data, level, color = process_levels(LEVELS[level_num - 1], BGCOLOR)
            gravity = data[2]
            player.respawn(data[0])
            return None
        elif obj.name in ["spike", "enemy", "falling_enemy"]:
            player.hit_count += 1
        elif obj.name == "gravity" and player.gravity_switch == 0:
            gravity, player.gravity_switch = gravity * -1, GRAV_COOLDOWN
        elif obj.name == "checkpoint":
            data[0] = [
                obj.rect.centerx - player.size // 2,
                obj.rect.centery - player.size // 2 - 64,
            ]
        elif obj.name == "bouncepad":
            angles, b = [3, 1, 2, 0], [BOUNCE_STRENGTH, -BOUNCE_STRENGTH] * 2
            for i in range(4):
                if obj.angle == angles[i] * 90 and player.collide[i] == obj:
                    obj.bounced, obj.anim = 1, 0
                    if i // 2 == 0:
                        player.xvel = b[i] * (abs(player.xvel)) ** 0.9 * 8
                    else:
                        player.yvel = b[i] * (abs(player.yvel)) ** 0.9
        elif obj.name == "sticky":
            player.xvel, player.yvel = 0, 0
            if obj in player.collide[2:]:
                player.short_jump = True
        elif obj.name == "unstable":
            obj.count_to_break -= 1
        elif obj.name == "ice":
            player.friction = 0.99
            player.mspeed, player.agile = MSPEED * 1.5, AGILE * 0.5
            player.stop = STOP * 0.5
        elif obj.name == "moving":
            if obj.move_axis == 0:
                player.stationary_xvel = obj.speed
            else:
                player.stationary_yvel = obj.speed
        elif obj.name == "item":
            obj.collected = True
            player.inventory.append(obj)


def handle_inputs():
    global gravity, data, level, level_num, color
    keys = pygame.key.get_pressed()

    m_agile = player.mspeed - player.agile
    if keys[pygame.K_l]:
        level_num = (level_num + 1) % len(LEVELS)
        data, level, color = process_levels(LEVELS[level_num - 1], BGCOLOR)
        gravity = data[2]
        player.respawn(data[0])
        return None
    if keys[pygame.K_r]:
        player.respawn(data[0])
    if keys[pygame.K_c]:
        player.inventory = []
    if player.block_spawn_timer > 0:
        player.block_spawn_timer = max(
            0.0, player.block_spawn_timer - clock.get_time() / 1000.0
        )
    if pygame.mouse.get_pressed(num_buttons=3)[2]:
        if player.block_spawn_timer <= 0:
            parameters = deepcopy(player.selected[1])
            parameters[0] = (
                list(resolve_block_placement(mouse, block_size=parameters[0]))
                + parameters[0]
            )
            level.append((ALL_OBJECTS[parameters[1]] if parameters[1] in ALL_OBJECTS \
                           else Object)(*parameters))
            player.block_spawn_timer = (
                player.selected[1][0][0] * player.selected[1][0][1] / 4096.0)
    if keys[pygame.K_p]:
        pygame.display.toggle_fullscreen()
        pygame.display.set_icon(pygame.image.load(join(PATH, ICON)))
        pygame.display.update()
    if (keys[pygame.K_LEFT] or keys[pygame.K_a]) and (not player.collide[0]):
        player.walking = True
        player.xvel = (
            -player.mspeed if -player.xvel >= m_agile else player.xvel - player.agile
        )
    elif (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and (not player.collide[1]):
        player.walking = True
        player.xvel = (
            player.mspeed if player.xvel >= m_agile else player.xvel + player.agile
        )
    if (keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]) and (
        player.fallcount == 0
    ):
        player.yvel = -JUMP * (gravity // abs(gravity))
        if player.short_jump:
            player.yvel *= 0.2
        player.fallcount = 0

    key_map = {
        pygame.K_1: (0, 0),
        pygame.K_2: (0, 1),
        pygame.K_3: (0, 2),
        pygame.K_4: (0, 3),
        pygame.K_5: (0, 4),
        pygame.K_6: (1, 0),
        pygame.K_7: (1, 1),
        pygame.K_8: (1, 2),
        pygame.K_9: (1, 3),
        pygame.K_0: (1, 4),
    }
    for key, (toolbar_index, slot_index) in key_map.items():
        if keys[key]:
            player.selected[toolbar_index] = player.toolbar[toolbar_index][slot_index]
            player.select_num[toolbar_index] = slot_index
            break

    # spawn an enemy near the player when 'y' is pressed
    if keys[pygame.K_y]:
        spawn_point = resolve_enemy_spawn(
            (player.rect.centerx + 500, player.rect.centery)
        )
        if spawn_point:
            level.append(
                FlyingEnemy([spawn_point[0], spawn_point[1], 32, 32], "enemy", 0)
            )

    # spawn a falling enemy near the player when 'u' is pressed
    if keys[pygame.K_u]:
        spawn_point = resolve_enemy_spawn(
            (player.rect.centerx + 500, player.rect.centery)
        )
        if spawn_point:
            level.append(
                FallingEnemy([spawn_point[0], spawn_point[1], 32, 32], "enemy", 0)
            )


def block_rect_at_mouse(mouse_screen, size=[64, 64]) -> pygame.Rect:
    return pygame.Rect(
        mouse_screen[0] + t_offset[0] - size[0] // 2,
        mouse_screen[1] + t_offset[1] - size[1] // 2,
        size[0],
        size[1],
    )


def block_placement_blocked(rect) -> bool:
    return any(rect.colliderect(o.rect) for o in level) or rect.colliderect(player.rect)


def resolve_block_placement(
    mouse_screen, search_radius=SEARCH_RADIUS, block_size=[64, 64]
) -> tuple[int, int] | None:
    anchor = block_rect_at_mouse(mouse_screen, block_size)
    if not block_placement_blocked(anchor):
        return anchor.topleft

    def inch_toward_cursor(start_rect: pygame.Rect) -> tuple[int, int]:
        """Move a valid rect toward cursor 1 pixel at a time until contact."""
        candidate = start_rect.copy()
        cand_center = [candidate.centerx, candidate.centery]
        target_anchors = [anchor.centerx, anchor.centery]
        axes_done = [False] * 2
        while not (axes_done[0] and axes_done[1]):
            for i in [0, 1]:
                if not axes_done[i]:
                    disp = target_anchors[i] - cand_center[i]
                    if disp == 0:
                        axes_done[i] = True
                        continue
                    movement = ((disp > 0) - (disp < 0), 0) if i == 0 \
                        else (0, (disp > 0) - (disp < 0))
                    trial = candidate.move(*movement)
                    if block_placement_blocked(trial):
                        axes_done[i] = True
                    else:
                        candidate = trial
        return candidate.topleft

    # test precomputed candidates starting from closest and return the
    # first placement that can be inched toward the cursor
    max_block_side = max(*block_size)
    normalizer = [block_size[i] / max_block_side for i in [0, 1]]
    for (bx, by) in SEARCH_OFFSETS:
        if max(abs(bx), abs(by)) <= search_radius:
            candidate = anchor.move(int(bx * normalizer[1] * block_size[0]), \
                                    int(by * normalizer[0] * block_size[1]))
            if block_placement_blocked(candidate):
                continue
            return inch_toward_cursor(candidate)

    return None


def draw_aim_overlay() -> None:
    player_screen = (
        floor(player.rect.centerx - t_offset[0]),
        floor(player.rect.centery - t_offset[1]),
    )
    mouse_screen = (floor(mouse[0]), floor(mouse[1]))
    pygame.draw.line(wd, (255, 255, 255), player_screen, mouse_screen, 1)
    block_size = player.selected[1][0]
    resolved = resolve_block_placement(mouse, block_size=block_size)
    condition = resolved and resolved != block_rect_at_mouse(mouse, block_size).topleft
    naive_color = (255, 0, 0) if condition else (100, 255, 100)

    rect = (*[mouse[i] - block_size[i] // 2 for i in [0, 1]], *block_size)
    pygame.draw.rect(wd, naive_color, rect, 4)
    if resolved and condition:
        rect = (*[resolved[i] - t_offset[i] for i in [0, 1]], *block_size)
        pygame.draw.rect(wd, (100, 255, 100), rect, 4)


def draw() -> None:
    wd.fill(color)
    for o in onscreen_objects:
        o.draw()
    player.draw()
    draw_aim_overlay()
    pygame.display.update()


def title_loop():
    wd.fill(color)
    rect = (WIDTH // 2 - 128, HEIGHT // 3 - 64, 256, 128)
    wd.blit(load_image_cached(join(PATH, "title.png")), rect)
    rect = (WIDTH // 2 - 64, 2 * HEIGHT // 3 - 32, 128, 64)
    wd.blit(load_image_cached(join(PATH, "play.png")), rect)
    if (
        WIDTH // 2 - 64 < mouse[0] < WIDTH // 2 + 64
        and 2 * HEIGHT // 3 - 32 < mouse[1] < 2 * HEIGHT // 3 + 32
        and pygame.mouse.get_pressed(num_buttons=3)[0]
    ):
        global gamestate
        gamestate = "play"
    pygame.display.update()


def scroll() -> None:
    global offset, look_offset, t_offset
    if 0 <= mouse[0] <= WIDTH and 0 <= mouse[1] <= HEIGHT:
        look_offset = [floor((mouse[i] - DIMS[i] // 2) * PERCEPTION) for i in (0, 1)]
    for i in (0, 1):
        b = [
            player.float_rect[i] + player.float_rect[i + 2] - DIMS[i] + SCROLL[i],
            player.float_rect[i] - SCROLL[i],
        ]
        for j in (0, 1):
            if offset[i] <= b[0] if j == 0 else offset[i] >= b[1]:
                offset[i] = b[j]
    t_offset = [offset[i] + look_offset[i] for i in (0, 1)]


def loop_color():
    global clock, mouse, tick, color
    clock.tick(FPS)
    tick += 1
    mouse = pygame.mouse.get_pos()
    if tick % 600 == 0:
        tick, color = 0, random_color()


def update_onscreen_objects():
    global onscreen_objects
    onscreen_objects = [o for o in level
        if all(-o.rect[i + 2] < o.rect[i] - t_offset[i] < DIMS[i] for i in (0, 1))]


def main() -> None:
    global offset, look_offset, t_offset, mouse, gravity, gamestate, data, level, player
    global level_num, color, clock, tick, onscreen_objects
    data, level, color = process_levels(LEVELS[level_num - 1], BGCOLOR)
    clock, tick = pygame.time.Clock(), 0
    mouse = pygame.mouse.get_pos()
    gamestate = "title"
    gravity = data[2]
    offset, look_offset, t_offset = [0, 0], [0, 0], [0, 0]
    player = Player(PLAYER_SIZE)
    onscreen_objects = []

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        loop_color()
        if gamestate == "play":
            scroll()
            update_onscreen_objects()
            player.loop()
            obj_loop()
            obj_interaction()
            handle_inputs()
            draw()
        elif gamestate == "title":
            title_loop()

    pygame.quit()


ALL_OBJECTS = {
    "object": Object,
    "bouncepad": Bouncepad,
    "block": Block,
    "unstable": Unstable,
    "moving": Moving,
    "item": Item,
    "lock": Lock,
}

if __name__ == "__main__":
    main()
