import pygame
from level import LEVELS, DEFAULT_BLOCK_HEALTH
from colors import COLORS
from guns import GUN_STATS
from os import listdir
from os.path import isfile, join
from random import randint
from math import floor, ceil, sin, pi, sqrt
from collections import Counter

pygame.init()

CAPTION = "A Tour of Colors"
ICON = "goal"

ANIM_DELAY = 7
WIDTH = 1000
HEIGHT = 800
DIMS = [WIDTH, HEIGHT]
FPS = 60

MSPEED = 15  # max ground speed
AGILE = 4  # ability to change direction
JUMP = 20
FRICTION = 0.5
STOP = 1
TVEL = 300  # max falling speed
PERCEPTION = 0.3
SCROLL = [WIDTH // 3, HEIGHT // 3]  # distance from side of screen to scroll x, y
RESP_BUFFER = 0.15  # secs before player goes back to start after dying
BOUNCE_STRENGTH = 1.14  # amount bouncepads bounce
GRAV_COOLDOWN = 30
SEARCH_RADIUS = 100  # radius in blocks to search for valid block placement
# precompute candidate offsets (bx, by) within SEARCH_RADIUS sorted by distance
SEARCH_OFFSETS = [
    (bx, by)
    for bx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1)
    for by in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1)
    if max(abs(bx), abs(by)) <= SEARCH_RADIUS
]
SEARCH_OFFSETS.sort(key=lambda t: (t[0] * t[0] + t[1] * t[1], abs(t[0]) + abs(t[1])))
# coral = (255, 96, 96)
# lime = (196, 255, 14)
BGCOLOR = "random"

CHARACTER = "plus"
PLAYER_SIZE = 32  # size of player sprite

# Toolbar / UI for equipping guns and blocks
TOOLBAR_HEIGHT = 64
TOOLBAR_ICON_SIZE = 48
TOOLBAR_PADDING = 16
TOOLBAR_BORDER = 4
# Toolbar block placement: tuples of (identifier, health)
BLOCK_PLACEMENT_OPTIONS = [
    (4, DEFAULT_BLOCK_HEALTH[4]),
    (8, DEFAULT_BLOCK_HEALTH[8]),
    (16, DEFAULT_BLOCK_HEALTH[16]),
    ("bouncepad", DEFAULT_BLOCK_HEALTH["bouncepad"]),
    ("ice", DEFAULT_BLOCK_HEALTH["ice"]),
]
# currently selected tools

level_num = 4

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


# ====================================


def random_color():
    return tuple([randint(color_range[i][0], color_range[i][1]) for i in range(3)])


def blit_rotated(rotated_surface, world_center) -> None:
    """Blit a pygame.rotate() surface, keeping its center at world_center."""
    screen_center = (
        floor(world_center[0] - t_offset[0]),
        floor(world_center[1] - t_offset[1]),
    )
    wd.blit(rotated_surface, rotated_surface.get_rect(center=screen_center))


def obj_on_screen(obj) -> bool:
    return all(-obj.rect[i + 2] < obj.rect[i] - t_offset[i] < DIMS[i] for i in (0, 1))


def update_onscreen_objects() -> None:
    global onscreen_objects
    onscreen_objects = [o for o in level if obj_on_screen(o)]


def block_rect_at_mouse(mouse_screen, size=64) -> pygame.Rect:
    half = size // 2
    return pygame.Rect(
        mouse_screen[0] + t_offset[0] - half,
        mouse_screen[1] + t_offset[1] - half,
        size,
        size,
    )


def block_placement_blocked(rect) -> bool:
    return any(rect.colliderect(o.rect) for o in level) or rect.colliderect(player.rect)


def resolve_enemy_spawn(
    target_center, search_radius=SEARCH_RADIUS, size=64
) -> tuple[int, int] | None:
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


def resolve_block_placement(
    mouse_screen, search_radius=SEARCH_RADIUS, block_size=64
) -> tuple[int, int] | None:
    anchor = block_rect_at_mouse(mouse_screen, block_size)
    if not block_placement_blocked(anchor):
        return anchor.topleft

    def inch_toward_cursor(start_rect: pygame.Rect) -> tuple[int, int]:
        """Move a valid rect toward cursor 1 pixel at a time until contact."""
        candidate = start_rect.copy()
        target_x = anchor.centerx
        target_y = anchor.centery
        x_done = y_done = False
        while not (x_done and y_done):
            if not x_done:
                dx = target_x - candidate.centerx
                if dx == 0:
                    x_done = True
                else:
                    trial_x = candidate.move((dx > 0) - (dx < 0), 0)
                    if block_placement_blocked(trial_x):
                        x_done = True
                    else:
                        candidate = trial_x

            if not y_done:
                dy = target_y - candidate.centery
                if dy == 0:
                    y_done = True
                else:
                    trial_y = candidate.move(0, (dy > 0) - (dy < 0))
                    if block_placement_blocked(trial_y):
                        y_done = True
                    else:
                        candidate = trial_y
        return candidate.topleft

    # test precomputed candidates starting from closest and return the
    # first placement that can be inched toward the cursor
    spiral = [
        (bx, by)
        for (bx, by) in SEARCH_OFFSETS
        if max(abs(bx), abs(by)) <= search_radius
    ]
    for bx, by in spiral:
        candidate = anchor.move(bx * block_size, by * block_size)
        if block_placement_blocked(candidate):
            continue
        # pygame.draw.rect(
        #     wd,
        #     (255, 0, 0),
        #     [
        #         candidate.x - t_offset[0],
        #         candidate.y - t_offset[1],
        #         candidate.width,
        #         candidate.height,
        #     ],
        #     2,
        # )
        return inch_toward_cursor(candidate)

    return None


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


def load_img(path, width, height):
    image = pygame.image.load(path)
    surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
    rect = pygame.Rect(0, 0, width, height)
    surface.blit(image.convert_alpha(), (0, 0), rect)
    return surface


def valid_obj_coll(obj):
    return (
        obj.name != "layer"
        and not (obj.name == "unstable" and obj.count_to_break <= 0)
        and not (obj.name == "item")
        and not (obj.name == "lock" and obj.unlocked)
    )


def enemy_can_see_player(enemy, player) -> bool:
    start = enemy.rect.center
    end = player.rect.center
    for obj in level:
        if obj.name != "enemy" and obj.rect.clipline(start, end):
            return False
    return True


def process_levels(level, color):
    wd.fill([randint(0, 255) for _ in range(3)])
    pos = (WIDTH // 2 - 128, HEIGHT // 2 - 32)
    wd.blit(load_img(join(PATH, "load.png"), 256, 64), pos)
    pygame.display.update()

    def create_object(obj):
        special_objs = {
            "bouncepad": Bouncepad,
            "block": Block,
            "unstable": Unstable,
            "moving": Moving,
            "item": Item,
            "lock": Lock,
        }
        # Extract health if present in level entry: [space, name, health] for healthable objects
        obj_name = obj[1]
        health = obj[2] if len(obj) > 2 and isinstance(obj[2], int) else None
        obj_type = special_objs.get(obj_name, Object)

        if health is not None and obj_name == "block":
            created_obj = obj_type(obj[0], obj[1], health=health)
        elif health is not None and obj_name == "bouncepad":
            created_obj = obj_type(obj[0], health=health)
        elif health is not None and obj_name == "unstable":
            created_obj = obj_type(obj[0], health=health)
        elif health is not None and obj_name in ["ice", "sticky", "moving"]:
            # Special block objects that can have health
            created_obj = obj_type(obj[0], obj_name, health=health)
        else:
            created_obj = obj_type(*obj)
        return (
            created_obj.wires + [created_obj]
            if created_obj.name == "moving"
            else [created_obj]
        )

    global color_range
    color_range, looking_for_color = COLORS[0], True
    while looking_for_color:
        option = COLORS[randint(0, len(COLORS) - 1)]
        if color_range != option:
            color_range = option
            looking_for_color = False
    level_with_objs = []
    for obj in level[1:]:
        level_with_objs += create_object(obj)
    return (level[0], level_with_objs, random_color() if color == "random" else color)


class OnScreenObject(pygame.sprite.Sprite):
    def __init__(self, rect, image=None, float_rect=None) -> None:
        super().__init__()
        self.rect = pygame.Rect(rect)
        self.screen_rect = self.rect.copy()
        self.image = image
        self.float_rect = (
            list(float_rect)
            if float_rect is not None
            else [
                float(self.rect.x),
                float(self.rect.y),
                float(self.rect.w),
                float(self.rect.h),
            ]
        )
        self.name = None
        self.path = None
        self.dead = False


class Player(OnScreenObject):
    def __init__(self, size) -> None:
        super().__init__(
            rect=pygame.Rect(*data[0], size, size),
            float_rect=[*data[0], size, size],
        )
        self.name = "player"
        self.size = size
        self.image = load_img(join(PATH, "characters", CHARACTER + ".png"), size, size)
        self.images = [flip_image([self.image])[0], self.image]
        self.xvel = self.yvel = self.stationary_xvel = self.stationary_yvel = 0
        self.direction, self.walking, self.fallcount = 1, False, 0
        self.inventory, self.gravity_switch, self.friction = [], 0, FRICTION
        self.mspeed, self.agile, self.stop = MSPEED, AGILE, STOP
        self.hit_count = self.loaded = 0
        self.collide, self.last_block_on = [None] * 4, None
        self.angle = 0
        self.bullets = []
        self.toolbar = [
            [
                GUN_STATS["debug_gun"],
                GUN_STATS["pistol"],
                GUN_STATS["pistol"],
                GUN_STATS["pistol"],
                GUN_STATS["pistol"],
            ],
            BLOCK_PLACEMENT_OPTIONS,
        ]
        self.selected = [self.toolbar[0][0], self.toolbar[1][0]]
        self.select_num = [0, 0]
        self.update_sprite()
        self.respawn(data[0])
        self.update_gun()

    def update_sprite(self) -> None:
        self.direction = int(self.direction)
        self.image = self.images[(self.direction + 1) // 2]
        self.update()

    def update(self) -> None:
        self.rect = self.image.get_rect(
            topleft=tuple([round(self.float_rect[i]) for i in [0, 1]])
        )
        self.screen_rect = self.rect.copy()

    def bullet_targets(self):
        return [o for o in level if valid_obj_coll(o) or o.name in ("block")]

    def update_aim(self) -> None:
        mouse_pos_diff_x = (
            mouse[0] + t_offset[0] - self.float_rect[0] - self.float_rect[2] / 2
        )
        mouse_pos_diff_y = (
            mouse[1] + t_offset[1] - self.float_rect[1] - self.float_rect[3] / 2
        )
        mouse_pos_diff_mag = sqrt(mouse_pos_diff_x**2 + mouse_pos_diff_y**2)

        self.angle = -pygame.Vector2(mouse_pos_diff_x, mouse_pos_diff_y).as_polar()[1]
        self.aim_dir = [
            mouse_pos_diff_x / mouse_pos_diff_mag,
            mouse_pos_diff_y / mouse_pos_diff_mag,
        ]

    def update_gun(self) -> None:
        self.update_aim()

        if pygame.mouse.get_pressed(num_buttons=3)[0] and self.loaded >= 0:
            self.bullets.append(
                Bullet(
                    self,
                    self.bullet_targets(),
                    pygame.Rect(
                        self.rect.centerx - 32,
                        self.rect.centery - 32,
                        64,
                        64,
                    ),
                    self.selected[0]["bullet_image"],
                )
            )
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
                bullet.loop(self, self.bullet_targets())

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
        x = DIMS[0] // 2 - (
            len(self.toolbar[0]) * (TOOLBAR_ICON_SIZE + TOOLBAR_PADDING)
            + TOOLBAR_PADDING
        )
        y = HEIGHT - TOOLBAR_HEIGHT + (TOOLBAR_HEIGHT - TOOLBAR_ICON_SIZE) // 2

        # draw guns
        for i, gun in enumerate(self.toolbar[0]):
            img = load_img(
                join(PATH, "guns", gun["gun_image"] + ".png"),
                TOOLBAR_ICON_SIZE,
                TOOLBAR_ICON_SIZE,
            )
            rect = pygame.Rect(x, y, TOOLBAR_ICON_SIZE, TOOLBAR_ICON_SIZE)
            wd.blit(img, rect.topleft)
            if i == self.select_num[0]:
                pygame.draw.rect(wd, (255, 255, 0), rect, TOOLBAR_BORDER)
            else:
                pygame.draw.rect(wd, (0, 0, 0), rect, TOOLBAR_BORDER)
            x += TOOLBAR_ICON_SIZE + TOOLBAR_PADDING

        # separator
        x += TOOLBAR_PADDING

        # draw block placement selectors, including special items
        block_ids = [
            item[0] if isinstance(item, tuple) else item for item in self.toolbar[1]
        ]
        max_size = max(item for item in block_ids if isinstance(item, int))
        for i, item in enumerate(self.toolbar[1]):
            block_id = item[0] if isinstance(item, tuple) else item
            box_rect = pygame.Rect(x, y, TOOLBAR_ICON_SIZE, TOOLBAR_ICON_SIZE)
            if isinstance(block_id, int):
                inner_w = int((block_id / max_size) * (TOOLBAR_ICON_SIZE - 8))
                inner_h = int((block_id / max_size) * (TOOLBAR_ICON_SIZE - 8))
                inner_x = box_rect.x + (TOOLBAR_ICON_SIZE - inner_w) // 2
                inner_y = box_rect.y + (TOOLBAR_ICON_SIZE - inner_h) // 2
                pygame.draw.rect(wd, (0, 0, 0), box_rect)
                pygame.draw.rect(
                    wd,
                    (200, 200, 200),
                    (inner_x, inner_y, inner_w, inner_h),
                )
            else:
                img = load_img(
                    join(PATH, "objects", block_id + ".png"),
                    TOOLBAR_ICON_SIZE,
                    TOOLBAR_ICON_SIZE,
                )
                wd.blit(img, box_rect.topleft)
            if i == self.select_num[1]:
                pygame.draw.rect(wd, (255, 255, 0), box_rect, TOOLBAR_BORDER)
            else:
                pygame.draw.rect(wd, (0, 0, 0), box_rect, TOOLBAR_BORDER)
            x += TOOLBAR_ICON_SIZE + TOOLBAR_PADDING

    def draw(self) -> None:
        for bullet in self.bullets:
            if not bullet.dead:
                blit_rotated(bullet.rotated_image, bullet.rect.center)
        image_pos = [self.float_rect[i] - t_offset[i] for i in [0, 1]]
        wd.blit(self.image, [floor(image_pos[i]) for i in [0, 1]])
        blit_rotated(self.rotated_gun, self.rect.center)
        self.draw_toolbar()

    def respawn(self, start) -> list[float, float]:
        global gravity
        self.float_rect[:2], self.collide, self.fallcount = start, [None] * 4, 1
        gravity, self.last_block_on, self.xvel, self.yvel, self.inventory = (
            data[2],
            None,
            0,
            0,
            [],
        )
        self.bullets, self.loaded = [], 0
        for obj in level:
            if obj.name == "item":
                obj.collected = False
            if obj.name == "unstable":
                obj.count_to_break = 60
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
        if self.hit_count > FPS * RESP_BUFFER:
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

        self.yvel = (
            TVEL if self.yvel > TVEL else -TVEL if self.yvel < -TVEL else self.yvel
        )

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
            if self.collide[i]:
                self.collide[i] = next(
                    (obj for obj in onscreen_objects if self.try_direction(axis, obj)),
                    None,
                )
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
                if (obj.rect.centerx - self.rect.centerx) ** 2 + (
                    obj.rect.centery - self.rect.centery
                ) ** 2 > (obj.squared_max_diag / 4) + (2 * self.size**2) + (
                    self.size * (obj.rect.width + obj.rect.height)
                ):
                    continue
                collided = self.has_collided(obj)
                if not collided:
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


class Bullet(OnScreenObject):
    def __init__(self, player, objects, rect, path) -> None:
        super().__init__(rect=rect)
        self.angle = player.angle
        self.speed = player.selected[0]["bullet_speed"]
        self.damage = player.selected[0].get("damage", 0)
        self.path = path
        self.dead = False
        self.movement = [
            player.aim_dir[0] * self.speed,
            player.aim_dir[1] * self.speed,
        ]
        self.loop(player, objects)

    def loop(self, player, objects) -> None:
        self.float_rect[0] += self.movement[0]
        self.float_rect[1] += self.movement[1]
        self.rect.x, self.rect.y = round(self.float_rect[0]), round(self.float_rect[1])
        self.screen_rect = self.rect.copy()

        # use cached load/rotation for bullet image
        base_path = (
            join(PATH, "bullets", self.path + ".png")
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
            if not self.rect.colliderect(object.rect):
                continue
            if getattr(object, "health", None) is not None:
                try:
                    object.health -= self.damage
                except Exception:
                    object.health = 0
                if object.health <= 0:
                    try:
                        level.remove(object)
                    except Exception:
                        object.dead = True
                    try:
                        onscreen_objects.remove(object)
                    except Exception:
                        pass
            self.dead = True
            break


class Object(OnScreenObject):
    def __init__(self, space, name, path=None, angle=0, health=None) -> None:
        super().__init__(
            rect=pygame.Rect(*space),
            float_rect=[
                float(space[0]),
                float(space[1]),
                float(space[2]),
                float(space[3]),
            ],
        )
        if isinstance(path, int):
            angle, path = path, name
        self.name, self.path = name, path or name
        self.squared_max_diag = self.rect.width**2 + self.rect.height**2
        self.color, self.cooldown = random_color(), 50
        self.bord = tuple(abs(i - randint(0, 35)) for i in self.color)
        if name in ["block", "bouncepad", "unstable", "ice", "sticky", "moving"]:
            if health is None:
                health = 100
            self.health = health
            self.max_health = health
        if name not in ["block", "item", "lock"]:
            self.image = rotate_image(
                [load_img(join(PATH, "objects", self.path + ".png"), *space[2:])], angle
            )[0]
        else:
            self.image = pygame.surface.Surface(self.rect.size)

    def draw(self) -> None:
        pos = [self.rect.x, self.rect.y]
        wd.blit(self.image, tuple(pos[i] - t_offset[i] for i in [0, 1]))
        if getattr(self, "health", None) is not None:
            max_h = getattr(self, "max_health", 100) or 100
            health_ratio = max(0, min(1, self.health / max_h))
            bar_color = [int(self.color[i] * health_ratio) for i in range(3)]
            overlay = pygame.Surface(
                (self.rect.w - 8, self.rect.h - 8), pygame.SRCALPHA
            )
            overlay.fill((*bar_color, 150))
            wd.blit(
                overlay, (self.rect.x - t_offset[0] + 4, self.rect.y - t_offset[1] + 4)
            )
            pygame.draw.rect(
                wd,
                (*bar_color, 160),
                (
                    self.rect.x - t_offset[0] + 4,
                    self.rect.y - t_offset[1] + 4,
                    self.rect.w - 8,
                    self.rect.h - 8,
                ),
                2,
            )


class Block(Object):
    def __init__(self, space, _, path=None, health=None):
        super().__init__(space, "block", path=path, health=health)
        # blocks have health from BLOCK_HEALTH config, overridable
        if health is not None:
            self.health = health
            self.max_health = health
        if path is None:
            path = f"block{space[2]//64}x{space[3]//64}"

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
    def __init__(self, space, path="bouncepad", angle=0, health=None) -> None:
        super().__init__(space, "bouncepad", health=health)
        sprite_sheet = load_sprite_sheets(join(PATH, "objects"), space[2], space[3])
        self.sprites = rotate_image(sprite_sheet[path], angle)
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
    def __init__(
        self,
        space,
        path="unstable",
        angle=0,
        count_to_break=60,
        respawn_buffer=120,
        health=None,
    ) -> None:
        super().__init__(space, "unstable", path, angle, health=health)
        if health is not None:
            self.health = health
            self.max_health = health
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
    def __init__(
        self,
        space,
        path="moving",
        _=0,
        move_axis=0,
        move_range=10 * 64,
        speed=4,
        health=None,
    ):
        self.angle = move_axis * 90
        super().__init__(space, "moving", path, self.angle)
        self.sprite_sheet = load_sprite_sheets(
            join(PATH, "objects"), space[2], space[3]
        )
        self.sprites = rotate_image(self.sprite_sheet[path], self.angle)
        wire = [64, 0] if move_axis == 0 else [0, 64]
        args = ["layer", "wire", self.angle]
        self.wires = [
            Object([space[0] + wire[0] * i, space[1] + wire[1] * i, 64, 64], *args)
            for i in range((move_range // 64) + 1)
        ]
        self.anim, self.count, self.anim_delay = 0, 0, 1
        self.change_direction = 0
        self.start_pos = [space[0], space[1]]
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
    def __init__(self, space, name, path, anim_delay=30):
        super().__init__(space, name)
        self.item_name, self.anim_delay, self.count, self.collected, self.orig_y = (
            path,
            anim_delay,
            0,
            False,
            self.rect.y,
        )
        self.image = load_img(join(PATH, "items", path + ".png"), *space[2:])

    def draw(self):
        if not self.collected:
            super().draw()

    def loop(self):
        self.rect.y = self.orig_y + 2 * sin((tick % 60) / 30 * pi)


class Lock(Object):
    def __init__(self, space, name, key_name):
        super().__init__(space, name)
        self.images = [
            load_img(join(PATH, "objects", p + ".png"), *self.rect.size)
            for p in ["lock_" + key_name[-1], "open_" + key_name[-1]]
        ]
        self.image, self.key_name, self.unlocked = self.images[0], key_name, False

    def loop(self):
        self.unlocked = any(i.item_name == self.key_name for i in player.inventory)
        self.image = self.images[self.unlocked]


class Enemy(Object):
    def __init__(self, space, path=None, health=200, speed=3):
        super().__init__(space, "enemy", health=health)
        self.health = health
        self.max_health = health
        self.speed = speed

    def loop(self):
        if not enemy_can_see_player(self, player):
            return None

        # move toward the player in small steps and stop on collision
        dx = player.rect.centerx - self.rect.centerx
        dy = player.rect.centery - self.rect.centery
        mag = sqrt(dx * dx + dy * dy) or 1
        move_x = (dx / mag) * self.speed
        move_y = (dy / mag) * self.speed

        # ensure we have float tracking for smooth movement
        if not hasattr(self, "float_rect"):
            self.float_rect = [
                float(self.rect.x),
                float(self.rect.y),
                float(self.rect.w),
                float(self.rect.h),
            ]

        steps = max(1, ceil(max(abs(move_x), abs(move_y))))
        step_x = move_x / steps
        step_y = move_y / steps
        stopped = False
        for _ in range(steps):
            self.float_rect[0] += step_x
            self.float_rect[1] += step_y
            self.rect.x = int(round(self.float_rect[0]))
            self.rect.y = int(round(self.float_rect[1]))

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
                self.rect.x = int(round(self.float_rect[0]))
                self.rect.y = int(round(self.float_rect[1]))
                stopped = True
                break
            if stopped:
                break


def obj_loop() -> None:
    for obj in level:
        if obj.dead:
            continue
        if obj.name in ["moving", "bouncepad", "item", "lock", "unstable", "enemy"]:
            obj.loop()


def obj_interaction() -> bool:
    global gravity, data, level, level_num, color
    keys, short_jump = pygame.key.get_pressed(), False
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

    if keys[pygame.K_l]:
        level_num = (level_num + 1) % len(LEVELS)
        data, level, color = process_levels(LEVELS[level_num - 1], BGCOLOR)
        gravity = data[2]
        player.respawn(data[0])
        return None
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
        elif obj.name == "spike":
            player.hit_count += 1
        elif obj.name == "enemy":
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
                short_jump = True
        elif obj.name == "unstable":
            obj.count_to_break -= 1
        elif obj.name == "ice":
            player.friction = 0.99
            player.mspeed, player.agile, player.stop = (
                MSPEED * 1.5,
                AGILE * 0.5,
                STOP * 0.5,
            )
        elif obj.name == "moving":
            if obj.move_axis == 0:
                player.stationary_xvel = obj.speed
            else:
                player.stationary_yvel = obj.speed
        elif obj.name == "item":
            obj.collected = True
            player.inventory.append(obj)

    m_agile = player.mspeed - player.agile
    if keys[pygame.K_r]:
        player.respawn(data[0])
    if keys[pygame.K_c]:
        player.inventory = []
    if pygame.mouse.get_pressed(num_buttons=3)[2]:
        block_data = player.selected[1]
        if isinstance(block_data, tuple):
            block_choice, block_health = block_data
        else:
            block_choice, block_health = block_data, 100
        block_size = block_choice if isinstance(block_choice, int) else PLAYER_SIZE
        pos = resolve_block_placement(mouse, block_size=block_size)
        if pos:
            if isinstance(block_choice, int):
                level.append(
                    Block((*pos, block_size, block_size), "block", health=block_health)
                )
            elif block_choice == "bouncepad":
                level.append(
                    Bouncepad((*pos, block_size, block_size), health=block_health)
                )
            elif block_choice == "unstable":
                level.append(
                    Unstable((*pos, block_size, block_size), health=block_health)
                )
            else:
                obj = Object((*pos, block_size, block_size), block_choice)
                if hasattr(obj, "health"):
                    obj.health = block_health
                level.append(obj)
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
        if short_jump:
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
            level.append(Enemy([spawn_point[0], spawn_point[1], 64, 64]))


def draw_aim_overlay() -> None:
    player_screen = (
        floor(player.rect.centerx - t_offset[0]),
        floor(player.rect.centery - t_offset[1]),
    )
    mouse_screen = (floor(mouse[0]), floor(mouse[1]))
    pygame.draw.line(wd, (255, 255, 255), player_screen, mouse_screen, 1)
    block_data = player.selected[1]
    if isinstance(block_data, tuple):
        block_choice = block_data[0]
    else:
        block_choice = block_data
    block_size = block_choice if isinstance(block_choice, int) else PLAYER_SIZE
    half = block_size // 2
    resolved = resolve_block_placement(mouse, block_size=block_size)
    naive_color = (
        (255, 0, 0)
        if resolved and resolved != block_rect_at_mouse(mouse, block_size).topleft
        else (100, 255, 100)
    )

    pygame.draw.rect(
        wd,
        naive_color,
        (mouse[0] - half, mouse[1] - half, block_size, block_size),
        4,
    )
    if resolved and resolved != block_rect_at_mouse(mouse, block_size).topleft:
        pygame.draw.rect(
            wd,
            (100, 255, 100),
            (
                resolved[0] - t_offset[0],
                resolved[1] - t_offset[1],
                block_size,
                block_size,
            ),
            4,
        )


def draw() -> None:
    wd.fill(color)
    for o in onscreen_objects:
        o.draw()
    player.draw()
    draw_aim_overlay()
    pygame.display.update()


def title_loop():
    wd.fill(color)
    wd.blit(
        load_img(join(PATH, "title.png"), 256, 128),
        (WIDTH // 2 - 128, HEIGHT // 3 - 64, 256, 128),
    )
    wd.blit(
        load_img(join(PATH, "play.png"), 128, 64),
        (WIDTH // 2 - 64, 2 * HEIGHT // 3 - 32, 128, 64),
    )
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
            draw()
        elif gamestate == "title":
            title_loop()

    pygame.quit()


if __name__ == "__main__":
    main()
