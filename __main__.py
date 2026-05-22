import pygame
from level import LEVELS
from colors import COLORS
from os import listdir
from os.path import isfile, join
from random import randint
from math import floor, ceil, sin, pi
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
PERCEPTION = 0.7
SCROLL = [250, 175]  # distance from side of screen to scroll x, y
RESP_BUFFER = 0.15  # secs before player goes back to start after dying
BOUNCE_STRENGTH = 1.14  # amount bouncepads bounce
GRAV_COOLDOWN = 30
# coral = (255, 96, 96)
# lime = (196, 255, 14)
BGCOLOR = "random"

CHARACTER = "plus"
PLAYER_SIZE = 128  # size of player sprite

level_num = 8

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


def random_color():
    return tuple([randint(color_range[i][0], color_range[i][1]) for i in range(3)])


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
            "movableblock": MovableBlock,  # Add new class here
        }
        obj = special_objs.get(obj[1], Object)(*obj)
        return obj.wires + [obj] if obj.name == "moving" else [obj]

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


class Player(pygame.sprite.Sprite):
    def __init__(self, size) -> None:
        super().__init__()
        self.name = "player"
        self.image = load_img(join(PATH, "characters", CHARACTER + ".png"), size, size)
        self.images, self.size = [flip_image([self.image])[0], self.image], size
        self.float_rect = [*data[0], size, size]
        self.xvel = self.yvel = self.stationary_xvel = self.stationary_yvel = 0
        self.mask, self.direction, self.walking, self.fallcount = None, 1, False, 0
        self.inventory, self.gravity_switch, self.friction = [], 0, FRICTION
        self.mspeed, self.agile, self.stop = MSPEED, AGILE, STOP
        self.hit_count = self.loaded = 0
        self.collide, self.last_block_on = [None] * 4, None
        self.update_sprite()
        self.respawn(data[0])

    def update_sprite(self) -> None:
        self.direction = int(self.direction)
        self.image = self.images[(self.direction + 1) // 2]
        self.update()

    def update(self) -> None:
        self.rect = self.image.get_rect(
            topleft=tuple([round(self.float_rect[i]) for i in [0, 1]])
        )
        self.mask = pygame.mask.from_surface(self.image)

    def draw(self) -> None:
        image_pos = [self.float_rect[i] - t_offset[i] for i in [0, 1]]
        wd.blit(self.image, [floor(image_pos[i]) for i in [0, 1]])

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
        for obj in level:
            if obj.name == "item":
                obj.collected = False
            if obj.name == "movableblock":
                obj.__init__(obj.space, obj.name)
            if obj.name == "target":
                obj.dead = False
            if obj.name == "unstable":
                obj.count_to_break = 60
        self.update()

    def loop(self) -> list[float, float]:
        self.adjust_speed()
        self.collision()
        self.end_collision()
        self.update_sprite()

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
        colliding = pygame.sprite.collide_mask(self, obj)
        if colliding and obj.name == "item":
            self.inventory.append(obj)
            obj.collected = True
        return colliding and valid_obj_coll(obj)

    def collision(self) -> None:
        def try_push(b, inc):
            b.add_incr(*inc)
            moved.append(b)
            b.pushed = True
            for o in level:
                if o not in moved and o is not self and b.has_collided(o):
                    if o.name == "movableblock":
                        if not try_push(o, inc):
                            return False
                    elif valid_obj_coll(o):
                        return False
            return True

        self.xvel_before_coll, self.yvel_before_coll = self.xvel, self.yvel
        axes = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for i, axis in enumerate(axes):
            if self.collide[i]:
                self.collide[i] = next(
                    (obj for obj in level if self.try_direction(axis, obj)), None
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
            for obj in level:
                collided = self.has_collided(obj)
                if not collided:
                    continue
                if collided and obj.name == "movableblock":
                    x_clears, y_clears = (
                        (not self.try_direction(li, obj))
                        for li in [[-increment[0], 0], [0, -increment[1]]]
                    )
                    p_inc, p_dir = (
                        [li[0] * x_clears, li[1] * y_clears]
                        for li in [increment, direction]
                    )
                    moved = []

                    if (
                        any(p_inc)
                        and try_push(obj, p_inc)
                        and not self.has_collided(obj)
                    ):
                        for b in moved:
                            b.push(p_dir)
                        coll = [self.try_direction(i, obj) for i in axes]
                        for i in range(4):
                            if coll[i] and direction[i // 2] not in axes[i % 2]:
                                self.collide[i] = obj
                            elif self.collide[i] == obj and (
                                (axes[i][0] and axes[i][0] == p_inc[0])
                                or (axes[i][1] and axes[i][1] == p_inc[1])
                            ):
                                self.collide[i] = None
                        continue
                    for m in moved:
                        m.add_incr(-p_inc[0], -p_inc[1])
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
        if self.xvel != 0:
            self.try_mask(abs(self.xvel) // self.xvel, level)

    def try_mask(self, direction, objects) -> bool:
        orig_direction, self.direction = self.direction, direction
        self.update_sprite()
        for obj in objects:
            if self.has_collided(obj):
                self.direction = orig_direction
                self.update_sprite()
                return False
        return True


class Object(pygame.sprite.Sprite):
    def __init__(self, space, name, path=None, angle=0) -> None:
        super().__init__()
        if isinstance(path, int):
            angle, path = path, name
        self.rect, self.name, self.path = pygame.Rect(*space), name, path or name
        self.color, self.cooldown = random_color(), 50
        self.bord = tuple(abs(i - randint(0, 35)) for i in self.color)
        self.dead = False
        if name not in ["block", "item", "lock"]:
            self.image = rotate_image(
                [load_img(join(PATH, "objects", self.path + ".png"), *space[2:])], angle
            )[0]
        else:
            self.image = pygame.surface.Surface(self.rect.size)
        self.update_mask()

    def draw(self) -> None:
        pos = [self.rect.x, self.rect.y]
        wd.blit(self.image, tuple(pos[i] - t_offset[i] for i in [0, 1]))

    def update_mask(self) -> None:
        self.mask = pygame.mask.from_surface(self.image)


class Block(Object):
    def __init__(self, space, _, path=None):
        super().__init__(space, "block")
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


class Bouncepad(Object):
    def __init__(self, space, path="bouncepad", angle=0) -> None:
        super().__init__(space, "bouncepad")
        sprite_sheet = load_sprite_sheets(join(PATH, "objects"), space[2], space[3])
        self.sprites = rotate_image(sprite_sheet[path], angle)
        self.anim, self.bounced, self.angle = 0, 0, angle

    def loop(self) -> None:
        if 0 < self.bounced <= 2 * len(self.sprites):
            self.update_mask()
            self.bounced += 1
            self.image = self.sprites[(self.anim // 2) % len(self.sprites)]
            self.anim = 0 if self.anim // 2 > len(self.sprites) else self.anim + 1
        else:
            self.anim, self.bounced = 0, 0
            self.image = self.sprites[-1]


class Unstable(Object):
    def __init__(
        self, space, path="unstable", angle=0, count_to_break=60, respawn_buffer=120
    ) -> None:
        super().__init__(space, "unstable", path, angle)
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
                pygame.sprite.collide_mask(self, player)
                or any(
                    o.name in ["movableblock", "moving"]
                    and pygame.sprite.collide_mask(self, o)
                    for o in level
                    if o is not self
                )
            ):
                self.count_to_break = self.break_len


class Moving(Object):
    def __init__(
        self, space, path="moving", _=0, move_axis=0, move_range=10 * 64, speed=4
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
        self.update_mask()

        if self.change_direction:
            return None

        colliding = pygame.sprite.collide_mask(player, self)
        move_direction = abs(self.speed) // self.speed
        axis = self.move_axis == 1
        while colliding:
            colliding = pygame.sprite.collide_mask(player, self)
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
        self.update_mask()


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
        self.update_mask()


class MovableBlock(Object):
    def __init__(self, space, name="movableblock", path=None, angle=0):
        super().__init__(space, name, path, angle)
        # if not path:
        #     self.image = pygame.Surface(self.rect.size, pygame.SRCALPHA, 32)
        #     self.image.fill((255, 255, 255))
        self.float_rect, self.xvel = [float(f) for f in space], 0.0
        self.yvel = self.stationary_xvel = self.stationary_yvel = 0.0
        (
            self.fallcount,
            self.on_ground,
            self.push_speed,
            self.friction,
        ) = (0, False, MSPEED, 0.3)
        self.collide, self.space, self.pushed = [None] * 4, space, False
        self.update_mask()

    def update(self):
        self.rect.topleft = (round(self.float_rect[0]), round(self.float_rect[1]))
        self.update_mask()

    def add_incr(self, x, y):
        self.float_rect[0] += x
        self.float_rect[1] += y
        self.update()

    def has_collided(self, obj) -> bool:
        if obj is self:
            return False
        colliding = pygame.sprite.collide_mask(self, obj)
        return colliding and (obj is player or valid_obj_coll(obj))

    def try_direction(self, direction, obj):
        self.add_incr(direction[0], direction[1])
        collided = self.has_collided(obj)
        self.add_incr(-direction[0], -direction[1])
        return obj if collided else None

    def push(self, direction):
        if direction[0] != 0:
            self.xvel = player.xvel
        if direction[1] != 0:
            if abs(player.yvel) > abs(JUMP) * 0.5:
                self.yvel = direction[1] * abs(JUMP)
                self.fallcount = 1
            else:
                self.yvel = player.yvel

    def collision(self):
        axes = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        collision_list = level + [player]
        for i, axis in enumerate(axes):
            if self.collide[i]:
                self.collide[i] = next(
                    (obj for obj in collision_list if self.try_direction(axis, obj)),
                    None,
                )

        fx, fy = ceil(abs(self.xvel)), ceil(abs(self.yvel))
        max_speed = fx if fx > fy else fy
        if max_speed == 0:
            return None

        increment = [self.xvel / max_speed, self.yvel / max_speed]
        direction = [abs(i) / i if i else 0 for i in [self.xvel, self.yvel]]

        for _ in range(max_speed):
            self.add_incr(increment[0], increment[1])
            for obj in collision_list:
                collided = self.has_collided(obj)
                if collided and obj.name == "movableblock":
                    self.add_incr(0, -increment[1])
                    y_clears = not self.has_collided(obj)
                    self.add_incr(0, increment[1])
                    self.add_incr(-increment[0], 0)
                    x_clears = not self.has_collided(obj)
                    self.add_incr(increment[0], 0)

                    p_inc = [
                        increment[0] if x_clears else 0,
                        increment[1] if y_clears else 0,
                    ]
                    p_dir = [
                        direction[0] if x_clears else 0,
                        direction[1] if y_clears else 0,
                    ]
                    moved = []

                    def try_push(b, inc):
                        b.add_incr(inc[0], inc[1])
                        moved.append(b)
                        b.pushed = True
                        for o in collision_list:
                            if o is b or o in moved or o is self:
                                continue
                            if b.has_collided(o):
                                if o.name == "movableblock":
                                    if not try_push(o, inc):
                                        return False
                                elif valid_obj_coll(o):
                                    return False
                        return True

                    if (
                        any(p_inc)
                        and try_push(obj, p_inc)
                        and not self.has_collided(obj)
                    ):
                        for b in moved:
                            b.push(p_dir)
                        coll = [self.try_direction(i, obj) for i in axes]
                        for i in range(4):
                            if coll[i] and direction[i // 2] not in axes[i % 2]:
                                self.collide[i] = obj
                            elif self.collide[i] == obj and (
                                (axes[i][0] and axes[i][0] == p_inc[0])
                                or (axes[i][1] and axes[i][1] == p_inc[1])
                            ):
                                self.collide[i] = None
                        continue
                    for m in moved:
                        m.add_incr(-p_inc[0], -p_inc[1])
                if not collided:
                    continue
                self.add_incr(-increment[0], -increment[1])
                coll = [self.try_direction(i, obj) for i in axes]
                self.collide = [
                    obj if (coll[i] and (direction[i // 2] in axes[i % 2])) else None
                    for i in range(4)
                ]
                return None

    def end_collision(self):
        self.float_rect[0] = round(self.float_rect[0])
        self.float_rect[1] = round(self.float_rect[1])

        zero_set = [self.xvel, self.yvel]
        for i in range(4):
            if self.collide[i]:
                zero_set[i // 2] = 0
        self.xvel, self.yvel = zero_set

        if (self.collide[2] and gravity < 0) or (self.collide[3] and gravity > 0):
            self.fallcount = 0

    def loop(self):
        if self.rect.y > data[1][1]:
            self.dead = True
            return None
        # Apply gravity
        self.yvel = (
            TVEL if self.yvel > TVEL else -TVEL if self.yvel < -TVEL else self.yvel
        )
        self.fallcount += 1
        if not ((self.collide[2] and gravity < 0) or (self.collide[3] and gravity > 0)):
            self.yvel += (self.fallcount / FPS) * gravity

        if not self.pushed:
            self.collision()
        else:
            # If pushed horizontally, still need to check for falling/grounding
            y_save, self.xvel = self.xvel, 0
            self.collision()
            self.xvel = y_save

        self.end_collision()

        # Friction
        if (self.collide[3] and gravity > 0) or (self.collide[2] and gravity < 0):
            self.xvel *= self.friction
            if abs(self.xvel) < 0.5:
                self.xvel = 0
        if self.collide[3]:
            if self.collide[3].name == "player":
                self.xvel = player.xvel
        self.update()


def obj_loop() -> None:
    for obj in level:
        if obj.dead:
            continue
        if obj.name in [
            "moving",
            "bouncepad",
            "item",
            "lock",
            "movableblock",
            "unstable",
        ]:
            obj.loop()


def obj_interaction() -> bool:
    global gravity, data, level, level_num, color
    keys, short_jump = pygame.key.get_pressed(), False
    player.friction, last = FRICTION, player.last_block_on
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
        elif obj.name == "spike":
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
        elif obj.name == "movableblock":
            pushed, keys_p = False, {
                0: [pygame.K_LEFT, pygame.K_a],
                1: [pygame.K_RIGHT, pygame.K_d],
                2: [pygame.K_UP, pygame.K_w, pygame.K_SPACE],
                3: [pygame.K_DOWN, pygame.K_s],
            }
            for i, k_list in keys_p.items():
                if player.collide[i] == obj and any(keys[k] for k in k_list):
                    obj.push([(i == 1) - (i == 0), (i == 3) - (i == 2)])
                    pushed = True
            if not pushed:
                if player.collide[2] == obj:
                    obj.xvel = player.xvel
                if player.collide[3] == obj:
                    player.stationary_xvel = obj.xvel
        if obj.name == "goal" or keys[pygame.K_l] or keys[pygame.K_k]:
            level_num = (
                (level_num + 1) % len(LEVELS)
                if keys[pygame.K_l] or obj.name == "goal"
                else (level_num - 1 % len(LEVELS))
            )
            data, level, color = process_levels(LEVELS[level_num - 1], BGCOLOR)
            gravity = data[2]
            player.respawn(data[0])
            break

    m_agile = player.mspeed - player.agile
    if keys[pygame.K_r]:
        player.respawn(data[0])
    if keys[pygame.K_c]:
        player.inventory = []
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


def draw() -> None:
    wd.fill("black")
    for o in level:
        if all(-o.rect[i + 2] < o.rect[i] - t_offset[i] < DIMS[i] for i in (0, 1)):
            o.draw()
    player.draw()
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
    print("\n --- RUNNING --- \n")
    global offset, look_offset, t_offset, mouse, gravity, gamestate, data, level, player
    global level_num, color, clock, tick
    data, level, color = process_levels(LEVELS[level_num - 1], BGCOLOR)
    clock, tick = pygame.time.Clock(), 0
    mouse = pygame.mouse.get_pos()
    gamestate = "title"
    gravity = data[2]
    offset, look_offset, t_offset = [0, 0], [0, 0], [0, 0]
    player = Player(PLAYER_SIZE)

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                if gamestate == "play":
                    pos = [event.pos[i] + t_offset[i] - 32 for i in (0, 1)]
                    level.append(Block((*pos, 64, 64), "block"))

        loop_color()
        if gamestate == "play":
            scroll()
            for o in level:
                if o.name == "movableblock":
                    o.pushed = False
            player.loop()
            obj_loop()
            obj_interaction()
            draw()
        elif gamestate == "title":
            title_loop()

    print("\n --- QUITTING --- \n")
    pygame.quit()


if __name__ == "__main__":
    main()
