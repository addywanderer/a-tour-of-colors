import pygame
from level import LEVELS
from os import listdir
from os.path import isfile, join
from random import randint
from math import floor, ceil

pygame.init()

CAPTION = "A Tour of Colors"
ICON = "goal"

COLORS = [
    [[150, 255], [0, 50], [130, 200]],
    [[0, 50], [200, 255], [200, 255]],
    [[0, 50], [200, 255], [50, 100]],
    [[200, 255], [30, 70], [30, 70]],
    [[200, 255], [200, 255], [0, 50]],
]

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


def process_levels(level, color):
    wd.fill([randint(0, 255) for _ in range(3)])
    pos = (WIDTH // 2 - 128, HEIGHT // 2 - 32)
    wd.blit(load_img(join(PATH, "load.png"), 256, 64), pos)
    pygame.display.update()

    def create_object(obj):
        special_objs = {"bouncepad": Bouncepad, "block": Block}
        return special_objs.get(obj[1], Object)(*obj)

    global color_range
    color_range, looking_for_color = COLORS[0], True
    while looking_for_color:
        option = COLORS[randint(0, len(COLORS) - 1)]
        if color_range != option:
            color_range = option
            looking_for_color = False
    return (
        level[0],
        [create_object(obj) for obj in level[1:]],
        random_color() if color == "random" else color,
    )


class Player(pygame.sprite.Sprite):
    def __init__(self, start, size) -> None:
        super().__init__()
        self.image = load_img(join(PATH, "characters", "plus" + ".png"), size, size)
        self.images = [flip_image([self.image])[0], self.image]
        self.float_rect = [start[0], start[1], size, size]
        self.xvel, self.yvel = 0, 0
        self.mask, self.direction, self.walking = None, 1, False
        self.fallcount = 0
        self.gravity_switch = 0
        self.hit_count, self.loaded = 0, 0
        self.size = size
        self.collide = [None] * 4
        self.update_sprite()
        self.respawn(start)

    def update_sprite(self) -> None:
        self.image = self.images[self.direction]
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
        global offset, look_offset, t_offset, gravity
        offset, look_offset = scroll(self, look_offset)
        self.float_rect[0], self.float_rect[1] = start
        t_offset = [offset[i] + look_offset[i] for i in [0, 1]]
        self.collide, self.fallcount = [None] * 4, 1
        gravity = data[2]
        self.xvel, self.yvel = 0, 0
        self.update()

    def loop(self, fps, objects, data) -> list[float, float]:
        self.adjust_speed()
        self.collision(objects)
        self.update_sprite()

        if self.gravity_switch > 0:
            self.gravity_switch -= 1

        if self.hit_count:
            self.hit_count += 1
        if self.hit_count > fps * RESP_BUFFER:
            self.hit_count = 0
            self.respawn(data[0])
        elif not (data[1][0] <= self.float_rect[1] <= data[1][1]):
            self.respawn(data[0])

    def adjust_speed(self) -> None:
        if not self.walking:
            if self.xvel != 0:
                self.xvel *= FRICTION
            if -STOP <= self.xvel <= STOP:
                self.xvel = 0
        self.walking = False

        self.yvel = (
            TVEL if self.yvel > TVEL else -TVEL if self.yvel < -TVEL else self.yvel
        )

        self.fallcount += 1
        if not ((self.collide[2] and gravity < 0) or (self.collide[3] and gravity > 0)):
            self.yvel += (self.fallcount / FPS) * gravity  # gravity

    def collision(self, objects) -> None:
        def add_incr(x, y) -> None:
            self.float_rect[0] += x
            self.float_rect[1] += y
            self.update()

        def try_direction(direction, obj) -> Object:
            add_incr(direction[0], direction[1])
            collided = has_collided(obj)
            add_incr(-direction[0], -direction[1])
            return obj if collided else None

        def has_collided(obj) -> bool:
            return pygame.sprite.collide_mask(self, obj) and obj.name != "layer"

        def try_mask(direction) -> bool:
            orig_direction, self.direction = self.direction, direction
            self.update_sprite()
            for obj in objects:
                if has_collided(obj):
                    self.direction = orig_direction
                    self.update_sprite()
                    return False
            return True

        def end() -> None:
            for same in range(4):
                if same_coll[same]:
                    self.collide[same] = same_coll[same]
            for i in range(4):
                self.float_rect[i] = round(self.float_rect[i])
            for j in [0, 1]:
                if self.collide[j] and not (
                    self.collide[j].name == "bouncepad"
                    and self.collide[j].angle == [270, 90][j]
                ):
                    self.xvel = 0
            for k in [2, 3]:
                if self.collide[k] and not (
                    self.collide[k].name == "bouncepad"
                    and self.collide[k].angle == [180, 0][k - 2]
                ):
                    self.yvel = 0
            if (self.collide[2] and gravity < 0) or (self.collide[3] and gravity > 0):
                self.fallcount = 0
            if self.xvel < 0:
                try_mask(0)
            elif self.xvel > 0:
                try_mask(1)

        axes = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for i in range(4):
            changed_coll = False
            if not self.collide[i]:
                continue
            for obj in objects:
                if try_direction(axes[i], obj):
                    self.collide[i] = obj
                    changed_coll = True
                    break
            if not changed_coll:
                self.collide[i] = None
        fx, fy = ceil(abs(self.xvel)), ceil(abs(self.yvel))
        max_speed, same_coll = fx if fx > fy else fy, [None] * 4
        if max_speed == 0:
            end()
            return None
        if self.xvel == 0:
            same_coll[:1] = self.collide[:1]
        if self.yvel == 0:
            same_coll[1:] = self.collide[1:]
        increment = [self.xvel / max_speed, self.yvel / max_speed]
        direction = [abs(i) / i if i else 0 for i in [self.xvel, self.yvel]]
        for _ in range(max_speed):
            add_incr(increment[0], increment[1])
            for obj in objects:
                if not has_collided(obj):
                    continue

                add_incr(-increment[0], -increment[1])
                coll = [try_direction(i, obj) for i in axes]
                self.collide = [
                    obj if (coll[i] and (direction[i // 2] in axes[i % 2])) else None
                    for i in range(4)
                ]  # left, right, top, bottom
                end()
                return None
        end()


class Object(pygame.sprite.Sprite):
    def __init__(self, space, name, path=None, angle=0) -> None:  # space = x, y, w, h
        super().__init__()
        path = name if path is None else path
        if type(path) is int:
            angle, path = path, name

        self.rect, self.name, self.path = pygame.Rect(*space), name, path
        self.color, self.cooldown = random_color(), 50
        self.bord = tuple(abs(i - randint(0, 35)) for i in self.color)
        if name != "block":
            self.image = load_img(
                join(PATH, "objects", path + ".png"), space[2], space[3]
            )
            self.image = rotate_image([self.image], angle)[0]
        else:
            self.image = pygame.surface.Surface((self.rect.w, self.rect.h))
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
        if self.cooldown == 0:
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


def obj_interaction(player, level_num, level, color, start) -> bool:
    def bounce_func(vel):
        return bounce[i] * (abs(vel)) ** 0.9

    keys = pygame.key.get_pressed()
    global gravity, data
    for obj in player.collide:
        if not obj:
            continue
        if obj.name == "block":
            obj.cooldown -= 1
        if obj.name == "spike":
            player.hit_count += 1
        if obj.name == "gravity" and player.gravity_switch == 0:
            gravity *= -1
            player.gravity_switch = GRAV_COOLDOWN
        if obj.name == "checkpoint":
            global data
            data[0] = [
                obj.rect.centerx - player.size // 2,
                obj.rect.centery - player.size // 2 - 64,
            ]
        if obj.name == "bouncepad":
            angles, bounce = [3, 1, 2, 0], [BOUNCE_STRENGTH, -BOUNCE_STRENGTH] * 2
            for i in range(4):
                if obj.angle == angles[i] * 90 and player.collide[i] == obj:
                    obj.bounced, obj.anim = 1, 0
                    if i // 2 == 0:
                        player.xvel = bounce_func(player.xvel)
                        # player.xvel = -(player.xvel + 10) * bounce[i]
                    else:
                        player.yvel = bounce_func(player.yvel)
        if obj.name == "goal" or keys[pygame.K_l]:
            level_num += 1
            data, level, color = process_levels(LEVELS[level_num - 1], BGCOLOR)
            gravity = data[2]
            player.respawn(data[0])
            break

    if keys[pygame.K_r]:
        player.respawn(start)
    if keys[pygame.K_p]:
        pygame.display.toggle_fullscreen()
        pygame.display.set_icon(pygame.image.load(join(PATH, ICON)))
        pygame.display.update()
    if (keys[pygame.K_LEFT] or keys[pygame.K_a]) and (not player.collide[0]):
        player.walking = True
        player.xvel = -MSPEED if player.xvel <= AGILE - MSPEED else player.xvel - AGILE
    elif (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and (not player.collide[1]):
        player.walking = True
        player.xvel = MSPEED if player.xvel >= MSPEED - AGILE else player.xvel + AGILE
    if (
        keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_SPACE]
    ) and player.fallcount == 0:
        player.yvel = -JUMP * (gravity // abs(gravity))
        player.fallcount = 0
    return level_num, data, level, color


def draw(wd, player, objects, color) -> None:
    tile_image = pygame.image.load(join(PATH, "background", TILES[level_num - 1]))
    tile_dims = tile_image.get_rect()[2:]
    ranges = [range(DIMS[i] // tile_dims[i] + 10) for i in [0, 1]]
    [
        [wd.blit(tile_image, (i * tile_dims[0], j * tile_dims[1])) for j in ranges[1]]
        for i in ranges[0]
    ]
    if color:
        wd.fill((0, 0, 0))
    for obj in objects:
        offscreen = False
        for i in [0, 1]:
            screen_pos = obj.rect[i] - t_offset[i]
            if not (0 < screen_pos + obj.rect[i + 2] and screen_pos < DIMS[i]):
                offscreen = True
        if not offscreen:
            obj.draw()
    player.draw()
    pygame.display.update()


def title_loop(color):
    wd.fill(color)
    wd.blit(
        load_img(join(PATH, "title" + ".png"), 256, 128),
        (WIDTH // 2 - 128, HEIGHT // 3 - 64, 256, 128),
    )
    wd.blit(
        load_img(join(PATH, "play" + ".png"), 128, 64),
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


def scroll(player, look_offset) -> list[float, float]:  # offset amount up, left
    if 0 <= mouse[0] <= WIDTH and 0 <= mouse[1] <= HEIGHT:
        look_offset = [floor((mouse[i] - DIMS[i] // 2) * PERCEPTION) for i in [0, 1]]
    for i in [0, 1]:
        border = [
            player.float_rect[i] + player.float_rect[i + 2] - DIMS[i] + SCROLL[i],
            player.float_rect[i] - SCROLL[i],
        ]
        offset[i] = border[0] if offset[i] <= border[0] else offset[i]
        offset[i] = border[1] if offset[i] >= border[1] else offset[i]
    return offset, look_offset


def main(wd, level_num) -> None:
    print("\n --- RUNNING --- \n")
    global offset, look_offset, t_offset, mouse, gravity, gamestate, data
    data, level, color = process_levels(LEVELS[level_num - 1], BGCOLOR)
    clock = pygame.time.Clock()
    mouse = pygame.mouse.get_pos()
    gamestate = "title"
    gravity = data[2]
    offset, look_offset = [0, 0], [0, 0]
    player = Player(data[0], PLAYER_SIZE)

    run = True
    tick = 0
    while run:
        clock.tick(FPS)
        tick += 1
        mouse = pygame.mouse.get_pos()
        if tick % 600 == 0:
            tick, color = 0, random_color()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        if gamestate == "play":
            offset, look_offset = scroll(player, look_offset)
            t_offset = [offset[i] + look_offset[i] for i in [0, 1]]
            player.loop(FPS, level, data)
            [obj.loop() for obj in level if obj.name == "bouncepad"]
            level_num, data, level, color = obj_interaction(
                player, level_num, level, color, data[0]
            )
            draw(wd, player, level, color)
        elif gamestate == "title":
            title_loop(color)

    print("\n --- QUITTING --- \n")
    pygame.quit()


if __name__ == "__main__":
    main(wd, level_num)
