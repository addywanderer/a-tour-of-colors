import pygame
from level import LEVELS
from os import listdir
from os.path import isfile, join
from random import randint
from math import floor, ceil

pygame.init()

CAPTION = "Idle Marbles"
ICON = "goal"

COLORS = [
    [[150, 255], [0, 50], [130, 200]],
    [[0, 50], [200, 255], [200, 255]],
    [[0, 50], [200, 255], [50, 100]],
    [[200, 255], [30, 70], [30, 70]],
    [[200, 255], [200, 255], [0, 50]],
]

WIDTH = 1000
HEIGHT = 800
DIMS = [WIDTH, HEIGHT]
FPS = 60

track_size = [3, 3]
num_balls = 1
color_range = COLORS[1]

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


def title_loop(color):
    gamestate = "title"
    wd.fill(color)
    wd.blit(
        load_img(join(PATH, "title" + ".png"), 256, 128),
        (WIDTH // 2 - 128, HEIGHT // 3 - 64, 256, 128),
    )
    wd.blit(
        load_img(join(PATH, "play" + ".png"), 128, 64),
        (WIDTH // 2 - 64, 2 * HEIGHT // 3 - 32, 128, 64),
    )
    print(mouse, pygame.mouse.get_pressed(num_buttons=3)[0])
    if (
        WIDTH // 2 - 64 < mouse[0] < WIDTH // 2 + 64
        and 2 * HEIGHT // 3 - 32 < mouse[1] < 2 * HEIGHT // 3 + 32
        and pygame.mouse.get_pressed(num_buttons=3)[0]
    ):
        print("got")
        gamestate = "play"
    pygame.display.update()
    return gamestate


class Object(pygame.sprite.Sprite):
    def __init__(self, space, name, path=None, angle=0) -> None:  # space = x, y, w, h
        super().__init__()

        self.rect, self.name, self.path = pygame.Rect(*space), name, path
        if path:
            self.image = load_img(
                    join(PATH, "objects", path + ".png"), space[2], space[3]
                )
            self.image = rotate_image([self.image], angle)[0]
        else:
            self.image = pygame.surface.Surface((self.rect.w, self.rect.h))
        self.update_mask()

    def draw(self) -> None:
        pos = [self.rect.x, self.rect.y]
        wd.blit(self.image, tuple(pos[i] for i in [0, 1]))

    def update_mask(self) -> None:
        self.mask = pygame.mask.from_surface(self.image)


class Colored_Rect(Object):
    def __init__(self, space):
        super().__init__(space, "block")
        self.color = random_color()
        self.bord = tuple(abs(i - 10) for i in self.color)
        self.cooldown = 50

    def draw(self):
        if self.cooldown == 0:
            self.cooldown = 50
            self.color = random_color()
            self.bord = tuple(abs(i - 10) for i in self.color)
        pygame.draw.rect(
            wd,
            self.bord,
            self.rect
        )
        pygame.draw.rect(
            wd,
            self.color,
            [self.rect[i] + 4 for i in [0, 1]]
            + [self.rect.w - 8, self.rect.h - 8],
        )


class Track:
    def __init__(self, track_size):
        self.size = track_size
        self.sides = [Colored_Rect([WIDTH//2 - 2 * 64 + 48, HEIGHT//2 - 64, 64 - 48, 128888888888888888888888889999988888888888888888])]
    
    def draw(self):
        for side in self.sides:
            side.draw()


class Ball:
    def __init__(self, pos):
        self.pos = pos
        self.rect = pygame.rect.Rect(pos[0], pos[1], 30, 30)
    
    def draw(self):
        pygame.draw.circle(wd, random_color(), self.pos, 15.0)


def draw(balls, objects, track): #upgrades?
    wd.fill((0, 0, 0))
    track.draw()
    for ball in balls:
        ball.draw()
    for obj in objects:
        obj.draw()
    pygame.display.update()


def main(wd) -> None:
    print("\n --- RUNNING --- \n")
    global mouse, balls, objects, track
    balls = [Ball([100, 100])] * num_balls
    objects = []
    track = Track(track_size)
    clock = pygame.time.Clock()
    gamestate = "title"

    run = True
    tick = 0
    while run:
        clock.tick(FPS)
        mouse = pygame.mouse.get_pos()
        if tick % 600 == 0:
            tick, color = 0, random_color()
        tick += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        if gamestate == "play":
            print("wot")
            draw(balls, objects, track)
        elif gamestate == "title":
            gamestate = title_loop(color)

    print("\n --- QUITTING --- \n")
    pygame.quit()


if __name__ == "__main__":
    main(wd)
