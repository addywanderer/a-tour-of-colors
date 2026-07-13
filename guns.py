# Per-gun stats: each entry describes a gun's properties used by the game.
# Keys: name, damage, reload (frames), bullet_speed, bullet_image, gun_image
BULLET_RADII = {
    "ammo": 8,
    "shell": 12,
    "round": 10,
    "bb": 6,
}

GUN_STATS = {
    "debug_gun": {
        "name": "debug_gun",
        "damage": float("inf"),
        "reload": 0,
        "bullet_speed": 10,
        "recoil": 0,
        "bullet_image": "ammo",
        "gun_image": "pistol",
    },
    "pistol": {
        "name": "pistol",
        "damage": 10,
        "reload": 1,
        "bullet_speed": 10,
        "recoil": 1,
        "bullet_image": "ammo",
        "gun_image": "pistol",
    },
    "shotgun": {
        "name": "shotgun",
        "damage": 6,
        "reload": 30,
        "bullet_speed": 40,
        "recoil": 80,
        "bullet_image": "shell",
        "gun_image": "shotgun",
    },
    "rifle": {
        "name": "rifle",
        "damage": 20,
        "reload": 20,
        "bullet_speed": 80,
        "recoil": 40,
        "bullet_image": "round",
        "gun_image": "rifle",
    },
    "smg": {
        "name": "smg",
        "damage": 6,
        "reload": 5,
        "bullet_speed": 60,
        "recoil": 10,
        "bullet_image": "bb",
        "gun_image": "smg",
    },
}
