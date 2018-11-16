import keyboard as kb
import random

for _ in range(1000):
    let = random.choice(["w", "a", "s", "d"])
    kb.send(let)
