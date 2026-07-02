from __future__ import annotations

import random
import threading
import time

from tqdm.auto import tqdm


BARS = 6
UPDATES = 120
SLEEP = 0.05
LEAVE = True
STAGES = ("proof_generation", "proof_verify", "proof_refine")


def worker(candidate: int) -> None:
    rng = random.Random(1234 + candidate)
    total = rng.choice((60_000, 80_000, 122_880))
    stage = STAGES[candidate % len(STAGES)]
    with tqdm(
        total=total,
        desc=f"P{candidate} {stage}",
        unit="tok",
        position=candidate,
        leave=LEAVE,
        dynamic_ncols=True,
        mininterval=0.2,
    ) as pbar:
        remaining = total
        for step in range(UPDATES):
            if remaining <= 0:
                break
            if step == UPDATES // 3:
                stage = "proof_verify"
                pbar.set_description_str(f"P{candidate} {stage}")
            elif step == (UPDATES * 2) // 3:
                stage = "proof_refine"
                pbar.set_description_str(f"P{candidate} {stage}")
            delta = min(remaining, rng.randint(512, 2048))
            remaining -= delta
            pbar.update(delta)
            time.sleep(SLEEP * rng.uniform(0.5, 1.5))


threads = [
    threading.Thread(target=worker, args=(candidate,), daemon=False)
    for candidate in range(BARS)
]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
