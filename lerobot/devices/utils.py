import time


def fps_wait(fps: int, tm: float):
    while (time.time() - tm) < (1 / fps):
        time.sleep(0.0005)
