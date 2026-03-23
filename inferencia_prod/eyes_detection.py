from mediapipe.tasks.python.components.containers.category import Category


def get_eyes_(face_bs: list[Category]):
    left_blink = float(int(face_bs[9].score * 1000)) / 10
    right_blink = float(int(face_bs[10].score * 1000)) / 10
    # text_overlay(out, f"Blink: {left_blink}, {right_blink}", (20, 40))
