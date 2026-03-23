import time
from enum import Enum

TIME_FALSE=1
class LoopState(Enum):
    LoopStart = 0
    LoopTrue = 1
    LoopFalse = 2

class StatusGetter:
    def __init__(self, time_loop: float):
        self.time_loop = time_loop
        self.start_loop_true=time.time()
        self.start_loop_false=time.time()
        self.loop_state = LoopState.LoopStart
        self.status=False

    def get_status(self, detection: bool)->bool:
        if self.loop_state == LoopState.LoopStart:
            if detection:
                self.loop_state = LoopState.LoopTrue
                self.start_loop_true = time.time()
        elif self.loop_state == LoopState.LoopTrue:
            if not detection:
                self.loop_state = LoopState.LoopFalse
                self.start_loop_false = time.time()
            if time.time() - self.start_loop_true > self.time_loop:
                self.status = True
        elif self.loop_state == LoopState.LoopFalse:
            if time.time() - self.start_loop_false > TIME_FALSE:
                self.status = False
                self.loop_state = LoopState.LoopStart
        return self.status
