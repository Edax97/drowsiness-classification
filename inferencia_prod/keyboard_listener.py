import evdev
from evdev import categorize, ecodes
import threading

class KeyboardListener:
    def __init__(self, on_press=None, on_release=None):
        self.on_press = on_press
        self.on_release = on_release
        self._threads = []

    def _get_keyboards(self):
        devices = [evdev.InputDevice(p) for p in evdev.list_devices()]
        return [d for d in devices if ecodes.EV_KEY in d.capabilities()]

    def _get_char(self, keycode:str):
        return keycode.removeprefix("KEY_")

    def _listen(self, device):
        for event in device.read_loop():
            if event.type == ecodes.EV_KEY:
                key = categorize(event)
                if event.value == 1 and self.on_press:   # key down
                    self.on_press(self._get_char(key.keycode))
                elif event.value == 0 and self.on_release: # key up
                    self.on_release(self._get_char(key.keycode))

    def start(self):
        for kb in self._get_keyboards():
            t = threading.Thread(target=self._listen, args=(kb,), daemon=True)
            t.start()
            self._threads.append(t)

    def join(self):
        for t in self._threads:
            t.join()

