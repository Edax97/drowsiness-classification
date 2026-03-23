import time
import serial
from keyboard_listener import KeyboardListener

class SerialSender:
    def __init__(self, port, baudrate):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1
        )
        self.last_time_dsm = time.time()

    def send_serial(self, data_str: str):
        if not self.ser.is_open:
            return
        bytes_data = f"{data_str}\r\n".encode()
        print(f"Sending data: {data_str}")
        time.sleep(0.01)
        try:
            self.ser.write(bytes_data)
            self.ser.flush()
        except Exception as e:
            print(f"Serial error: {e}")

    def close_serial(self):
        self.ser.close()

class ScannerListener:
    def __init__(self):
        self.listener = KeyboardListener(self.on_press, self.on_release)
        self.sending = False
        self.data = ""
        self.data_full = ""

        self.listener.start()

    def get_data(self):
        if not self.sending:
            return ""
        self.sending = False
        return self.data_full

    def on_press(self, key):
        #print("Key {0} pressed".format(key))
        if len(key) == 1:
            self.data += key
            self.sending = False
        elif key == "ENTER" and not self.sending:
            self.sending = True
            self.data_full = self.data
            self.data = ""

    def on_release(self, key):
        if key == "ESC":
            return False

    def join(self):
        self.listener.join()

