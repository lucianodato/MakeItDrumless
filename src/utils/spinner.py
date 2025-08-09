import threading
import time

def spinner(msg, stop_event):
    spinner_chars = "|/-\\"
    idx = 0
    while not stop_event.is_set():
        print(f"\r{msg} {spinner_chars[idx % len(spinner_chars)]}", end="", flush=True)
        idx += 1
        time.sleep(0.1)
    print("\r" + " " * (len(msg) + 2) + "\r", end="", flush=True)

