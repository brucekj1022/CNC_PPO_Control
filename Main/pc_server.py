import json
import signal
import socket
import sys
import threading
import time

import numpy as np
#位址設定
HOST = "0.0.0.0"
PORT = 5005
RUN = True

#CMD關閉運行
def _on_sigint(*_):
    global RUN
    RUN = False
    print("\n[INFO] Ctrl+C received, shutting down...")

def _input_watcher():
    global RUN
    try:
        while RUN:
            s = input().strip().lower()
            if s in ("q", "quit", "exit"):
                RUN = False
                print("[INFO] Quit command received, shutting down...")
                break
    except Exception:
        pass

signal.signal(signal.SIGINT, _on_sigint)

#輔助用
def cal_data_length(s: str) -> str:
    """
    計算字串的 UTF-8 位元組長度並轉成 4 位數字字串。
    例如：138 個位元組 → "0138"
    不加換行。
    """
    nbytes = len(s.encode("utf-8"))   # 真實位元組長度
    if nbytes > 9999:
        raise ValueError(f"payload too long: {nbytes} bytes (max 9999)")
    return nbytes+2

def array_to_str(arr: np.ndarray, float_fmt=".3f") -> str:
    """將 numpy array 轉成逗號分隔字串，例如 '1.000,2.000,3.000'。"""
    a = np.asarray(arr, dtype=np.float32).ravel()
    return ",".join(format(float(x), float_fmt) for x in a)

def str_to_array(s: str) -> np.ndarray:
    """將逗號分隔字串轉成 numpy array(float32)，例如 '1.000,2.000,3.000' → np.array([1.0,2.0,3.0])."""
    s = s.strip().strip("[]").replace(";", ",")
    tokens = [t for t in s.split(",") if t.strip()]
    return np.array([float(t) for t in tokens], dtype=np.float32)
 
#TCP相關
def create_server(host, port, backlog=8, accept_timeout=None):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(backlog)
    if accept_timeout is not None:
        srv.settimeout(accept_timeout)
    return srv

def recv(conn):
    REPLY_DEADLINE_S = 10.0
    TRY_INTERVAL_S   = 0.05
    try:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(2.0)
        f = conn.makefile("r", encoding="utf-8", newline="\n")

        t0 = time.time()
        deadline = t0 + REPLY_DEADLINE_S
        while True:
            now = time.time()
            if now >= deadline:
                raise TimeoutError(f"no reply within {REPLY_DEADLINE_S:.1f}s")

            try:
                receive = f.readline()
            except socket.timeout:
                time.sleep(TRY_INTERVAL_S)
                continue

            if receive == "":
                time.sleep(TRY_INTERVAL_S)
                continue

            receive = receive.rstrip("\n")
            return str_to_array(receive)

    except TimeoutError as e:
        print(f"[ERROR] {e}")
        return None
    except ConnectionResetError:
        print("[INFO] client reset connection")
        return None
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

def send(conn, data):
    def send_line(sock, obj):
        if isinstance(obj, str):
            data = (obj + "\n").encode("utf-8")
        else:
            data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
        sock.sendall(data)
    try:
        if not isinstance(data, str):
            data = array_to_str(data)
        nbytes = cal_data_length(data)
        send_line(conn, f"{nbytes:04d}")
        send_line(conn, data)
        return True

    except TimeoutError as e:
        print(f"[ERROR] {e}")
        return False
    except ConnectionResetError:
        print("[INFO] client reset connection")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def main():
    global RUN
    print(f"[INFO] Listening on {HOST}:{PORT}")
    print("[INFO] Press Ctrl+C or type 'q' then Enter to exit.\n")
    PC_MESSAGE = "0.000,0.000,0.000,0.000,635.300,-3394.000,6732.000,-5875.000,1905.000,0.000,0.000,0.000,0.000,0.000,-0.063,0.596,-2.019,3.443,-2.955,1.000"

    t = threading.Thread(target=_input_watcher, daemon=True)
    t.start()

    srv=create_server(HOST, PORT)
    while RUN:
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            continue
        except KeyboardInterrupt:
            break

        t0=time.time()
        receive=recv(conn)
        print(f"[LabView] {receive} ")
        send(conn, PC_MESSAGE)
        rtt_ms = (time.time() - t0) * 1000.0
        print(f"[PC] {PC_MESSAGE} (RTT={rtt_ms:.3f} ms)\n")
        conn.close()

    print("[INFO] Server exiting.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted (Ctrl+C). Bye!")
        sys.exit(0)
