import json
import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request


def _pick_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    return port


def _http_get(url: str, timeout_sec: float = 2.0):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return resp.status, resp.read()


def _http_post_form(url: str, form: dict, timeout_sec: float = 2.0):
    data = urllib.parse.urlencode(form).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return resp.status, resp.read()


def _terminate_and_collect(proc: subprocess.Popen) -> str:
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        out, _ = proc.communicate(timeout=2.0)
        return out or ""
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        return ""


class TestStreamWebUI(unittest.TestCase):
    def test_webui_update_texts_and_latest_image(self):
        port = _pick_free_port()
        base = f"http://127.0.0.1:{port}"

        env = os.environ.copy()
        env["LALIU_DUMMY"] = "1"
        env["LALIU_SAMPLE_INTERVAL_SEC"] = "1"

        python_exe = sys.executable
        if os.path.exists(os.path.join(".venv", "bin", "python3")):
            python_exe = os.path.join(".venv", "bin", "python3")

        proc = subprocess.Popen(
            [python_exe, "test_stream.py", "--host", "127.0.0.1", "--port", str(port)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            deadline = time.time() + 10.0
            last_err = None
            while time.time() < deadline:
                try:
                    status, body = _http_get(f"{base}/status", timeout_sec=1.0)
                    if status == 200:
                        break
                except Exception as e:
                    last_err = e
                time.sleep(0.2)
            else:
                out = _terminate_and_collect(proc)
                raise AssertionError(f"服务未在超时内启动: {last_err}\n{out}")

            status, _ = _http_post_form(
                f"{base}/set_texts",
                {"texts": "a\nb\nc\n"},
                timeout_sec=2.0,
            )
            self.assertEqual(status, 200)

            status, body = _http_get(f"{base}/texts", timeout_sec=2.0)
            self.assertEqual(status, 200)
            payload = json.loads(body.decode("utf-8"))
            self.assertEqual(payload["texts"], ["a", "b", "c"])

            deadline = time.time() + 8.0
            while time.time() < deadline:
                try:
                    status, body = _http_get(f"{base}/latest.jpg", timeout_sec=2.0)
                    if status == 200 and len(body) > 1024:
                        break
                except urllib.error.HTTPError:
                    pass
                time.sleep(0.5)
            else:
                out = _terminate_and_collect(proc)
                raise AssertionError(f"未在超时内生成最新图片\n{out}")
        finally:
            _terminate_and_collect(proc)


if __name__ == "__main__":
    unittest.main()
