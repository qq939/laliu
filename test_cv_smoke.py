import os
import subprocess
import sys
import time
import unittest


class TestCVSmoke(unittest.TestCase):
    def test_cv_generates_outputs(self):
        repo_dir = os.path.dirname(__file__)
        streaming_dir = os.path.join(repo_dir, "streaming", f"test_cv_{os.getpid()}")
        output_dir = os.path.join(repo_dir, "runs", "stream", f"test_cv_{os.getpid()}")

        env = os.environ.copy()
        env["LALIU_DUMMY"] = "1"
        env["LALIU_CV_DISABLE_UI"] = "1"
        env["LALIU_SAMPLE_INTERVAL_SEC"] = "0.2"
        env["LALIU_STREAMING_DIR"] = streaming_dir
        env["LALIU_OUTPUT_DIR"] = output_dir

        python_exe = sys.executable
        if os.path.exists(os.path.join(repo_dir, ".venv", "bin", "python3")):
            python_exe = os.path.join(repo_dir, ".venv", "bin", "python3")

        proc = subprocess.Popen(
            [python_exe, os.path.join(repo_dir, "test_cv.py"), "--no-ui"],
            cwd=repo_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            deadline = time.time() + 6.0
            last = os.path.join(streaming_dir, "last.jpg")
            last_processed = os.path.join(streaming_dir, "last-processed.jpg")
            lbl_json = os.path.join(output_dir, "labels", "last-labels.json")
            ultra_img = os.path.join(output_dir, "ultralytics", "predict", "last.jpg")
            ultra_lbl = os.path.join(output_dir, "ultralytics", "predict", "labels", "last.txt")

            while time.time() < deadline:
                ok = True
                for p in [last, last_processed, ultra_img]:
                    ok = ok and os.path.exists(p) and os.path.getsize(p) > 1024
                ok = ok and os.path.exists(lbl_json) and os.path.getsize(lbl_json) > 10
                ok = ok and os.path.exists(ultra_lbl)
                if ok:
                    break
                time.sleep(0.1)
            else:
                out = ""
                try:
                    out, _ = proc.communicate(timeout=1.0)
                except Exception:
                    out = ""
                raise AssertionError(f"test_cv 未在超时内生成输出\n{out}")
        finally:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.communicate(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.communicate(timeout=1.0)
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
