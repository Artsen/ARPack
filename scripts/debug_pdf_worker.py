from __future__ import annotations
import sys
import traceback
import multiprocessing as mp

def worker(path: str, q: mp.Queue):
    try:
        import fitz
        doc = fitz.open(path)
        parts = [p.get_text("text") for p in doc]
        txt = "\n".join(parts)
        q.put(("ok", len(txt)))
    except Exception as e:
        tb = traceback.format_exc()
        q.put(("err", tb))

def main():
    if len(sys.argv) < 2:
        print("Usage: debug_pdf_worker.py <pdf_path> [timeout_sec]")
        return
    path = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    q = mp.Queue()
    p = mp.Process(target=worker, args=(path, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        print(f"child timed out after {timeout}s")
    else:
        try:
            status, payload = q.get_nowait()
            if status == 'ok':
                print(f"child ok, chars={payload}")
            else:
                print("child error:\n", payload)
        except Exception as e:
            print("child exited but no message; maybe crashed or queue empty")

if __name__ == '__main__':
    main()
