"""Capture README screenshots of the web UI via the Chrome DevTools Protocol.

Plain `chrome --screenshot` cannot click anything and mis-composites after a
scroll, so this drives a real Chrome over CDP instead: it sets a device metric
override (so the whole page is laid out at once, no scrolling), clicks a sample
clip, waits for the prediction to render, and captures exact pixel regions.

    .venv\\Scripts\\python.exe scripts\\capture_ui.py

Requires the backend to be running on port 5001.
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

try:
    import websocket  # websocket-client
except ImportError:
    sys.exit("pip install websocket-client")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
URL = "http://127.0.0.1:5001/"
WIDTH = 1440
TALL = 6400   # viewport height used for capture; see setDeviceMetricsOverride

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    found = shutil.which("chrome") or shutil.which("google-chrome")
    if found:
        return found
    sys.exit("Chrome not found.")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Tab:
    """Minimal CDP client: send a command, wait for its reply."""

    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=60)
        self.id = 0

    def send(self, method: str, **params):
        self.id += 1
        self.ws.send(json.dumps({"id": self.id, "method": method,
                                 "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def eval(self, expr: str):
        r = self.send("Runtime.evaluate", expression=expr, returnByValue=True,
                      awaitPromise=True)
        return r.get("result", {}).get("value")

    def close(self):
        self.ws.close()


def wait_for(tab: Tab, expr: str, what: str, timeout: float = 40) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if tab.eval(expr):
            return
        time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for {what}")


def capture(tab: Tab, name: str, box: dict) -> None:
    shot = tab.send("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                    clip={**box, "scale": 1})
    import base64
    path = OUT / name
    path.write_bytes(base64.b64decode(shot["data"]))
    print(f"  {name}  {box['width']}x{int(box['height'])}")


def section_boxes(tab: Tab) -> dict:
    """Measure real section offsets from the DOM."""
    js = """(() => {
      const off = e => { let y = 0, n = e; while (n) { y += n.offsetTop; n = n.offsetParent } return y };
      const hdr = document.querySelector('header');
      const secs = [...document.querySelectorAll('main > section')];
      const ftr = document.querySelector('footer');
      return JSON.stringify({
        header: [off(hdr), hdr.offsetHeight],
        analyze: [off(secs[0]), secs[0].offsetHeight],
        dash: [off(secs[1]), secs[1].offsetHeight],
        method: [off(secs[2]), secs[2].offsetHeight],
        footer: [off(ftr), ftr.offsetHeight],
        total: document.documentElement.scrollHeight,
      });
    })()"""
    return json.loads(tab.eval(js))


def run(theme: str, click_sample: bool, names: dict) -> None:
    chrome = find_chrome()
    port = free_port()
    profile = Path(tempfile.mkdtemp(prefix="ser-shot-"))
    proc = subprocess.Popen(
        [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
         f"--window-size={WIDTH},{TALL}", "--remote-allow-origins=*",
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url = None
        for _ in range(80):
            try:
                pages = json.loads(urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json", timeout=2).read())
                page = next(p for p in pages if p["type"] == "page")
                ws_url = page["webSocketDebuggerUrl"]
                break
            except Exception:
                time.sleep(0.25)
        if not ws_url:
            raise RuntimeError("Chrome debugging endpoint never came up")

        tab = Tab(ws_url)
        tab.send("Page.enable")
        tab.send("Runtime.enable")
        # The viewport is made as tall as the whole page on purpose. Sections
        # animate in with Framer Motion's whileInView, which never fires for
        # content outside the viewport -- with a short viewport every section
        # below the fold captures blank at opacity 0.
        tab.send("Emulation.setDeviceMetricsOverride", width=WIDTH, height=TALL,
                 deviceScaleFactor=1, mobile=False)
        tab.send("Page.navigate", url=URL)
        wait_for(tab, "document.readyState === 'complete'", "load")
        wait_for(tab, "!!document.querySelector('main > section')", "app render")

        tab.eval(f"localStorage.setItem('ser-theme','{theme}');"
                 f"document.documentElement.setAttribute('data-theme','{theme}')")

        if click_sample:
            wait_for(tab, "document.querySelectorAll('[aria-pressed]').length > 0",
                     "sample buttons")
            # Click the 'Happy' held-out clip through a real DOM click.
            clicked = tab.eval("""(() => {
              const btns = [...document.querySelectorAll('[aria-pressed]')];
              const b = btns.find(x => x.innerText.trim().startsWith('Happy'));
              if (!b) return false; b.click(); return true;
            })()""")
            if not clicked:
                raise RuntimeError("could not find the Happy sample button")
            wait_for(tab, "document.querySelectorAll('[role=meter]').length === 8",
                     "prediction result", timeout=60)

        # Let springs settle so bars are at their final width.
        time.sleep(2.5)
        b = section_boxes(tab)
        print(f"[{theme}] sections: {b}")
        OUT.mkdir(parents=True, exist_ok=True)

        for key, fname in names.items():
            if key == "hero":
                top, h = 0, b["header"][0] + b["header"][1]
            else:
                top, h = b[key]
            capture(tab, fname, {"x": 0, "y": top, "width": WIDTH, "height": h})
        tab.close()
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    print("Capturing dark theme...")
    run("dark", True, {
        "hero": "ui-hero.png",
        "analyze": "ui-analyze.png",
        "dash": "ui-dashboard.png",
        "method": "ui-method.png",
    })
    print("Capturing light theme...")
    run("light", True, {"analyze": "ui-analyze-light.png"})
    print("done ->", OUT)
