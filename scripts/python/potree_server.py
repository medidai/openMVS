#!/usr/bin/env python3
"""
Simple HTTP server for viewing Potree 2.0 point clouds exported by OpenMVS.

Usage:
    python potree_server.py <potree_directory> [--port PORT] [--browser]

The directory should contain: metadata.json, hierarchy.bin, octree.bin
"""

import argparse
import http.server
import os
import sys
import webbrowser
import threading

VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenMVS Potree Viewer</title>
<script src="https://cdn.jsdelivr.net/gh/potree/potree@develop/build/potree/potree.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/potree/potree@develop/build/potree/potree.css">
<style>
    body { margin: 0; padding: 0; overflow: hidden; }
    #potree_render_area { width: 100vw; height: 100vh; }
</style>
</head>
<body>
<div id="potree_render_area"></div>
<script>
const viewer = new Potree.Viewer(document.getElementById("potree_render_area"));
viewer.setEDLEnabled(true);
viewer.setFOV(60);
viewer.setPointBudget(2_000_000);
viewer.setBackground("gradient");
viewer.loadSettingsFromURL();

Potree.loadPointCloud("./data/metadata.json", "pointcloud", function(e) {
    viewer.scene.addPointCloud(e.pointcloud);
    const material = e.pointcloud.material;
    material.size = 1;
    material.pointSizeType = Potree.PointSizeType.ADAPTIVE;
    material.activeAttributeName = "rgba";
    viewer.fitToScreen();
});
</script>
</body>
</html>
"""


class PotreeHandler(http.server.SimpleHTTPRequestHandler):
    """Handler that serves the viewer HTML at root and potree data under /data/."""

    def __init__(self, *args, potree_dir: str = "", **kwargs):
        self.potree_dir = potree_dir
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(VIEWER_HTML)))
            self.end_headers()
            self.wfile.write(VIEWER_HTML.encode("utf-8"))
        elif self.path.startswith("/data/"):
            rel_path = self.path[len("/data/"):]
            file_path = os.path.join(self.potree_dir, rel_path)
            if os.path.isfile(file_path):
                self.send_response(200)
                content_type = "application/json" if file_path.endswith(".json") else "application/octet-stream"
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(os.path.getsize(file_path)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, f"File not found: {rel_path}")
        else:
            self.send_error(404)

    def log_message(self, _format, *_args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Serve Potree 2.0 point cloud for web viewing")
    parser.add_argument("directory", help="Path to Potree output directory (containing metadata.json)")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--browser", action="store_true", help="Open browser automatically")
    args = parser.parse_args()

    potree_dir = os.path.abspath(args.directory)
    metadata_path = os.path.join(potree_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        print(f"Error: {metadata_path} not found. Is this a valid Potree directory?", file=sys.stderr)
        sys.exit(1)

    handler = lambda *a, **kw: PotreeHandler(*a, potree_dir=potree_dir, **kw)
    server = http.server.HTTPServer(("", args.port), handler)

    url = f"http://localhost:{args.port}"
    print(f"Serving Potree viewer at {url}")
    print(f"Point cloud data: {potree_dir}")
    print("Press Ctrl+C to stop")

    if args.browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
        server.server_close()


if __name__ == "__main__":
    main()
