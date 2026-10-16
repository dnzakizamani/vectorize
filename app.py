#!/usr/bin/env python3
"""
Image to SVG Web Server
Provides Web UI + API to convert raster images (PNG, JPG, WEBP) to SVG.
Uses vtracer if installed, otherwise built-in PIL contour tracer.
"""
import os
import io
import json
import math
import cgi
from http.server import HTTPServer, SimpleHTTPRequestHandler
from collections import defaultdict
from PIL import Image

try:
    import vtracer
    HAS_VTRACER = True
except ImportError:
    HAS_VTRACER = False

def distance_sq(c1, c2):
    return (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2

def rdp(points, epsilon):
    if len(points) < 3:
        return points
    first, last = points[0], points[-1]
    dx = last[0] - first[0]
    dy = last[1] - first[1]
    line_len_sq = dx*dx + dy*dy
    max_dist = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        p = points[i]
        if line_len_sq == 0:
            dist = math.hypot(p[0] - first[0], p[1] - first[1])
        else:
            num = abs(dy * p[0] - dx * p[1] + last[0]*first[1] - last[1]*first[0])
            dist = num / math.sqrt(line_len_sq)
        if dist > max_dist:
            max_dist = dist
            index = i
    if max_dist > epsilon:
        left = rdp(points[:index+1], epsilon)
        right = rdp(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [first, last]

def trace_contours(mask, width, height):
    edges = defaultdict(list)
    for y in range(height):
        for x in range(width):
            val = mask[y * width + x]
            if not val:
                continue
            if y == 0 or not mask[(y - 1) * width + x]:
                edges[(x, y)].append((x + 1, y))
            if y + 1 == height or not mask[(y + 1) * width + x]:
                edges[(x + 1, y + 1)].append((x, y + 1))
            if x == 0 or not mask[y * width + x - 1]:
                edges[(x, y + 1)].append((x, y))
            if x + 1 == width or not mask[y * width + x + 1]:
                edges[(x + 1, y)].append((x + 1, y + 1))

    visited_edges = set()
    loops = []
    for start_v in list(edges.keys()):
        for next_v in edges[start_v]:
            edge = (start_v, next_v)
            if edge in visited_edges:
                continue
            loop = [start_v]
            curr = next_v
            visited_edges.add(edge)
            while curr != start_v:
                loop.append(curr)
                candidates = edges[curr]
                next_cand = None
                for c in candidates:
                    if (curr, c) not in visited_edges:
                        next_cand = c
                        break
                if next_cand is None:
                    break
                visited_edges.add((curr, next_cand))
                curr = next_cand
            if len(loop) >= 6 and curr == start_v:
                loops.append(loop)
    return loops

def trace_image_pil(image_bytes, num_colors=4, epsilon=1.2):
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = im.size

    # Auto scale down if huge for fast web response
    max_dim = 1000
    if max(W, H) > max_dim:
        scale = max_dim / max(W, H)
        W, H = int(W * scale), int(H * scale)
        im = im.resize((W, H), Image.Resampling.BILINEAR)

    # Quantize colors
    q_im = im.quantize(colors=num_colors, method=Image.Quantize.FASTOCTREE).convert("RGB")
    colors_hist = q_im.getcolors(maxcolors=256)
    colors_hist.sort(key=lambda x: x[0], reverse=True)

    pixels = list(q_im.getdata())
    unique_colors = [c[1] for c in colors_hist]

    # Find dominant background color (most frequent border pixel)
    border_pixels = []
    for x in range(W):
        border_pixels.append(pixels[x])
        border_pixels.append(pixels[(H - 1) * W + x])
    for y in range(H):
        border_pixels.append(pixels[y * W])
        border_pixels.append(pixels[y * W + W - 1])
    bg_color = max(set(border_pixels), key=border_pixels.count)

    svg_paths = []
    # Process from larger area to smaller
    for cnt, col in colors_hist:
        if col == bg_color and (col[0] > 230 and col[1] > 230 and col[2] > 230):
            continue  # Skip pure white/near-white backgrounds

        hex_color = "#{:02x}{:02x}{:02x}".format(col[0], col[1], col[2])
        mask = [1 if p == col else 0 for p in pixels]
        loops = trace_contours(mask, W, H)

        subpaths = []
        for loop in loops:
            simplified = rdp(loop + [loop[0]], epsilon)
            if len(simplified) < 4:
                continue
            d_parts = [f"M {pt[0]} {pt[1]}" for pt in simplified]
            d_parts.append("Z")
            subpaths.append(" ".join(d_parts))

        if subpaths:
            svg_paths.append((hex_color, " ".join(subpaths)))

    svg_out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '  <title>Vectorized Image</title>'
    ]
    for color, d_str in svg_paths:
        svg_out.append(f'  <path fill="{color}" fill-rule="evenodd" d="{d_str}"/>')
    svg_out.append('</svg>')
    return "\n".join(svg_out)

def convert_image(image_bytes, num_colors=6, epsilon=1.2):
    if HAS_VTRACER:
        try:
            return vtracer.convert_raw_image_to_svg(
                image_bytes,
                colormode='color',
                hierarchical='stacked',
                mode='spline',
                filter_speckle=4,
                color_precision=6,
                layer_difference=16,
                corner_threshold=60,
                length_threshold=4.0,
                max_iterations=10,
                splice_threshold=45,
                path_precision=2
            )
        except Exception as e:
            print("VTracer failed, falling back to PIL:", e)

    return trace_image_pil(image_bytes, num_colors=num_colors, epsilon=epsilon)

class TracerHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/trace':
            content_type = self.headers.get('Content-Type', '')
            if 'multipart/form-data' not in content_type:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Expected multipart/form-data"}')
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': content_type,
                }
            )

            if 'file' not in form:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "No file uploaded"}')
                return

            file_item = form['file']
            image_data = file_item.file.read()
            num_colors = int(form.getvalue('colors', 6))
            smoothness = float(form.getvalue('smoothness', 1.2))

            try:
                svg_content = convert_image(image_data, num_colors=num_colors, epsilon=smoothness)
                resp = json.dumps({"svg": svg_content}).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except Exception as e:
                err_resp = json.dumps({"error": str(e)}).encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(err_resp)
        else:
            self.send_response(404)
            self.end_headers()

def run(port=8080):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = HTTPServer(('0.0.0.0', port), TracerHandler)
    engine = "vtracer (high-speed bezier)" if HAS_VTRACER else "built-in PIL contour tracer"
    print(f"🚀 Image to SVG Server running on http://localhost:{port}")
    print(f"Engine: {engine}")
    server.serve_forever()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    run(port)
