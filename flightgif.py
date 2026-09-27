"""Render a recorded flight (run.episode(record=[...])) as an animated GIF.

Each frame: a high chase view of the aircraft, the onboard camera frame the model
was given (when it was given one), the current judgment, and a map of the course
with the aircraft's position. Rendered after the flight, so recording never slows
a real-time run.
"""
import io
import numpy as np
from PIL import Image, ImageDraw

CHASE_W, CHASE_H = 480, 360
PANEL_W = 256


def _font():
    try:
        from PIL import ImageFont
        return ImageFont.load_default(size=13)
    except Exception:
        return None


def make_gif(record, course, seed, title, directory=".", every=2, frame_ms=100, trim_after_s=8.0):
    """`every`: keep one snapshot in `every` (snapshots are 0.2 s apart, so every=2 at 100 ms
    per frame plays at 4x). Stops `trim_after_s` after the aircraft last made progress."""
    import mujoco
    import courses
    c = courses.make(course, seed)
    m = mujoco.MjModel.from_xml_path(c.write(directory))
    m.vis.map.zfar = 50.0
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, CHASE_H, CHASE_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = 7.5, -32.0

    # the map strip, with the rover's path, from courses.render
    mp = Image.open(io.BytesIO(courses.render(course, seed, directory)))
    map_w = CHASE_W + PANEL_W
    map_h = int(mp.height * map_w / mp.width)
    mp = mp.resize((map_w, map_h))
    span = c.end_x + 4.0
    to_map = lambda x, y: (map_w / 2 + (x - span / 2) / span * map_w, map_h / 2 - y / span * map_w)  # noqa: E731

    # trim a stuck ending: stop a few seconds after the last real progress
    xs = [s["qpos"][0] for s in record]
    best, last_gain = -1e9, 0
    for k, x in enumerate(xs):
        if x > best + 0.5:
            best, last_gain = x, k
    end = min(len(record), last_gain + int(trim_after_s / 0.2) + 1)

    font = _font()
    frames = []
    trail = []
    for snap in record[:end]:
        trail.append(snap["qpos"][:2].copy())
    for k in range(0, end, every):
        snap = record[k]
        d.qpos[:] = snap["qpos"]
        d.mocap_pos[:] = snap["mocap_pos"]
        d.mocap_quat[:] = snap["mocap_quat"]
        mujoco.mj_forward(m, d)
        m.vis.global_.fovy = 50.0
        cam.lookat[:] = snap["qpos"][:3]
        cam.azimuth = np.rad2deg(snap["yaw"])
        r.update_scene(d, cam)
        chase = Image.fromarray(r.render())

        canvas = Image.new("RGB", (map_w, CHASE_H + map_h), (18, 20, 24))
        canvas.paste(chase, (0, 0))
        dr = ImageDraw.Draw(canvas)
        if snap["rgb"] is not None:
            eye = Image.fromarray(snap["rgb"]).resize((PANEL_W, PANEL_W * 3 // 4))
            canvas.paste(eye, (CHASE_W, 0))
            dr.text((CHASE_W + 6, 4), "what Laya sees", fill=(255, 255, 255), font=font)
        j = snap["judg"]
        live = j.get("source") in ("jev", "laya") and (j.get("age_s") or 9) < 1.5
        y0 = PANEL_W * 3 // 4 + 8
        lines = [
            ("t = %5.1f s   x = %5.1f m" % (snap["t"], snap["qpos"][0]), (230, 230, 230)),
            ("answer: %s" % (j.get("maneuver") if live else "-"), (255, 210, 80) if live else (150, 150, 150)),
            ("  confidence %.2f" % (j.get("confidence") or 0), (200, 200, 200)),
            ("  risk %.2f   lost %.2f" % (j.get("risk") or 0, j.get("target_truly_lost") or 0), (200, 200, 200)),
            ("CLIMBING" if snap["climbing"] else "", (120, 220, 255)),
            ("REFLEX (code override)" if snap["reflex"] else "", (255, 110, 90)),
            ("rover in view" if snap["target_visible"] else "rover not in view",
             (140, 230, 140) if snap["target_visible"] else (230, 140, 140)),
            ("collisions: %d" % snap["hits"], (200, 200, 200)),
        ]
        for n, (txt, col) in enumerate(lines):
            if txt:
                dr.text((CHASE_W + 8, y0 + 18 * n), txt, fill=col, font=font)
        dr.rectangle([0, 0, CHASE_W - 1, 22], fill=(0, 0, 0))
        dr.text((6, 4), title, fill=(255, 255, 255), font=font)

        canvas.paste(mp, (0, CHASE_H))
        pts = [to_map(x, y) for x, y in trail[: k + 1]]
        if len(pts) > 1:
            dr.line([(u, CHASE_H + v) for u, v in pts], fill=(80, 200, 255), width=2)
        u, v = to_map(*snap["qpos"][:2])
        dr.ellipse([u - 5, CHASE_H + v - 5, u + 5, CHASE_H + v + 5], fill=(80, 200, 255), outline=(255, 255, 255))
        frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=128))

    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:], duration=frame_ms, loop=0, optimize=True)
    return buf.getvalue()
