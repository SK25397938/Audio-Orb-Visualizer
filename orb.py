# orb_ncs_glow_clean.py — Circular audio spectrum visualizer (with 3D Depth Illusion + Gradient Shadows)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import queue, sys
from matplotlib.colors import hsv_to_rgb

# ---------- Configuration ----------
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
DEVICE_INDEX = 16  # Change this to your loopback device index
NUM_BARS = 120
ALPHA = 0.1
MOTION_BLUR = True
BLUR_FRAMES = 5
NUM_SHADOW_RINGS = 8  # how many gradient shadow rings inside orb
# -----------------------------------

q = queue.Queue()
smoothed_fft = np.zeros(NUM_BARS)
smoothed_bass = 0.0
circle_history = []
audio_buffer = []

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

try:
    stream = sd.InputStream(device=DEVICE_INDEX, channels=2, samplerate=SAMPLE_RATE,
                            blocksize=BLOCK_SIZE, callback=audio_callback, dtype="float32")
    stream.start()
except Exception as e:
    print("Couldn't start audio stream:", e)
    sys.exit(1)

# --- Setup figure ---
plt.rcParams["toolbar"] = "none"
theta_circle = np.linspace(0, 2*np.pi, 500)

fig, ax = plt.subplots(figsize=(12, 12))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.axis("off")
ax.set_aspect("equal")

# expanded limits
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# fullscreen
figManager = plt.get_current_fig_manager()
try: figManager.full_screen_toggle()
except: pass

# --- Glowing Circle ---
circle_lines = [ax.plot([], [], lw=6 - g,
                        color=(0.2,1.0,1.0,0.12*(g+1)))[0] for g in range(5)]

# --- Motion Blur Circles ---
blur_circles = []
if MOTION_BLUR:
    for i in range(BLUR_FRAMES):
        alpha = 0.05 * (1 - i/BLUR_FRAMES)
        blur = ax.plot([], [], lw=5, color=(0.2,1.0,1.0,alpha))[0]
        blur_circles.append(blur)

# --- Spectrum Bars ---
angles = np.linspace(0, 2*np.pi, NUM_BARS, endpoint=False)
bars = [ax.plot([], [], lw=2, color="cyan")[0] for _ in range(NUM_BARS)]

# --- Reflection Bars ---
reflection_bars = [ax.plot([], [], lw=2, color="cyan", alpha=0.3)[0] for _ in range(NUM_BARS)]

# --- Reflection Circle ---
reflection_circle = ax.plot([], [], lw=2, color=(0.2, 1.0, 1.0, 0.25))[0]

# --- Depth Layers ---
depth_layers = []
NUM_DEPTH = 4
for i in range(NUM_DEPTH):
    alpha = 0.15 * (1 - i/NUM_DEPTH)
    depth = ax.plot([], [], lw=2, color=(0.1, 0.7, 1.0, alpha))[0]
    depth_layers.append(depth)

# --- Gradient Shadows (rings inside orb) ---
shadow_rings = []
for i in range(NUM_SHADOW_RINGS):
    alpha = 0.25 * (1 - i/NUM_SHADOW_RINGS)  # fading effect
    ring = ax.plot([], [], lw=1.5, color=(0, 0, 0, alpha))[0]
    shadow_rings.append(ring)

def process_block(block):
    global smoothed_fft, smoothed_bass
    if block.ndim > 1:
        block = block.mean(axis=1)
    block = block - np.mean(block)
    windowed = block * np.hanning(len(block))
    fft = np.abs(np.fft.rfft(windowed))
    fft = fft / np.max(fft + 1e-6)
    freqs = np.fft.rfftfreq(len(block), d=1.0/SAMPLE_RATE)
    bands = np.interp(np.linspace(0, len(fft)-1, NUM_BARS), np.arange(len(fft)), fft)
    smoothed_fft[:] = ALPHA * bands + (1 - ALPHA) * smoothed_fft
    bass_energy = np.mean(fft[freqs <= 50]) if np.any(freqs <= 50) else 0.0
    smoothed_bass = ALPHA * bass_energy + (1 - ALPHA) * smoothed_bass

def init():
    for c in circle_lines: c.set_data([], [])
    for blur in blur_circles: blur.set_data([], [])
    for bar in bars: bar.set_data([], [])
    for rbar in reflection_bars: rbar.set_data([], [])
    reflection_circle.set_data([], [])
    for depth in depth_layers: depth.set_data([], [])
    for ring in shadow_rings: ring.set_data([], [])
    return circle_lines + blur_circles + bars + reflection_bars + [reflection_circle] + depth_layers + shadow_rings

def animate(frame):
    global smoothed_fft, smoothed_bass, circle_history, audio_buffer

    while not q.empty():
        block = q.get_nowait()
        audio_buffer.append(block)
    if audio_buffer:
        process_block(audio_buffer[-1])
        if len(audio_buffer) > 5: audio_buffer = audio_buffer[-5:]
    else:
        smoothed_fft *= 0.95
        smoothed_bass *= 0.95

    base_r = 1.5 + smoothed_bass * 3.0
    x_circ = base_r * np.cos(theta_circle)
    y_circ = base_r * np.sin(theta_circle)

    # history for blur
    if MOTION_BLUR:
        circle_history.append((x_circ.copy(), y_circ.copy(), base_r))
        if len(circle_history) > BLUR_FRAMES:
            circle_history.pop(0)

    # glowing circle with dynamic pulse
    for g, c in enumerate(circle_lines):
        dynamic_alpha = 0.12 * (g+1) + smoothed_bass * 0.25
        dynamic_alpha = min(dynamic_alpha, 1.0)  # clamp so it doesn't blow out
        c.set_data(x_circ, y_circ)
        c.set_linewidth(6 - g + smoothed_bass*3)  # stronger thickness on bass
        c.set_color((0.2, 1.0, 1.0, dynamic_alpha))

    # reflection circle with dynamic pulse
    dynamic_reflection_alpha = 0.25 + smoothed_bass * 0.3
    dynamic_reflection_alpha = min(dynamic_reflection_alpha, 0.8)  # keep subtle
    reflection_circle.set_data(x_circ, -y_circ)
    reflection_circle.set_linewidth(2.5 + smoothed_bass*2)
    reflection_circle.set_color((0.2, 1.0, 1.0, dynamic_reflection_alpha))


    # motion blur
    if MOTION_BLUR and len(circle_history) > 1:
        for i, blur in enumerate(blur_circles):
            if i < len(circle_history) - 2:
                hist_x, hist_y, hist_r = circle_history[-(i+2)]
                scale = 1 - 0.04 * (i+1)
                blur_x = hist_r * scale * np.cos(theta_circle)
                blur_y = hist_r * scale * np.sin(theta_circle)
                blur.set_data(blur_x, blur_y)
                blur.set_linewidth(5 - i)
            else:
                blur.set_data([], [])

    # rotation + wave
    rotation = frame * 0.02
    wave = 0.15 * np.sin(angles * 3 + frame * 0.1)
    rotated_angles = angles + rotation + wave

    # bars + reflection
    for i, bar in enumerate(bars):
        length = smoothed_fft[i] * 2.0
        x = [base_r * np.cos(rotated_angles[i]),
             (base_r + length) * np.cos(rotated_angles[i])]
        y = [base_r * np.sin(rotated_angles[i]),
             (base_r + length) * np.sin(rotated_angles[i])]

        hue = 0.5 + 0.3 * smoothed_fft[i]
        color = hsv_to_rgb([hue % 1.0, 1, 1])

        bar.set_data(x, y)
        bar.set_linewidth(3.0 + smoothed_fft[i]*2)
        bar.set_color((*color, 0.9))

        reflection_bars[i].set_data(x, [-yy for yy in y])
        reflection_bars[i].set_linewidth(2.0 + smoothed_fft[i])
        reflection_bars[i].set_color((*color, 0.25))

    reflection_circle.set_data(x_circ, -y_circ)

    # depth illusion layers
    for i, depth in enumerate(depth_layers):
        offset = 0.05 * (i+1)
        scale = 1.0 + 0.03 * i
        x_depth = (base_r + offset) * np.cos(theta_circle + i*0.1)
        y_depth = (base_r + offset) * np.sin(theta_circle + i*0.1)
        depth.set_data(x_depth * scale, y_depth * scale)

    # gradient shadow rings
    shadow_angle = frame * 0.015
    for i, ring in enumerate(shadow_rings):
        scale = 1 - 0.1 * (i / NUM_SHADOW_RINGS)  # shrink inner rings
        ring_x = base_r * scale * np.cos(theta_circle + shadow_angle*(i+1))
        ring_y = base_r * scale * np.sin(theta_circle + shadow_angle*(i+1))
        ring.set_data(ring_x, ring_y)
        ring.set_linewidth(1.5 + smoothed_bass*1.5)

    return circle_lines + blur_circles + bars + reflection_bars + [reflection_circle] + depth_layers + shadow_rings

ani = animation.FuncAnimation(fig, animate, init_func=init, interval=1,
                              blit=True, cache_frame_data=False)
plt.show()
