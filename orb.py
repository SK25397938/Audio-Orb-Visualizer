import numpy as np
import pygame
import sounddevice as sd
import queue
import threading
import time
from collections import deque

# ---------- Configuration ----------
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
DEVICE_INDEX = 16  
NUM_BARS = 180
ALPHA = 0.15
MOTION_BLUR = True
BLUR_FRAMES = 6
MAX_QUEUE_SIZE = 5
FPS = 240  # Ultra-smooth motion
PEAK_COMPRESSION = 0.7
MIN_RADIUS_FACTOR = 0.8
MAX_RADIUS_FACTOR = 2.5

# --- Motion smoothing tuning ---
RADIUS_SMOOTHING = 0.85
BAR_SMOOTHING = 0.35

# --- Inner ring behavior ---
NUM_INNER_RINGS = 8
INNER_REACTION_STRENGTH = 0.18  
INNER_DECAY = 0.08  

# ---------- Shared audio data ----------
audio_data = {
    'smoothed_fft': np.zeros(NUM_BARS),
    'smoothed_bass': 0.0,
    'last_update': 0.0,
    'lock': threading.Lock()
}
q = queue.Queue(maxsize=MAX_QUEUE_SIZE)

def audio_callback(indata, frames, time_, status):
    if status:
        print(status)
    try:
        q.put_nowait(indata.copy())
    except queue.Full:
        pass

def audio_processing_thread():
    """Processes live audio in a separate thread."""
    while True:
        try:
            block = q.get(timeout=0.1)
            if block.ndim > 1:
                block = block.mean(axis=1)
            block = block - np.mean(block)
            windowed = block * np.hanning(len(block))
            fft = np.abs(np.fft.rfft(windowed))
            fft = fft / np.max(fft + 1e-6)

            freqs = np.fft.rfftfreq(len(block), d=1.0/SAMPLE_RATE)
            bands = np.interp(np.linspace(0, len(fft)-1, NUM_BARS),
                              np.arange(len(fft)), fft)

            if np.max(bands) > 0.1:
                avg = np.mean(bands)
                peak_threshold = avg + (np.max(bands) - avg) * 0.7
                mask = bands > peak_threshold
                if np.any(mask):
                    compression_factor = 1.0 - PEAK_COMPRESSION * (bands[mask] - peak_threshold) / (np.max(bands) - peak_threshold + 1e-6)
                    bands[mask] = peak_threshold + (bands[mask] - peak_threshold) * compression_factor

            bands = np.clip(bands, 0, 1)

            with audio_data['lock']:
                audio_data['smoothed_fft'][:] = (
                    ALPHA * bands + (1 - ALPHA) * audio_data['smoothed_fft']
                )
                bass_energy = np.mean(fft[freqs <= 60]) if np.any(freqs <= 60) else 0.0
                audio_data['smoothed_bass'] = (
                    ALPHA * bass_energy + (1 - ALPHA) * audio_data['smoothed_bass']
                )
                audio_data['last_update'] = time.time()

        except queue.Empty:
            with audio_data['lock']:
                now = time.time()
                if now - audio_data['last_update'] > 0.1:
                    audio_data['smoothed_fft'] *= 0.95
                    audio_data['smoothed_bass'] *= 0.95
                    audio_data['last_update'] = now

threading.Thread(target=audio_processing_thread, daemon=True).start()

try:
    stream = sd.InputStream(device=DEVICE_INDEX, channels=2, samplerate=SAMPLE_RATE,
                            blocksize=BLOCK_SIZE, callback=audio_callback, dtype="float32")
    stream.start()
except Exception as e:
    print("Couldn't start audio stream:", e)
    exit(1)

# ---------- Pygame setup ----------
pygame.init()
info = pygame.display.Info()
WINDOW_WIDTH, WINDOW_HEIGHT = info.current_w, info.current_h
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.NOFRAME | pygame.SCALED)
pygame.display.set_caption("Circular Spectrum Visualizer")
clock = pygame.time.Clock()

center_x = WINDOW_WIDTH // 2
center_y = WINDOW_HEIGHT // 2
scale_factor = min(WINDOW_WIDTH, WINDOW_HEIGHT) / 10

angles = np.linspace(0, 2*np.pi, NUM_BARS, endpoint=False)
alpha_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
reflection_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

# ---------- Dynamic Neon Color ----------
def neon_color(val, brightness=1.0):
    """Returns a brightness-modulated neon color."""
    val = np.clip(val, 0.0, 1.0)
    r = int((120 + 135 * val) * brightness)
    g = int((100 + 155 * (1 - val)) * brightness)
    b = int(255 * val * brightness)
    return (min(r, 255), min(g, 255), min(b, 255))

display_radius = 0
display_fft = np.zeros(NUM_BARS)

# --- Inner ring state ---
inner_ring_offsets = [0.0 for _ in range(NUM_INNER_RINGS)]
inner_ring_smooth = [INNER_DECAY + 0.02 * i for i in range(NUM_INNER_RINGS)]

# ---------- Main Loop ----------
running = True
frame_count = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    screen.fill((0, 0, 0))
    alpha_surface.fill((0, 0, 0, 0))
    reflection_surface.fill((0, 0, 0, 0))

    with audio_data['lock']:
        smoothed_fft = audio_data['smoothed_fft'].copy()
        smoothed_bass = audio_data['smoothed_bass']

    # --- Dynamic brightness based on contraction/expansion ---
    brightness = np.clip(0.6 + smoothed_bass * 1.2, 0.6, 1.5)

    target_radius = (2.0 + smoothed_bass * 2.5) * scale_factor
    delta_r = (target_radius - display_radius) * RADIUS_SMOOTHING
    display_radius += delta_r * 0.9 + delta_r * 0.1 * np.sin(frame_count * 0.05)
    base_r = display_radius
    display_fft += (smoothed_fft - display_fft) * BAR_SMOOTHING

    min_radius = base_r * MIN_RADIUS_FACTOR
    max_radius = base_r * MAX_RADIUS_FACTOR

    rotation = frame_count * 0.01
    cos_vals = np.cos(angles + rotation)
    sin_vals = np.sin(angles + rotation)
    reflection_cos_vals = np.cos(-angles - rotation)
    reflection_sin_vals = np.sin(-angles - rotation)

    # --- Glowing bars ---
    for i in range(NUM_BARS):
        length = display_fft[i] * 2.2 * scale_factor
        outer_r = min(base_r + length, max_radius)
        constrained_base_r = np.clip(base_r, min_radius, max_radius)
        x_base = center_x + constrained_base_r * cos_vals[i]
        y_base = center_y + constrained_base_r * sin_vals[i]
        x_out = center_x + outer_r * cos_vals[i]
        y_out = center_y + outer_r * sin_vals[i]
        color = neon_color(display_fft[i], brightness)
        width = max(2, int((2.0 + display_fft[i]*2) * scale_factor / 50))

        for g in range(4):
            glow_alpha = max(0, int((100 - g * 20) * brightness))
            pygame.draw.line(alpha_surface, (*color, glow_alpha),
                             (x_base, y_base), (x_out, y_out), width + g*2)

        pygame.draw.line(screen, color, (x_base, y_base), (x_out, y_out), width)

        # Reflection
        reflection_outer_r = min(base_r + length, max_radius)
        reflection_x_base = center_x + constrained_base_r * reflection_cos_vals[i]
        reflection_y_base = center_y + constrained_base_r * reflection_sin_vals[i]
        reflection_x_out = center_x + reflection_outer_r * reflection_cos_vals[i]
        reflection_y_out = center_y + reflection_outer_r * reflection_sin_vals[i]
        for blur in range(3):
            blur_alpha = max(0, int((120 - blur * 30) * brightness))
            pygame.draw.line(reflection_surface, (*color, blur_alpha),
                             (reflection_x_base, reflection_y_base),
                             (reflection_x_out, reflection_y_out),
                             max(1, width - blur))
    # --- Luminous baseline ring ---
    constrained_ring_r = np.clip(base_r, 0, max_radius)
    circle_points = [
        (center_x + constrained_ring_r * np.cos(a),
         center_y + constrained_ring_r * np.sin(a))
        for a in np.linspace(0, 2*np.pi, 400)
    ]

    energy_level = np.mean(display_fft)
    ring_color = neon_color(energy_level, brightness)

    # Clamp color safely
    def clamp_color(c):
        return tuple(int(np.clip(x, 0, 255)) for x in c)

    ring_color = clamp_color(ring_color)
    ring_alpha = int(np.clip(220 * brightness, 0, 255))

    pygame.draw.lines(alpha_surface, (*ring_color, ring_alpha), True, circle_points, 3)

    for g in range(1, 8):
        glow_alpha = int(np.clip((180 - g * 25) * brightness, 0, 255))
        fade_factor = 1.0 - g * 0.1
        glow_color = clamp_color((
            ring_color[0] * fade_factor,
            ring_color[1] * fade_factor,
            ring_color[2] * fade_factor
        ))
        pygame.draw.lines(alpha_surface, (*glow_color, glow_alpha), True, [
            (center_x + (constrained_ring_r + g * 2) * np.cos(a),
             center_y + (constrained_ring_r + g * 2) * np.sin(a))
            for a in np.linspace(0, 2*np.pi, 400)
        ], 3 + g)

    # --- Inner holographic rings ---
    inner_rotation = frame_count * 0.008
    for i in range(NUM_INNER_RINGS):
        target = smoothed_bass * scale_factor * (INNER_REACTION_STRENGTH + 0.02 * i)
        smooth = inner_ring_smooth[i]
        inner_ring_offsets[i] += (target - inner_ring_offsets[i]) * smooth

        scale = 1.0 - 0.08 * (i / max(1, NUM_INNER_RINGS))
        inner_r = max(8.0, constrained_ring_r * scale - inner_ring_offsets[i])
        alpha_level = int(120 * (1 - i / NUM_INNER_RINGS) * brightness)
        fade_factor = 1.0 - (i / NUM_INNER_RINGS) * 0.6
        glow_color = (
            int(ring_color[0] * fade_factor),
            int(ring_color[1] * fade_factor),
            int(ring_color[2] * fade_factor),
            alpha_level
        )

        inner_points = [
            (center_x + inner_r * np.cos(a + inner_rotation * (i + 1)),
             center_y + inner_r * np.sin(a + inner_rotation * (i + 1)))
            for a in np.linspace(0, 2*np.pi, 240)
        ]
        if len(inner_points) > 1:
            pygame.draw.lines(alpha_surface, glow_color, True, inner_points, 2)

    # --- Blit layers ---
    screen.blit(reflection_surface, (0, 0))
    screen.blit(alpha_surface, (0, 0))
    pygame.display.flip()
    clock.tick(FPS)
    frame_count += 1

pygame.quit()
stream.stop()
stream.close() 
