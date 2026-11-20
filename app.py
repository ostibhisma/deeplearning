
import cv2
import numpy as np
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# =====================================
# CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent

CASCADE_PATH = (
    BASE_DIR / "fer2013" /
    "haarcascade_frontalface_default.xml"
)

MODEL_PATH = BASE_DIR / "training" / "emotion_model.h5"

# YuNet face detector (MIT).  Detects real faces only, unlike a
# Haar cascade, which also fires on hands and background texture.
FACE_DETECTOR_PATH = (
    BASE_DIR / "models" /
    "face_detection_yunet_2023mar.onnx"
)
YUNET_SCORE_THRESHOLD = 0.85
MIN_FACE_SIZE_RATIO = 0.05

# Detect on a downscaled copy; boxes are mapped back to full size.
# This keeps CPU use low without hurting accuracy at webcam range.
DETECT_MAX_WIDTH = 640

# CAMERA MODE:
# "auto"   = Try available local cameras
# "local"  = Use a specific camera index
# "iphone" = Use iPhone IP video stream

# "auto" is the safest default: camera indices are not stable
# when several cameras are connected (built-in, USB, iPhone).
CAMERA_MODE = "auto"
CAMERA_INDEX = 1
CAMERA_SCAN_RANGE = 5   # auto mode tries indices 0..4

# Capture at 720p: full 1080p makes Haar detection and the
# Pillow overlay noticeably slower with no accuracy benefit,
# since the CNN input is only 48x48.
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

# For iPhone streaming app:
# Replace with the actual stream URL.
IPHONE_STREAM_URL = ""

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

# Emoji shown next to each predicted expression.
# These are plain Unicode characters, so no extra image assets are needed.
EMOTION_EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

EMOJI_FONT_CANDIDATES = [
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "/System/Library/Fonts/Supplemental/Apple Color Emoji.ttc",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "C:/Windows/Fonts/seguiemj.ttf",
]

TEXT_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

PANEL_WIDTH = 252
PANEL_MARGIN = 12
PANEL_ROW_HEIGHT = 30
PANEL_HEADER_HEIGHT = 30

# =====================================
# CAMERA FUNCTIONS
# =====================================


def test_camera(camera, name):
    if not camera.isOpened():
        camera.release()
        return None

    # Allow camera to initialize
    time.sleep(1)

    for attempt in range(10):
        ret, frame = camera.read()

        if (
            ret
            and frame is not None
            and frame.size > 0
        ):
            height, width = frame.shape[:2]
            print(f"Connected to {name} ({width}x{height})")
            return camera

        time.sleep(0.2)

    print(f"{name} opened but delivered no frames; skipping.")
    camera.release()
    return None


def open_local_camera(index):
    print(f"Testing local camera {index}...")

    camera = cv2.VideoCapture(
        index,
        cv2.CAP_AVFOUNDATION
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAPTURE_WIDTH
    )
    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAPTURE_HEIGHT
    )

    return test_camera(
        camera,
        f"camera {index}"
    )


def find_camera():

    if CAMERA_MODE == "iphone":
        if not IPHONE_STREAM_URL:
            raise ValueError(
                "Set IPHONE_STREAM_URL first."
            )

        print("Connecting to iPhone stream...")

        camera = cv2.VideoCapture(
            IPHONE_STREAM_URL
        )

        result = test_camera(
            camera,
            "iPhone IP camera"
        )

        if result is not None:
            return result

        raise RuntimeError(
            "Cannot connect to iPhone stream."
        )

    if CAMERA_MODE == "local":
        camera = open_local_camera(
            CAMERA_INDEX
        )

        if camera is not None:
            return camera

        raise RuntimeError(
            f"Camera {CAMERA_INDEX} failed.\n"
            "On macOS, allow camera access for your terminal/IDE in\n"
            "System Settings > Privacy & Security > Camera, then\n"
            "quit and reopen that app. Also check that CAMERA_INDEX\n"
            "matches a connected camera (indexes are not stable)."
        )

    if CAMERA_MODE == "auto":
        # Camera indices are not stable across machines or when
        # USB / Continuity cameras are connected, so probe a range.
        for index in range(CAMERA_SCAN_RANGE):
            camera = open_local_camera(index)

            if camera is not None:
                print(f"Using camera index {index}.")
                return camera

        raise RuntimeError(
            "No working camera found.\n"
            "  * On macOS, allow camera access for your terminal/IDE:\n"
            "    System Settings > Privacy & Security > Camera.\n"
            "  * Quit and reopen that app after granting access.\n"
            "  * List connected cameras with:\n"
            "    system_profiler SPCameraDataType\n"
            "  * The iPhone/Continuity camera may not appear as a\n"
            "    numbered OpenCV device; use the built-in camera or\n"
            "    the iPhone streaming mode (CAMERA_MODE = \"iphone\")."
        )

    raise ValueError(
        f"Unknown CAMERA_MODE: {CAMERA_MODE}"
    )


# =====================================
# FACE DETECTION
# =====================================


class FaceDetector:
    """
    Face detection front-end.

    YuNet is preferred: it returns real faces only, so hands and
    background texture no longer reach the emotion classifier.
    The Haar cascade remains as a fallback if the model is missing.
    """

    def __init__(self):
        self.yunet = None
        self.haar = None
        self.name = ""

        if (
            hasattr(cv2, "FaceDetectorYN")
            and FACE_DETECTOR_PATH.exists()
        ):
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    str(FACE_DETECTOR_PATH),
                    "",
                    (320, 320),
                    YUNET_SCORE_THRESHOLD,
                    0.3,
                    5000
                )
                self.name = "YuNet (DNN)"
            except Exception as error:
                print(f"YuNet could not be loaded: {error}")

        if self.yunet is None:
            cascade_path = CASCADE_PATH

            try:
                alt2 = (
                    Path(cv2.data.haarcascades)
                    / "haarcascade_frontalface_alt2.xml"
                )
                if alt2.exists():
                    cascade_path = alt2
            except Exception:
                pass

            self.haar = cv2.CascadeClassifier(
                str(cascade_path)
            )

            if self.haar.empty():
                raise RuntimeError(
                    "Face detector could not be loaded."
                )

            self.name = f"Haar cascade ({cascade_path.name})"

        print(f"Face detector: {self.name}")

    def detect(self, frame):
        """Return a list of (x, y, w, h) face boxes for a BGR frame."""
        height, width = frame.shape[:2]
        min_face = max(
            40,
            int(min(height, width) * MIN_FACE_SIZE_RATIO)
        )

        scale = min(1.0, DETECT_MAX_WIDTH / width)
        small_width = max(1, int(round(width * scale)))
        small_height = max(1, int(round(height * scale)))

        if scale < 1.0:
            small = cv2.resize(
                frame,
                (small_width, small_height),
                interpolation=cv2.INTER_AREA
            )
        else:
            small = frame

        if self.yunet is not None:
            self.yunet.setInputSize(
                (small_width, small_height)
            )
            _, faces = self.yunet.detect(small)

            boxes = []

            if faces is not None:
                for face in faces:
                    x, y, w, h = [
                        float(value) for value in face[:4]
                    ]

                    # Map back from the downscaled detection frame.
                    x = x / scale
                    y = y / scale
                    w = w / scale
                    h = h / scale

                    x = int(max(0, round(x)))
                    y = int(max(0, round(y)))
                    w = min(int(round(w)), width - x)
                    h = min(int(round(h)), height - y)

                    if w < min_face or h < min_face:
                        continue

                    # Faces are roughly square; this also drops
                    # slim background/shoulder detections.
                    if not 0.6 <= w / h <= 1.6:
                        continue

                    boxes.append((x, y, w, h))

            return boxes

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = self.haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(
                max(1, int(min_face * scale)),
                max(1, int(min_face * scale))
            )
        )

        boxes = []

        for (x, y, w, h) in faces:
            w = int(round(w / scale))
            h = int(round(h / scale))

            if w < min_face or h < min_face:
                continue

            if not 0.6 <= w / h <= 1.6:
                continue

            boxes.append((
                int(round(x / scale)),
                int(round(y / scale)),
                w,
                h
            ))

        return boxes


# =====================================
# EMOJI / OVERLAY HELPERS
# =====================================

_font_cache = {}
_emoji_master_cache = {}
_emoji_cache = {}

# Bitmap colour fonts (Apple Color Emoji, Noto Color Emoji) only
# contain a few fixed strike sizes.  Trying an unsupported size
# raises "invalid pixel size", so render at the first size that
# works and scale the result down.
EMOJI_RENDER_SIZES = [160, 109, 96, 64, 48, 40, 32, 20]


def _load_font(candidates, size):
    """Return the first font that exists, or None."""
    key = (tuple(candidates), size)

    if key in _font_cache:
        return _font_cache[key]

    font = None

    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            continue

    _font_cache[key] = font
    return font


def _emoji_master(emoji):
    """Render one emoji once, at a size the font supports."""
    if emoji in _emoji_master_cache:
        return _emoji_master_cache[emoji]

    for size in EMOJI_RENDER_SIZES:
        font = _load_font(EMOJI_FONT_CANDIDATES, size)

        if font is None:
            continue

        try:
            canvas = Image.new(
                "RGBA",
                (size * 2, size * 2),
                (0, 0, 0, 0)
            )
            ImageDraw.Draw(canvas).text(
                (0, 0),
                emoji,
                font=font,
                embedded_color=True
            )
        except Exception:
            continue

        bbox = canvas.getbbox()

        if bbox is None:
            continue

        glyph = canvas.crop(bbox)

        # Centre-crop to a square so every emoji scales equally.
        side = min(glyph.size)
        left = (glyph.width - side) // 2
        top = (glyph.height - side) // 2
        glyph = glyph.crop(
            (left, top, left + side, top + side)
        )

        _emoji_master_cache[emoji] = glyph
        return glyph

    _emoji_master_cache[emoji] = None
    return None


def _emoji_tile(emoji, size):
    """Return a transparent square RGBA tile of the requested size."""
    key = (emoji, size)

    if key in _emoji_cache:
        return _emoji_cache[key]

    master = _emoji_master(emoji)

    if master is None:
        _emoji_cache[key] = None
        return None

    tile = master.resize((size, size), Image.LANCZOS)

    _emoji_cache[key] = tile
    return tile


def _text_size(draw, text, font, stroke_width=0):
    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font,
        stroke_width=stroke_width
    )
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_face_overlay(frame, faces, show_panel=True,
                      show_emoji=True):
    """
    Draw the emoji, expression name and confidence on a BGR frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame from OpenCV.
    faces : list of dict
        Each item has keys ``box`` (x, y, w, h), ``emotion``,
        ``confidence`` (0-100) and ``probs`` (7 probabilities).
    """
    width = frame.shape[1]

    if not faces or (not show_emoji and not show_panel):
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")

    for face in faces:
        x, y, w, h = face["box"]
        emotion = face["emotion"]
        confidence = face["confidence"]

        emoji = EMOTION_EMOJI.get(emotion, "")

        # The panel owns the right margin, so keep the face label
        # to its left when the panel is visible.
        avail_right = (
            width - PANEL_WIDTH - PANEL_MARGIN - 8
            if show_panel
            else width - 4
        )
        avail_right = max(avail_right, 120)

        emoji_size = min(max(int(h * 0.45), 28), 64)
        font_size = max(14, int(emoji_size * 0.38))

        label = f"{emotion.capitalize()} {confidence:.1f}%"

        font = _load_font(TEXT_FONT_CANDIDATES, font_size)
        tw, th = _text_size(draw, label, font, 1)

        tile = (
            _emoji_tile(emoji, emoji_size)
            if (show_emoji and emoji)
            else None
        )

        pill_w = (emoji_size if tile else 0) + (8 if tile else 0) + tw + 16
        pill_h = max(emoji_size, th) + 10

        # Shrink the label if the face is close to the panel or
        # to the right edge of the frame.
        if pill_w > avail_right - 4:
            scale = max(0.5, (avail_right - 4) / pill_w)

            emoji_size = max(22, int(emoji_size * scale))
            font_size = max(12, int(font_size * scale))

            font = _load_font(TEXT_FONT_CANDIDATES, font_size)
            tw, th = _text_size(draw, label, font, 1)

            tile = (
                _emoji_tile(emoji, emoji_size)
                if (show_emoji and emoji)
                else None
            )

            pill_w = (
                (emoji_size if tile else 0)
                + (8 if tile else 0)
                + tw
                + 16
            )
            pill_h = max(emoji_size, th) + 10

        # Prefer showing the pill above the face box; otherwise
        # put it inside the box (top-left corner).
        px = x
        py = y - pill_h - 8

        if py < 2:
            py = y + 8

        px = max(2, min(px, avail_right - pill_w - 2))

        draw.rounded_rectangle(
            (px, py, px + pill_w, py + pill_h),
            radius=8,
            fill=(0, 0, 0, 165)
        )

        cursor_x = px + 8

        if tile is not None:
            image.alpha_composite(
                tile,
                (int(cursor_x), int(py + (pill_h - emoji_size) / 2))
            )
            cursor_x += emoji_size + 8

        draw.text(
            (cursor_x, py + (pill_h - th) / 2 - 1),
            label,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255)
        )

    if show_panel and faces:
        draw_confidence_panel(
            image,
            faces,
            width,
            show_emoji
        )

    return cv2.cvtColor(
        np.array(image.convert("RGB")),
        cv2.COLOR_RGB2BGR
    )


_panel_cache = {"layer": None, "key": None, "x": 0, "y": 0, "calls": 0}

# Redraw the side panel at most every N frames; the numbers move
# slowly enough that this is invisible and keeps the feed smooth.
PANEL_REFRESH_EVERY = 3


def _render_panel_layer(primary, width, show_emoji=True):
    """Render the confidence panel once onto a transparent RGBA layer."""
    probs = np.asarray(primary["probs"])
    top = int(np.argmax(probs))

    rows = len(EMOTIONS)
    panel_h = PANEL_HEADER_HEIGHT + rows * PANEL_ROW_HEIGHT + 14
    panel_x = width - PANEL_WIDTH - PANEL_MARGIN

    if panel_x < 0:
        return None, 0, 0

    layer = Image.new(
        "RGBA",
        (PANEL_WIDTH, panel_h),
        (0, 0, 0, 0)
    )
    draw = ImageDraw.Draw(layer, "RGBA")

    draw.rounded_rectangle(
        (0, 0, PANEL_WIDTH - 1, panel_h - 1),
        radius=12,
        fill=(12, 12, 12, 190),
        outline=(255, 255, 255, 45),
        width=1
    )

    header_font = _load_font(TEXT_FONT_CANDIDATES, 14)
    name_font = _load_font(TEXT_FONT_CANDIDATES, 16)
    pct_font = _load_font(TEXT_FONT_CANDIDATES, 15)

    draw.text(
        (14, 9),
        "EXPRESSION",
        font=header_font,
        fill=(255, 255, 255, 130)
    )

    header_w, _ = _text_size(draw, "CONFIDENCE", header_font)
    draw.text(
        (PANEL_WIDTH - 14 - header_w, 9),
        "CONFIDENCE",
        font=header_font,
        fill=(255, 255, 255, 130)
    )

    bar_x = 136
    bar_w = 58

    for index, emotion in enumerate(EMOTIONS):
        row_y = (
            PANEL_HEADER_HEIGHT
            + index * PANEL_ROW_HEIGHT
            + 5
        )

        is_top = index == top

        if is_top:
            draw.rounded_rectangle(
                (
                    6,
                    row_y - 3,
                    PANEL_WIDTH - 6,
                    row_y + PANEL_ROW_HEIGHT - 5
                ),
                radius=7,
                fill=(46, 160, 67, 110)
            )

        emoji = EMOTION_EMOJI[emotion]
        tile = (
            _emoji_tile(emoji, 21)
            if show_emoji
            else None
        )

        if tile is not None:
            layer.alpha_composite(
                tile,
                (12, int(row_y + 1))
            )
            name_x = 40
        else:
            name_x = 14

        name_color = (
            (255, 255, 255, 255)
            if is_top
            else (225, 225, 225, 210)
        )

        draw.text(
            (name_x, row_y),
            emotion.capitalize(),
            font=name_font,
            fill=name_color
        )

        probability = float(probs[index])

        # Track background + filled bar.
        draw.rounded_rectangle(
            (bar_x, row_y + 7, bar_x + bar_w, row_y + 13),
            radius=3,
            fill=(255, 255, 255, 55)
        )

        fill_w = max(3, int(bar_w * probability))

        draw.rounded_rectangle(
            (
                bar_x,
                row_y + 7,
                bar_x + fill_w,
                row_y + 13
            ),
            radius=3,
            fill=(110, 235, 130, 245) if is_top else (150, 150, 150, 210)
        )

        pct_text = f"{probability * 100:.1f}%"
        pct_w, _ = _text_size(draw, pct_text, pct_font)

        draw.text(
            (
                PANEL_WIDTH - 14 - pct_w,
                row_y
            ),
            pct_text,
            font=pct_font,
            fill=(255, 255, 255, 255)
        )

    return layer, panel_x, PANEL_MARGIN


def draw_confidence_panel(image, faces, width, show_emoji=True):
    """
    Draw a compact list of all seven expressions with their
    percentages for the largest detected face.
    """
    # Pick the largest face when several people are visible.
    primary = max(
        faces,
        key=lambda f: f["box"][2] * f["box"][3]
    )

    key = (
        width,
        show_emoji,
        primary["box"],
        tuple(np.round(np.asarray(primary["probs"]), 4))
    )

    _panel_cache["calls"] += 1

    needs_refresh = (
        _panel_cache["layer"] is None
        or (
            key != _panel_cache["key"]
            and _panel_cache["calls"] % PANEL_REFRESH_EVERY == 0
        )
    )

    if needs_refresh:
        layer, panel_x, panel_y = _render_panel_layer(
            primary,
            width,
            show_emoji
        )

        if layer is not None:
            _panel_cache.update(
                layer=layer,
                key=key,
                x=panel_x,
                y=panel_y
            )

    if _panel_cache["layer"] is not None:
        image.alpha_composite(
            _panel_cache["layer"],
            (_panel_cache["x"], _panel_cache["y"])
        )


# =====================================
# MAIN
# =====================================


def main():
    from tensorflow.keras.models import load_model

    print("Loading TensorFlow model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = load_model(
        str(MODEL_PATH),
        compile=False
    )

    print("Model loaded successfully!")

    face_detector = FaceDetector()

    video = find_camera()

    print("Camera initialized successfully!")
    print("Keys: Q = quit, E = toggle emoji, P = toggle panel.")

    failed_frames = 0
    show_emoji = True
    show_panel = True

    try:
        while True:

            ret, frame = video.read()

            if (
                not ret
                or frame is None
                or frame.size == 0
            ):
                failed_frames += 1

                if failed_frames >= 30:
                    print(
                        "Camera stopped providing frames."
                    )
                    break

                time.sleep(0.1)
                continue

            failed_frames = 0

            # Mirror local webcam display
            if CAMERA_MODE != "iphone":
                frame = cv2.flip(frame, 1)

            detected_faces = face_detector.detect(frame)

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            results = []

            for (x, y, w, h) in detected_faces:

                roi = gray[y:y+h, x:x+w]

                if roi.size == 0:
                    continue

                roi = cv2.resize(
                    roi,
                    (48, 48),
                    interpolation=cv2.INTER_AREA
                )

                roi = (
                    roi.astype("float32") / 255.0
                )

                image = roi.reshape(
                    1, 48, 48, 1
                )

                # predict_on_batch avoids the per-call overhead
                # of predict() (about 8x faster here).
                prediction = model.predict_on_batch(
                    image
                )[0]

                label = int(np.argmax(prediction))

                emotion = EMOTIONS[label]

                confidence = (
                    float(prediction[label]) * 100
                )

                # Draw bounding box (OpenCV part)
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                results.append({
                    "box": (x, y, w, h),
                    "emotion": emotion,
                    "confidence": confidence,
                    "probs": prediction
                })

            frame = draw_face_overlay(
                frame,
                results,
                show_panel=show_panel,
                show_emoji=show_emoji
            )

            cv2.imshow(
                "FER2013 Emotion Recognition",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("e"):
                show_emoji = not show_emoji

            if key == ord("p"):
                show_panel = not show_panel

    finally:
        video.release()
        cv2.destroyAllWindows()
        print("Application closed.")


if __name__ == "__main__":
    main()
