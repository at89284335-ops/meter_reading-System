
import cv2
import numpy as np
from pathlib import Path

# PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = Path(__file__).resolve().parent / "meter_dataset"

# print("DATASET PATH:", DATASET_DIR)
# print("EXISTS:", DATASET_DIR.exists())
# print("FILES:", list(DATASET_DIR.rglob("*"))[:5])

OUTPUT_DIR = BASE_DIR / "preprocessed_images"

OUTPUT_DIR.mkdir(
    exist_ok=True
)

# IMAGE RESIZE

def resize_image(image, max_width=1600):

    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

# BRIGHTNESS / CONTRAST

def improve_brightness_contrast(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(
        l_channel
    )

    enhanced = cv2.merge(
        [l_channel, a_channel, b_channel]
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )

    return enhanced

# DENOISING

def remove_noise(image):

    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        5,
        5,
        7,
        21
    )

# SHARPENING

def sharpen_image(image):

    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )

    return cv2.filter2D(
        image,
        -1,
        kernel
    )

# AUTOMATIC METER / DISPLAY REGION

def detect_meter_region(image):

    original = image.copy()

    height, width = original.shape[:2]

    if width < 500 or height < 500:
        return original

    gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    # Improve edge detection
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        40,
        120
    )

    # Connect broken meter boundaries
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = height * width
    candidates = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        area = w * h

        area_ratio = area / image_area

        # Reject tiny objects
        if area_ratio < 0.08:
            continue

        # Reject almost whole image
        if area_ratio > 0.90:
            continue

        aspect_ratio = w / float(h)

        # Allow both landscape and portrait meters
        if aspect_ratio < 0.30 or aspect_ratio > 4.5:
            continue

        # Prefer larger rectangular objects
        contour_area = cv2.contourArea(contour)

        if contour_area <= 0:
            continue

        rectangularity = area / contour_area

        if rectangularity > 8.0:
            continue

        # Center preference
        center_x = x + w / 2
        center_y = y + h / 2

        distance_x = abs(
            center_x - width / 2
        ) / (width / 2)

        distance_y = abs(
            center_y - height / 2
        ) / (height / 2)

        center_score = 1.0 - (
            (distance_x + distance_y) / 2
        )

        score = (
            area_ratio * 0.65
            + center_score * 0.35
        )

        candidates.append(
            (
                score,
                x,
                y,
                w,
                h
            )
        )

    # NO RELIABLE CANDIDATE

    if not candidates:

        # Don't make a risky crop.
        # Keep almost the complete image.
        margin_x = int(width * 0.03)
        margin_y = int(height * 0.03)

        return original[
            margin_y:height - margin_y,
            margin_x:width - margin_x
        ]

    # SELECT BEST CANDIDATE

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    score, x, y, w, h = candidates[0]

    # SAFETY CHECK

    selected_ratio = (w * h) / image_area

    # If selected region is suspiciously small,
    # don't crop aggressively.
    if selected_ratio < 0.20:

        margin_x = int(width * 0.05)
        margin_y = int(height * 0.05)

        return original[
            margin_y:height - margin_y,
            margin_x:width - margin_x
        ]

    # LARGE PADDING

    padding_x = int(w * 0.20)
    padding_y = int(h * 0.20)

    x1 = max(
        0,
        x - padding_x
    )

    y1 = max(
        0,
        y - padding_y
    )

    x2 = min(
        width,
        x + w + padding_x
    )

    y2 = min(
        height,
        y + h + padding_y
    )

    crop = original[
        y1:y2,
        x1:x2
    ]

    # Final safety check
    if crop.shape[0] < height * 0.20:
        return original

    if crop.shape[1] < width * 0.20:
        return original

    return crop

def detect_reading_region(image):

    original = image.copy()

    h, w = original.shape[:2]

    gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    # Contrast improve
    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # Create different threshold versions

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9
    )

    candidates = []

    # Function to find possible digit/display regions

    def collect_candidates(th):

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, 3)
        )

        merged = cv2.morphologyEx(
            th,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        contours, _ = cv2.findContours(
            merged,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            x, y, cw, ch = cv2.boundingRect(
                contour
            )

            area = cw * ch

            # Ignore tiny regions
            if area < (w * h * 0.005):
                continue

            # Ignore almost complete meter
            if area > (w * h * 0.70):
                continue

            ratio = cw / float(ch)

            # Reading display normally wider than high
            if ratio < 1.5:
                continue

            # Candidate crop
            pad_x = int(cw * 0.20)
            pad_y = int(ch * 0.40)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)

            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)

            crop = original[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            # Digit-like pixel analysis
    
            crop_gray = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2GRAY
            )

            crop_gray = cv2.resize(
                crop_gray,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC
            )

            crop_binary = cv2.threshold(
                crop_gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            # Find vertical digit-like components
            digit_contours, _ = cv2.findContours(
                crop_binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            digit_count = 0

            for dc in digit_contours:

                dx, dy, dw, dh = cv2.boundingRect(dc)

                if (
                    dh > crop.shape[0] * 0.15
                    and dh < crop.shape[0] * 0.95
                    and dw < dh
                    and dw > 2
                ):
                    digit_count += 1

            # Candidate score
            score = 0

            # More digit-like components = better
            if digit_count >= 2:
                score += 40

            if digit_count >= 3:
                score += 20

            if digit_count >= 4:
                score += 15

            # Wide display
            if ratio >= 2.0:
                score += 15

            if ratio >= 3.0:
                score += 10

            # Prefer central regions
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            center_x_score = 1 - (
                abs(cx - w / 2) / (w / 2)
            )

            center_y_score = 1 - (
                abs(cy - h / 2) / (h / 2)
            )

            score += max(
                0,
                center_x_score * 10
            )

            score += max(
                0,
                center_y_score * 5
            )

            candidates.append(
                (
                    score,
                    crop
                )
            )

    # Run both threshold methods
    collect_candidates(binary)
    collect_candidates(adaptive)

    # Select best candidate

    if candidates:

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        best_score, best_crop = candidates[0]

        # Only accept if reasonably strong
        if best_score >= 35:
            return best_crop

    # FALLBACK

    # If no reliable display is found,
    # return the complete meter crop instead
    return original


# ROTATION CORRECTION


def correct_rotation(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=100,
        maxLineGap=20
    )

    if lines is None:
        return image

    angles = []

    for line in lines:

        # OpenCV versions may return either
        # [x1, y1, x2, y2] or [[x1, y1, x2, y2]]
        points = np.asarray(line).reshape(-1)

        if len(points) != 4:
            continue

        x1, y1, x2, y2 = points

        angle = np.degrees(
            np.arctan2(
                y2 - y1,
                x2 - x1
            )
        )

        # Only near-horizontal lines
        if -20 <= angle <= 20:
            angles.append(angle)

    if not angles:
        return image

    angle = float(
        np.median(angles)
    )

    # Ignore almost straight images
    if abs(angle) < 1.0:
        return image

    height, width = image.shape[:2]

    center = (
        width // 2,
        height // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# OCR PREPARATION


def create_ocr_image(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Upscale small display regions
    height, width = gray.shape

    if width < 1000:

        scale = 1000 / width

        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(
        gray
    )

    # Light denoise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Adaptive threshold
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return threshold



# COMPLETE PREPROCESSING PIPELINE


def preprocess_image(image):

    image = resize_image(image)

    image = improve_brightness_contrast(image)

    image = remove_noise(image)

    crop = detect_meter_region(image)

    crop = correct_rotation(crop)

    crop = sharpen_image(crop)

    ocr_image = create_ocr_image(crop)

    return crop, ocr_image


# PROCESS ALL DATASET IMAGES


def process_dataset():

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    image_files = [
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    ]

    print(
        f"Total images: {len(image_files)}"
    )

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"[{index}/{len(image_files)}] "
            f"{image_path.name}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                "  Could not read image."
            )

            continue

        try:

            crop, ocr_image = preprocess_image(
                image
            )

            # Save original processed crop
            crop_name = (
                f"{index:04d}_"
                f"{image_path.stem}_crop.jpg"
            )

            ocr_name = (
                f"{index:04d}_"
                f"{image_path.stem}_ocr.jpg"
            )

            cv2.imwrite(
                str(
                    OUTPUT_DIR / crop_name
                ),
                crop
            )

            cv2.imwrite(
                str(
                    OUTPUT_DIR / ocr_name
                ),
                ocr_image
            )

        except Exception as exc:

            print(
                "  Error:",
                exc
            )


if __name__ == "__main__":
    process_dataset()

