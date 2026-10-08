
import cv2
import pytesseract
import re
import csv
from pathlib import Path

# TESSERACT PATH

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "preprocessed_images"
OUTPUT_DIR = BASE_DIR / "ocr_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_TXT = OUTPUT_DIR / "results.txt"
RESULT_CSV = OUTPUT_DIR / "results.csv"

# SETTINGS

MAX_REGIONS = 4

OCR_CONFIGS = [
    "--psm 7 -c tessedit_char_whitelist=0123456789.",
    "--psm 6 -c tessedit_char_whitelist=0123456789.",
]

# CREATE OCR VARIANTS

def create_variants(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Variant 1 - normal grayscale
    normal = gray

    # Variant 2 - OTSU
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return [normal, otsu]

# FIND POSSIBLE READING REGIONS

def find_possible_regions(image):

    h, w = image.shape[:2]

    candidates = []

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # 1. Blackhat - useful for dark digits

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 15)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, thresh = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        x, y, cw, ch = cv2.boundingRect(contour)

        area = cw * ch

        if area < (w * h * 0.005):
            continue

        if cw < 150 or ch < 30:
            continue

        ratio = cw / float(ch)

        # Reading/display usually wider than tall
        if ratio < 1.5:
            continue

        if ratio > 12:
            continue

        # Padding
        pad_x = int(cw * 0.15)
        pad_y = int(ch * 0.35)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)

        x2 = min(w, x + cw + pad_x)
        y2 = min(h, y + ch + pad_y)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        candidates.append(
            ("contour", crop, area)
        )


    # 2. Broad upper/middle/lower regions

    # Upper-middle
    y1 = int(h * 0.15)
    y2 = int(h * 0.55)

    upper = image[y1:y2, :]

    candidates.append(
        ("upper", upper, upper.shape[0] * upper.shape[1])
    )


    # Middle
    y1 = int(h * 0.25)
    y2 = int(h * 0.75)

    middle = image[y1:y2, :]

    candidates.append(
        ("middle", middle, middle.shape[0] * middle.shape[1])
    )


    # Lower-middle
    y1 = int(h * 0.45)
    y2 = int(h * 0.90)

    lower = image[y1:y2, :]

    candidates.append(
        ("lower", lower, lower.shape[0] * lower.shape[1])
    )


    # 3. Full image fallback

    candidates.append(
        ("full", image, w * h)
    )


    # Remove duplicate/small candidates

    filtered = []

    for name, crop, area in candidates:

        if crop is None or crop.size == 0:
            continue

        ch, cw = crop.shape[:2]

        if cw < 200 or ch < 50:
            continue

        filtered.append(
            (name, crop, area)
        )


    # Prefer contour candidates first
    # Then broad regions

    contour_candidates = [
        item for item in filtered
        if item[0] == "contour"
    ]

    broad_candidates = [
        item for item in filtered
        if item[0] != "contour"
    ]


    # Largest useful contour candidates
    contour_candidates.sort(
        key=lambda x: x[2],
        reverse=True
    )

    # Keep only top 2 contour candidates
    selected = contour_candidates[:2]

    # Add maximum 2 broad regions
    selected.extend(
        broad_candidates[:2]
    )


    # Final maximum

    return selected[:MAX_REGIONS]

# CLEAN OCR TEXT

def clean_text(text):

    if not text:
        return ""

    text = text.upper()

    replacements = {
        "O": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "B": "8",
        ",": "."
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"[^0-9.]",
        " ",
        text
    )

    return text

# EXTRACT NUMBERS

def extract_numbers(text):

    text = clean_text(text)

    if not text:
        return []

    matches = re.findall(
        r"\d+(?:\.\d+)?",
        text
    )

    valid = []

    for value in matches:

        digit_count = len(
            value.replace(".", "")
        )

        # Reject 1-2 digit values
        if digit_count < 3:
            continue

        # Reject huge OCR garbage
        if digit_count > 10:
            continue

        try:
            float(value)
        except:
            continue

        valid.append(value)

    return valid

# NORMALIZE READING

def normalize_reading(value):

    if not value:
        return ""

    # Multiple decimal points
    value = re.sub(
        r"\.{2,}",
        ".",
        value
    )

    value = value.strip(".")

    if not value:
        return ""

    if "." in value:

        integer_part, decimal_part = value.split(
            ".",
            1
        )

        integer_part = integer_part.lstrip("0")

        if integer_part == "":
            integer_part = "0"

        decimal_part = decimal_part.rstrip("0")

        if decimal_part:
            value = (
                integer_part
                + "."
                + decimal_part
            )
        else:
            value = integer_part

    else:

        value = value.lstrip("0")

        if value == "":
            value = "0"

    return value

# SCORE READING

def score_reading(
    value,
    confidence,
    region_name
):

    digits = len(
        value.replace(".", "")
    )

    score = float(confidence)

    # Meter readings commonly have multiple digits
    if 4 <= digits <= 8:
        score += 25

    elif digits == 3:
        score += 5

    elif digits >= 9:
        score -= 10


    # Decimal is useful
    if "." in value:
        score += 10


    # Very small decimal is suspicious
    try:

        if float(value) < 1:
            score -= 30

    except:
        pass


    # Contour candidate gets slight preference
    if region_name == "contour":
        score += 5


    return score

# OCR ONE REGION

def ocr_region(
    region,
    region_name
):

    variants = create_variants(
        region
    )

    best_reading = ""
    best_token = ""
    best_score = -1
    best_confidence = 0


    for variant in variants:

        for config in OCR_CONFIGS:

            try:

                data = pytesseract.image_to_data(
                    variant,
                    config=config,
                    output_type=pytesseract.Output.DICT
                )

            except Exception:
                continue


            texts = data["text"]
            confidences = data["conf"]


            for text, conf in zip(
                texts,
                confidences
            ):

                text = text.strip()

                if not text:
                    continue


                try:
                    confidence = float(conf)
                except:
                    confidence = 0


                if confidence < 15:
                    continue


                numbers = extract_numbers(
                    text
                )


                for number in numbers:

                    reading = normalize_reading(
                        number
                    )

                    if not reading:
                        continue


                    score = score_reading(
                        reading,
                        confidence,
                        region_name
                    )


                    if score > best_score:

                        best_score = score
                        best_reading = reading
                        best_token = text
                        best_confidence = confidence


    return {
        "reading": best_reading,
        "token": best_token,
        "score": best_score,
        "confidence": best_confidence
    }

# EXTRACT READING FROM FULL IMAGE

def extract_reading(image):

    regions = find_possible_regions(
        image
    )

    best = {
        "reading": "",
        "token": "",
        "score": -1,
        "confidence": 0,
        "region": ""
    }


    for region_name, region, area in regions:

        result = ocr_region(
            region,
            region_name
        )


        if not result["reading"]:
            continue


        if result["score"] > best["score"]:

            best = {
                "reading": result["reading"],
                "token": result["token"],
                "score": result["score"],
                "confidence": result["confidence"],
                "region": region_name
            }


    if not best["reading"]:

        return {
            "reading": "NOT_READ",
            "confidence": 0,
            "token": "",
            "region": ""
        }


    final_confidence = max(
        0,
        min(
            100,
            round(best["score"])
        )
    )


    return {
        "reading": best["reading"],
        "confidence": final_confidence,
        "token": best["token"],
        "region": best["region"]
    }

# PROCESS ALL IMAGES

def process_all():

    image_files = sorted(
        [
            p
            for p in INPUT_DIR.rglob("*")
            if p.is_file()
            and p.name.lower().endswith(
                "_crop.jpg"
            )
        ]
    )


    print()
    print("=" * 60)
    print("FAST METER OCR - LIMITED REGIONS")
    print("=" * 60)

    print(
        f"Total images: {len(image_files)}"
    )

    print()


    if not image_files:

        print("ERROR: No crop images found.")
        print(
            f"Check: {INPUT_DIR}"
        )
        return


    results = []


    with open(
        RESULT_TXT,
        "w",
        encoding="utf-8"
    ) as txt_file:

        txt_file.write(
            "METER OCR RESULTS\n"
        )

        txt_file.write(
            "=" * 60 + "\n\n"
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
                    "    IMAGE ERROR"
                )

                results.append([
                    image_path.name,
                    "IMAGE_ERROR",
                    0,
                    "",
                    ""
                ])

                continue


            result = extract_reading(
                image
            )


            reading = result["reading"]
            confidence = result["confidence"]
            token = result["token"]
            region = result["region"]


            print(
                f"    Reading: {reading}"
            )

            print(
                f"    Confidence: "
                f"{confidence}%"
            )

            print(
                f"    OCR token: "
                f"{token}"
            )

            print(
                f"    Region: "
                f"{region}"
            )


            txt_file.write(
                f"{index}. "
                f"{image_path.name}\n"
            )

            txt_file.write(
                f"Reading: {reading}\n"
            )

            txt_file.write(
                f"Confidence: "
                f"{confidence}%\n"
            )

            txt_file.write(
                f"OCR Token: {token}\n"
            )

            txt_file.write(
                f"Region: {region}\n"
            )

            txt_file.write(
                "-" * 50 + "\n"
            )


            results.append([
                image_path.name,
                reading,
                confidence,
                token,
                region
            ])
            
    # CSV

    with open(
        RESULT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow([
            "image",
            "reading",
            "confidence",
            "ocr_token",
            "region"
        ])

        writer.writerows(
            results
        )
        
    # SUMMARY

    total = len(results)

    successful = sum(
        1
        for row in results
        if row[1] not in [
            "NOT_READ",
            "IMAGE_ERROR"
        ]
    )

    failed = total - successful


    print()
    print("=" * 60)
    print("OCR COMPLETED")
    print("=" * 60)

    print(
        f"Total images : {total}"
    )

    print(
        f"Readings     : {successful}"
    )

    print(
        f"Not read     : {failed}"
    )

    print()
    print(
        f"Results TXT: {RESULT_TXT}"
    )

    print(
        f"Results CSV: {RESULT_CSV}"
    )

# MAIN

if __name__ == "__main__":
    process_all()
