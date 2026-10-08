
import csv
import re
from pathlib import Path

# PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = BASE_DIR / "ocr_results" / "results.csv"
OUTPUT_CSV = BASE_DIR / "ocr_results" / "validated_results.csv"
OUTPUT_TXT = BASE_DIR / "ocr_results" / "validated_results.txt"

# SETTINGS

MIN_DIGITS = 3
MAX_DIGITS = 10

MIN_CONFIDENCE = 30

# CLEAN OCR VALUE

def clean_value(value):

    if not value:
        return ""

    value = str(value).strip().upper()

    # Common OCR confusions
    replacements = {
        "O": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "B": "8",
        ",": "."
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    # Keep only digits and decimal point
    value = re.sub(
        r"[^0-9.]",
        "",
        value
    )

    # More than one decimal point
    if value.count(".") > 1:

        first_dot = value.find(".")

        value = (
            value[:first_dot + 1]
            + value[first_dot + 1:].replace(".", "")
        )

    # Remove decimal at beginning/end
    value = value.strip(".")

    return value

# VALIDATE READING

def validate_reading(value, confidence):

    value = clean_value(value)

    # Empty

    if not value:
        return "INVALID", "No numeric reading"

    # Format

    if not re.fullmatch(
        r"\d+(?:\.\d+)?",
        value
    ):
        return "INVALID", "Invalid numeric format"

    # Digit count

    digit_count = len(
        value.replace(".", "")
    )

    if digit_count < MIN_DIGITS:
        return "INVALID", "Less than 3 digits"


    if digit_count > MAX_DIGITS:
        return "INVALID", "More than 10 digits"

    # Confidence

    try:
        confidence = float(confidence)
    except:
        confidence = 0


    if confidence < MIN_CONFIDENCE:
        return "INVALID", "Low OCR confidence"

    # Numeric value

    try:
        number = float(value)
    except:
        return "INVALID", "Cannot convert to number"


    if number < 0:
        return "INVALID", "Negative reading"

    # Suspicious very small decimal

    if "." in value and number < 1:
        return "INVALID", "Suspicious small decimal"

    # Repeated digit garbage

    integer_part = value.split(".")[0]

    if len(integer_part) >= 4:

        if len(set(integer_part)) == 1:
            return "INVALID", "Repeated digit pattern"

    # Valid

    return "VALID", "Passed validation"

# NORMALIZE VALID READING

def normalize_reading(value):

    value = clean_value(value)

    if not value:
        return ""


    if "." in value:

        integer_part, decimal_part = value.split(
            ".",
            1
        )

        # Remove unnecessary leading zeros
        integer_part = integer_part.lstrip("0")

        if integer_part == "":
            integer_part = "0"

        # Remove unnecessary trailing zeros
        decimal_part = decimal_part.rstrip("0")

        if decimal_part:
            return (
                integer_part
                + "."
                + decimal_part
            )

        return integer_part


    # Integer
    value = value.lstrip("0")

    if value == "":
        value = "0"

    return value

# PROCESS CSV

def process_validation():

    print()
    print("=" * 60)
    print("METER READING VALIDATION")
    print("=" * 60)

    print(
        f"Input : {INPUT_CSV}"
    )


    if not INPUT_CSV.exists():

        print()
        print("ERROR: results.csv not found.")
        print()
        print("Run meter_ocr.py first.")
        return


    valid_count = 0
    invalid_count = 0
    total_count = 0

    output_rows = []

    # Read OCR CSV

    with open(
        INPUT_CSV,
        "r",
        encoding="utf-8"
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row in reader:

            total_count += 1

            image_name = row.get(
                "image",
                ""
            )

            reading = row.get(
                "reading",
                ""
            )

            confidence = row.get(
                "confidence",
                "0"
            )

            token = row.get(
                "ocr_token",
                ""
            )

            region = row.get(
                "region",
                ""
            )

            # Validate

            status, reason = validate_reading(
                reading,
                confidence
            )


            # Normalize only if valid
            if status == "VALID":

                final_reading = normalize_reading(
                    reading
                )

                valid_count += 1

            else:

                final_reading = "INVALID"

                invalid_count += 1


            output_rows.append([
                image_name,
                reading,
                final_reading,
                confidence,
                token,
                region,
                status,
                reason
            ])

            # Console

            # print(
            #     f"[{total_count}] "
            #     f"{image_name}"
            # )

            # print(
            #     f"    OCR Reading : {reading}"
            # )

            # print(
            #     f"    Final       : {final_reading}"
            # )

            # print(
            #     f"    Confidence  : {confidence}%"
            # )

            # print(
            #     f"    Status      : {status}"
            # )

            # print(
            #     f"    Reason      : {reason}"
            # )

    # SAVE CSV

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow([
            "image",
            "ocr_reading",
            "final_reading",
            "confidence",
            "ocr_token",
            "region",
            "status",
            "validation_reason"
        ])

        writer.writerows(
            output_rows
        )

    # SAVE TXT

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8"
    ) as txt_file:

        txt_file.write(
            "METER READING VALIDATION RESULTS\n"
        )

        txt_file.write(
            "=" * 60 + "\n\n"
        )


        for row in output_rows:

            (
                image_name,
                ocr_reading,
                final_reading,
                confidence,
                token,
                region,
                status,
                reason
            ) = row


            txt_file.write(
                f"Image: {image_name}\n"
            )

            txt_file.write(
                f"OCR Reading: {ocr_reading}\n"
            )

            txt_file.write(
                f"Final Reading: {final_reading}\n"
            )

            txt_file.write(
                f"Confidence: {confidence}%\n"
            )

            txt_file.write(
                f"OCR Token: {token}\n"
            )

            txt_file.write(
                f"Region: {region}\n"
            )

            txt_file.write(
                f"Status: {status}\n"
            )

            txt_file.write(
                f"Reason: {reason}\n"
            )

            txt_file.write(
                "-" * 50 + "\n"
            )

    # SUMMARY

    # print()
    # print("=" * 60)
    # print("VALIDATION COMPLETED")
    # print("=" * 60)

    # print(
    #     f"Total records : {total_count}"
    # )

    # print(
    #     f"VALID         : {valid_count}"
    # )

    # print(
    #     f"INVALID       : {invalid_count}"
    # )

    # print()

    # print(
    #     f"CSV saved at:"
    # )

    # print(
    #     OUTPUT_CSV
    # )

    # print()

    # print(
    #     f"TXT saved at:"
    # )

    # print(
    #     OUTPUT_TXT
    # )

# MAIN

if __name__ == "__main__":
    process_validation()
