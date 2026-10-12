import cv2
import numpy as np

from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

from .meter_preprocessing import preprocess_image
from .meter_ocr import extract_reading
from .meter_validation import validate_reading

from rest_framework import serializers
from drf_spectacular.utils import extend_schema, OpenApiResponse, inline_serializer


# HELPER FUNCTION

def process_meter_image(uploaded_file):
   

    # READ IMAGE    
    image_bytes = uploaded_file.read()

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return {
            "success": False,
            "status": "INVALID",
            "reading": None,
            "confidence": 0,
            "message": "Invalid image file."
        }

    # PREPROCESSING    
    try:
        processed_crop, ocr_image = preprocess_image(image)

    except Exception as e:
        return {
            "success": False,
            "status": "ERROR",
            "reading": None,
            "confidence": 0,
            "message": f"Preprocessing failed: {str(e)}"
        }

    # OCR    
    try:
        ocr_result = extract_reading(processed_crop)

    except Exception as e:
        return {
            "success": False,
            "status": "ERROR",
            "reading": None,
            "confidence": 0,
            "message": f"OCR failed: {str(e)}"
        }

    # OCR RESULT    
    if not isinstance(ocr_result, dict):

        return {
            "success": False,
            "status": "INVALID",
            "reading": None,
            "confidence": 0,
            "message": "OCR returned invalid result."
        }

    reading = ocr_result.get(
        "reading",
        ""
    )

    confidence = ocr_result.get(
        "confidence",
        0
    )

    # VALIDATION    
    try:
        validation_result = validate_reading(
            reading,
            confidence
        )

    except Exception as e:

        return {
            "success": False,
            "status": "ERROR",
            "reading": reading,
            "confidence": confidence,
            "message": f"Validation failed: {str(e)}"
        }

    # VALIDATION RESULT    
    if isinstance(validation_result, dict):

        final_reading = validation_result.get(
            "reading",
            validation_result.get(
                "value",
                reading
            )
        )

        final_confidence = validation_result.get(
            "confidence",
            confidence
        )

        validation_status = validation_result.get(
            "status",
            "VALID"
        )

        reason = validation_result.get(
            "reason",
            ""
        )

    else:

        final_reading = reading
        final_confidence = confidence

        if validation_result:
            validation_status = "VALID"
        else:
            validation_status = "INVALID"

        reason = ""

    # FINAL RESULT    
    return {
        "success": validation_status == "VALID",
        "status": validation_status,
        "reading": (
            final_reading
            if validation_status == "VALID"
            else None
        ),
        "confidence": round(
            float(final_confidence),
            2
        ),
        "reason": reason
    }

# SINGLE IMAGE API

@extend_schema(
    tags=["Meter Reading"],
    summary="Read a single meter image",
    request={
        "multipart/form-data": {
            "type": "object",
            "required": ["image"],
            "properties": {
                "image": {
                    "type": "string",
                    "format": "binary",
                }
            },
        }
    },
    responses={
        200: OpenApiResponse(description="Meter reading result"),
        400: OpenApiResponse(description="Invalid or missing image"),
    },
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def meter_read_api(request):

    uploaded_file = request.FILES.get("image")

    # IMAGE REQUIRED    
    if not uploaded_file:

        return Response(
            {
                "success": False,
                "status": "INVALID",
                "message": (
                    "Please upload image using "
                    "field name 'image'."
                )
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    # PROCESS IMAGE    
    result = process_meter_image(
        uploaded_file
    )

    # Add filename
    result["filename"] = uploaded_file.name

    return Response(
        result,
        status=status.HTTP_200_OK
    )

# BATCH IMAGE API

@extend_schema(
    tags=["Meter Reading"],
    summary="Read multiple meter images",
    request={
        "multipart/form-data": {
            "type": "object",
            "required": ["images"],
            "properties": {
                "images": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "format": "binary",
                    },
                }
            },
        }
    },
    responses={
        200: OpenApiResponse(description="Batch processing results"),
        400: OpenApiResponse(description="Images missing or invalid"),
    },
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def meter_batch_api(request):

    uploaded_files = request.FILES.getlist(
        "images"
    )

    # IMAGES REQUIRED    
    if not uploaded_files:

        return Response(
            {
                "success": False,
                "status": "INVALID",
                "message": (
                    "Please upload multiple images "
                    "using field name 'images'."
                )
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    # PROCESS ALL IMAGES    
    results = []

    valid_count = 0
    invalid_count = 0

    for uploaded_file in uploaded_files:

        result = process_meter_image(
            uploaded_file
        )

        result["filename"] = uploaded_file.name

        results.append(result)

        if result["status"] == "VALID":
            valid_count += 1
        else:
            invalid_count += 1

    # FINAL RESPONSE
    #     
    return Response(
        {
            "success": True,
            "total_images": len(uploaded_files),
            "valid_readings": valid_count,
            "invalid_readings": invalid_count,
            "results": results
        },
        status=status.HTTP_200_OK
    )