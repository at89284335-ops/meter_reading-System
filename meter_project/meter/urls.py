from django.urls import path

from .views import (
    meter_read_api,
    meter_batch_api
)


urlpatterns = [

    # Single meter image
    path(
        "api/v1/meter/read",
        meter_read_api,
        name="meter-read"
    ),

    # Multiple meter images
    path(
        "api/v1/meter/read/batch",
        meter_batch_api,
        name="meter-batch"
    ),
]