from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
#Imports required by the service's model
import io
import os
import numpy as np
import json
from PIL import Image
from ultralytics import YOLO

api_description = """
This service will use Yolov8 to analyse the image content according to the selected model type:
- detect: object detection
- segment: object segmentation
- pose: human pose estimation
- classify: image classification
"""

api_summary = """
Yolov8 trained model to detect entities.
"""

api_title = "Yolov8 API."
version = "1.0.0"

settings = get_settings()


class Results:
    def __init__(self, results_temp):
        self.top5 = results_temp[0].probs.top5
        self.top5conf = results_temp[0].probs.top5conf.numpy()
        self.results = {
            "top5": {
                results_temp[0].names[top]: conf
                for top, conf in zip(self.top5, self.top5conf)
            }
        }

    def tojson(self):
        return json.dumps(str(self.results))


class MyService(Service):
    """
    Yolov8 model
    """

    # Any additional fields must be excluded for Pydantic to work
    _logger: Logger
    _model_detect: object
    _model_seg: object
    _model_pose: object
    _model_class: object

    def __init__(self):
        super().__init__(
            name="Yolov8",
            slug="yolov8",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="image",
                    type=[
                        FieldDescriptionType.IMAGE_PNG,
                        FieldDescriptionType.IMAGE_JPEG,
                    ],
                ),
                FieldDescription(name="type", type=[FieldDescriptionType.TEXT_PLAIN]),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.APPLICATION_JSON]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_RECOGNITION,
                    acronym=ExecutionUnitTagAcronym.IMAGE_RECOGNITION,
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/yolov8/",
        )
        self._logger = get_logger(settings)

        self._model_detect = YOLO(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "model/yolov8x.pt"
            )
        )
        self._model_seg = YOLO(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "model/yolov8x-seg.pt"
            )
        )
        self._model_pose = YOLO(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "model/yolov8x-pose.pt"
            )
        )
        self._model_class = YOLO(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "model/yolov8x-cls.pt"
            )
        )

    def process(self, data):
        raw = data["image"].data
        process_type = data["type"].data.decode("utf-8")
        if process_type not in ["detect", "segment", "pose", "classify"]:
            raise Exception("Model type not supported.")

        buff = io.BytesIO(raw)
        img_pil = Image.open(buff)
        img = np.array(img_pil)

        if process_type == "detect":
            results = self._model_detect(img)[0]
        elif process_type == "segment":
            results = self._model_seg(img)[0]
        elif process_type == "pose":
            results = self._model_pose(img)[0]
        elif process_type == "classify":
            results = Results(self._model_class(img))
        else:
            raise Exception("Model type not supported.")

        task_data = TaskData(
            data=results.tojson(), type=FieldDescriptionType.APPLICATION_JSON
        )

        return {"result": task_data}

