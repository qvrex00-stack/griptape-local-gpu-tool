from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

from griptape.artifacts import ImageUrlArtifact, VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File

logger = logging.getLogger("griptape_nodes")

__all__ = ["PoseExtractor"]

TOOL_SERVER_URL = "http://127.0.0.1:8089"


def resolve_file_path(param_val) -> str:
    raw_path = None
    if isinstance(param_val, str): raw_path = param_val
    elif hasattr(param_val, "value"): raw_path = param_val.value
    elif isinstance(param_val, dict) and "value" in param_val: raw_path = param_val["value"]
    if not raw_path:
        raise ValueError("Could not extract file path.")
    try:
        resolved = File(raw_path).resolve()
        path = resolved.location if hasattr(resolved, "location") else str(resolved)
    except Exception:
        path = raw_path
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")
    return path


class PoseExtractor(BaseNode):
    """Extract human pose skeleton from image or video using OpenPose (via tool server)."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.category = "Image Nodes"
        self.description = "Extract human pose skeleton from image or video using OpenPose."

        self.add_parameter(ParameterImage(name="input_image", tooltip="Source image (optional if using video)", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}))
        self.add_parameter(ParameterVideo(name="input_video", tooltip="Source video (optional if using image)", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}))
        self.add_parameter(ParameterBool(name="include_body", default_value=True, tooltip="Extract body skeleton", allow_output=False))
        self.add_parameter(ParameterBool(name="include_hand", default_value=False, tooltip="Extract hand skeleton", allow_output=False))
        self.add_parameter(ParameterBool(name="include_face", default_value=False, tooltip="Extract face landmarks", allow_output=False))
        self.add_parameter(ParameterImage(name="output_image", tooltip="Pose skeleton (image input)", allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY}, settable=False, pulse_on_run=True))
        self.add_parameter(ParameterVideo(name="output_video", tooltip="Pose video (video input)", allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY}, settable=False, pulse_on_run=True))
        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="pose_output.mp4")
        self._output_file.add_parameter()

    def process(self) -> AsyncResult[None]:
        self.parameter_output_values["output_image"] = None
        self.parameter_output_values["output_video"] = None

        import requests
        from PIL import Image

        image_param   = self.get_parameter_value("input_image")
        video_param   = self.get_parameter_value("input_video")
        include_body  = self.get_parameter_value("include_body") or True
        include_hand  = self.get_parameter_value("include_hand") or False
        include_face  = self.get_parameter_value("include_face") or False

        if not image_param and not video_param:
            raise ValueError("input_image or input_video is required.")

        try:
            requests.get(f"{TOOL_SERVER_URL}/health", timeout=3)
        except Exception:
            raise RuntimeError("Tool server is not running! (port 8089)")

        def pil_to_b64(img):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

        if video_param:
            video_path = resolve_file_path(video_param)
            resp = requests.post(f"{TOOL_SERVER_URL}/pose_video",
                                 json={"video_path": video_path, "include_body": include_body,
                                       "include_hand": include_hand, "include_face": include_face},
                                 timeout=1800)
            resp.raise_for_status()
            out_path = resp.json().get("video_path", "")
            if not out_path or not os.path.exists(out_path):
                raise RuntimeError(f"Output video not found: {out_path}")
            artifact = VideoUrlArtifact(out_path)
            self.set_parameter_value("output_video", artifact)
            self.publish_update_to_parameter("output_video", artifact)
        else:
            img_path  = resolve_file_path(image_param)
            pil_image = Image.open(img_path).convert("RGB")
            resp = requests.post(f"{TOOL_SERVER_URL}/pose",
                                 json={"image_b64": pil_to_b64(pil_image), "detector": "openpose",
                                       "include_body": include_body, "include_hand": include_hand,
                                       "include_face": include_face},
                                 timeout=120)
            resp.raise_for_status()
            img_bytes = base64.b64decode(resp.json()["image_b64"])
            saved = self._output_file.build_file().write_bytes(img_bytes)
            artifact = ImageUrlArtifact(saved.location)
            self.set_parameter_value("output_image", artifact)
            self.publish_update_to_parameter("output_image", artifact)
