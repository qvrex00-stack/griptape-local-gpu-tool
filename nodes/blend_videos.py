from __future__ import annotations

import logging
import os
from typing import Any

from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File
from griptape_nodes.traits.clamp import Clamp

logger = logging.getLogger("griptape_nodes")

__all__ = ["BlendVideos"]

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


class BlendVideos(BaseNode):
    """
    Depth 비디오와 Pose 비디오를 픽셀 레벨로 알파 블렌딩.
    Wan Video Generation의 control_video로 사용.
    result = depth × (1 - alpha) + pose × alpha
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.category = "Video"
        self.description = "Blend depth video and pose video for Wan control_video. alpha=0.5 balances depth and pose."

        self.add_parameter(ParameterVideo(name="depth_video", tooltip="Depth Extractor 출력 영상",
                                          allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}))
        self.add_parameter(ParameterVideo(name="pose_video", tooltip="Pose Extractor 또는 Pose Smoothing 출력 영상",
                                          allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}))
        self.add_parameter(ParameterFloat(name="alpha", default_value=0.5,
                                          tooltip="0.0=depth 100% / 0.5=균형(권장) / 1.0=pose 100%",
                                          allow_output=False, traits={Clamp(min_val=0.0, max_val=1.0)}))
        self.add_parameter(ParameterVideo(name="output_video", tooltip="블렌딩된 control 영상",
                                          allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                                          settable=False, pulse_on_run=True))
        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="blended_control.mp4")
        self._output_file.add_parameter()

    def process(self) -> AsyncResult[None]:
        self.parameter_output_values["output_video"] = None
        import requests

        depth_video = self.get_parameter_value("depth_video")
        pose_video  = self.get_parameter_value("pose_video")
        alpha       = self.get_parameter_value("alpha") or 0.5

        if not depth_video: raise ValueError("depth_video is required.")
        if not pose_video:  raise ValueError("pose_video is required.")

        try:
            requests.get(f"{TOOL_SERVER_URL}/health", timeout=3)
        except Exception:
            raise RuntimeError("Tool server is not running! (port 8089)")

        resp = requests.post(f"{TOOL_SERVER_URL}/blend_video", json={
            "depth_video_path": resolve_file_path(depth_video),
            "pose_video_path":  resolve_file_path(pose_video),
            "alpha": float(alpha),
        }, timeout=1800)
        resp.raise_for_status()

        out_path = resp.json().get("video_path", "")
        logger.info(f"BlendVideos done: {out_path}")

        if not out_path or not os.path.exists(out_path):
            raise RuntimeError(f"Output video not found: {out_path}")

        artifact = VideoUrlArtifact(out_path)
        self.set_parameter_value("output_video", artifact)
        self.publish_update_to_parameter("output_video", artifact)
