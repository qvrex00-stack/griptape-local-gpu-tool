from __future__ import annotations

import logging
import os
from typing import Any

from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File
from griptape_nodes.traits.clamp import Clamp
from griptape_nodes.traits.options import Options

logger = logging.getLogger("griptape_nodes")

__all__ = ["PoseSmoothing"]

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


class PoseSmoothing(BaseNode):
    """
    OpenPose keypoint를 시간축으로 스무딩하여 과격한 동작에서
    팔다리가 튀거나 사라지는 현상을 보정한 pose 영상을 생성한다.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.category = "Video"
        self.description = "Smooth OpenPose keypoints across frames to fix jitter and missing limbs in fast action sequences."

        self.add_parameter(ParameterVideo(name="input_video", tooltip="원본 영상", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}))
        self.add_parameter(ParameterBool(name="include_body", default_value=True, tooltip="바디 스켈레톤 추출", allow_output=False))
        self.add_parameter(ParameterBool(name="include_hand", default_value=False, tooltip="손 스켈레톤 추출", allow_output=False))
        self.add_parameter(ParameterBool(name="include_face", default_value=False, tooltip="얼굴 랜드마크 추출", allow_output=False))
        self.add_parameter(ParameterString(name="smooth_method", default_value="gaussian",
                                           tooltip="gaussian: 전체 평탄화 / savgol: 피크 보존",
                                           allow_output=False, traits={Options(choices=["gaussian", "savgol"])}))
        self.add_parameter(ParameterInt(name="smooth_window", default_value=5,
                                        tooltip="스무딩 강도 (3~31, 클수록 부드러움)",
                                        allow_output=False, traits={Clamp(min_val=3, max_val=31)}))
        self.add_parameter(ParameterFloat(name="jump_threshold", default_value=0.05,
                                          tooltip="이상치 감지 민감도 (0.01~0.5)",
                                          allow_output=False, traits={Clamp(min_val=0.01, max_val=0.5)}))
        self.add_parameter(ParameterFloat(name="confidence_threshold", default_value=0.3,
                                          tooltip="이 값 이하 keypoint는 보간 처리",
                                          allow_output=False, traits={Clamp(min_val=0.0, max_val=1.0)}))
        self.add_parameter(ParameterVideo(name="output_video", tooltip="스무딩된 pose 영상",
                                          allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                                          settable=False, pulse_on_run=True))
        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="pose_smooth_output.mp4")
        self._output_file.add_parameter()

    def process(self) -> AsyncResult[None]:
        self.parameter_output_values["output_video"] = None
        import requests

        video_param          = self.get_parameter_value("input_video")
        include_body         = self.get_parameter_value("include_body") or True
        include_hand         = self.get_parameter_value("include_hand") or False
        include_face         = self.get_parameter_value("include_face") or False
        smooth_method        = self.get_parameter_value("smooth_method") or "gaussian"
        smooth_window        = self.get_parameter_value("smooth_window") or 5
        jump_threshold       = self.get_parameter_value("jump_threshold") or 0.05
        confidence_threshold = self.get_parameter_value("confidence_threshold") or 0.3

        if not video_param:
            raise ValueError("input_video is required.")

        try:
            requests.get(f"{TOOL_SERVER_URL}/health", timeout=3)
        except Exception:
            raise RuntimeError("Tool server is not running! (port 8089)")

        video_path = resolve_file_path(video_param)
        resp = requests.post(f"{TOOL_SERVER_URL}/pose_smooth_video", json={
            "video_path": video_path,
            "include_body": include_body, "include_hand": include_hand, "include_face": include_face,
            "smooth_method": smooth_method, "smooth_window": smooth_window,
            "jump_threshold": float(jump_threshold), "confidence_threshold": float(confidence_threshold),
        }, timeout=3600)
        resp.raise_for_status()

        result   = resp.json()
        out_path = result.get("video_path", "")
        logger.info(f"PoseSmoothing done: {result.get('frame_count')} frames → {out_path}")

        if not out_path or not os.path.exists(out_path):
            raise RuntimeError(f"Output video not found: {out_path}")

        artifact = VideoUrlArtifact(out_path)
        self.set_parameter_value("output_video", artifact)
        self.publish_update_to_parameter("output_video", artifact)
