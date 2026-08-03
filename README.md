# Griptape Local GPU Tool

Depth Estimation, Pose Extraction, Pose Smoothing, Video Blending을  
Griptape Nodes에서 로컬 GPU로 실행하기 위한 노드 패키지입니다.

> **이미지 생성 노드 (FLUX, LaMa, Real-ESRGAN, Kontext):**  
> [griptape-local-gpu-nodes](https://github.com/qvrex00-stack/griptape-local-gpu-nodes)

## 포함된 노드

| 노드 | 설명 |
|------|------|
| **Depth Extractor** | Depth Anything V2로 이미지/비디오 깊이맵 추출 |
| **Pose Extractor** | OpenPose로 이미지/비디오 포즈 스켈레톤 추출 |
| **Pose Smoothing** | 비디오 포즈 키포인트 스무딩 (튀는 관절 보정) |
| **Blend Videos** | Depth + Pose 비디오 알파 블렌딩 (Wan control_video용) |

## 구조

```
griptape-local-gpu-tool/
├── tool_server/
│   ├── server.py           ← FastAPI 서버 (port 8089)
│   └── requirements.txt
├── nodes/
│   ├── depth_extractor.py
│   ├── pose_extractor.py
│   ├── pose_smoothing.py
│   └── blend_videos.py
├── griptape-nodes-library-tools.json
├── install.py
└── README.md
```

## 설치

```powershell
git clone https://github.com/qvrex00-stack/griptape-local-gpu-tool.git
cd griptape-local-gpu-tool
python install.py --griptape-dir "C:\Foundry\Griptape"
```

설치 후 Griptape에서 라이브러리 등록:
```
<griptape_dir>\libraries\griptape-nodes-library-tools
```

## 엔드포인트 (port 8089)

| 엔드포인트 | 설명 |
|-----------|------|
| `GET  /health`            | 서버 상태 확인 |
| `POST /depth`             | 이미지 깊이맵 |
| `POST /depth_video`       | 비디오 깊이맵 |
| `POST /pose`              | 이미지 포즈 |
| `POST /pose_video`        | 비디오 포즈 |
| `POST /pose_smooth_video` | 포즈 스무딩 |
| `POST /blend_video`       | Depth+Pose 블렌딩 |

## 변경 이력

| 날짜 | 버전 | 내용 |
|------|------|------|
| 2026-07-21 | v1.0 | Depth Extractor, Pose Extractor |
| 2026-07-28 | v1.1 | Pose Smoothing, Blend Videos 추가 |
| 2026-08-03 | v2.0 | 레포 독립 구조로 재구성, install.py 추가 |

## 라이선스

MIT License
