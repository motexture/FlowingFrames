---
license: apache-2.0
---
# FlowingFrames

FlowingFrames is a text-to-video model that leverages past frames for conditioning, enabling the generation of infinite-length videos. It supports flexible resolutions, various configurations for frames and inference steps, and prompt interpolation for creating smooth scene changes.

## Installation

### Clone the Repository

```bash
git clone https://github.com/motexture/FlowingFrames.git
cd FlowingFrames
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
python run.py
```

Visit the provided URL in your browser to interact with the interface and start generating videos.
