![](../figures/interleaved.png)

Create and visualize interleaved slides-transcript representation of multimodal presentations.

[Gradio app](https://huggingface.co/spaces/gigant/slide-presentation-viz) to visualize interleaved multimodal presentations for the TIB benchmark.

# Usage

```python
from interleaved import get_interleaved

interleaved_content = get_interleaved(path_to_transcript, path_to_slides_json, path_to_slides_pdf)
```
