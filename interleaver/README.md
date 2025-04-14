![](../figures/interleaved.png)

Create and visualize interleaved slides-transcript representation of multimodal presentations. Our work highlighted that this representation offers cost-effective performance for summarization of multimodal presentations using VLMs.

[Gradio app](https://huggingface.co/spaces/gigant/slide-presentation-viz) to visualize interleaved multimodal presentations for the TIB benchmark.

# Usage

```python
from interleaved import get_interleaved

interleaved_content = get_interleaved(path_to_transcript, path_to_slides_json, path_to_slides_pdf)
```

# Citation

If you use an interleaved slides-transcript representation for multimodal presentations in your work, consider citing our paper that showed it is a cost-effective representation for summarization:

```
@inproceedings{
  TODO
}
```
