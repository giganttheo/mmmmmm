
![](../figures/zscore_slide_extraction.png)

Slide boundaries are estimated from high perceptual hash distances between consecutive frames.


---
# Usage

```python
from extractor import compute_batch_hashes, get_slides

filepath = "" #path to the video file
save_folder = "" #path to the folder where to save the extracted pdf

#compute the hash for each frame and the distances between consecutive frames
h = compute_batch_hashes(filepath)

#extract the slides and save the slidedeck as a pdf
get_slides(filepath, h, zscore_threshold=4, save_folder=save_folder) #save_imgs=True to save the frames as individual images
```

---
# Demo

[A demo app is available on the HuggingFace hub](https://huggingface.co/spaces/gigant/slideshow_extraction) (WIP).


---
# Benchmark

Evaluation on the videos and manually segmented slides from the [LPM dataset](https://github.com/dondongwon/LPMDataset). Precision, recall, coverage and purity metrics are computed using the [`pyannote-metrics`](https://github.com/pyannote/pyannote-metrics) library. Window diff follows the implementation from [`nltk`](https://github.com/nltk/nltk). When it is relevant, we use a tolerance of 1 second, and a sample rate of 1/tolerance = 1Hz. Our algorithm processes a 1 hour-long video in 2min 15s on CPU. 


| Method | Precision $\uparrow$ | Recall $\uparrow$ | Purity $\uparrow$ | Window Diff $\downarrow$ |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| [`vid2slides`](https://github.com/patrickmineault/vid2slides) (Hidden Markov Model) | 1.00 | 1.15 | 90.35 | 19.56 |
| [`PySceneDetect`](https://github.com/Breakthrough/PySceneDetect) (ContentDetector) | 64.12 | 22.10 | 37.08 | 32.89 |
| phash (zscore-threshold = 4)| **70.65**  | **88.16** | **93.45** | **13.06** |

---
# Citation

If you use this method in your work, you can cite our paper that introduced perceptual hash-based slide extraction from videos:

```
@inproceedings{gigant2023tib,
  title={TIB: A Dataset for Abstractive Summarization of Long Multimodal Videoconference Records},
  author={Gigant, Th{\'e}o and Dufaux, Fr{\'e}d{\'e}ric and Guinaudeau, Camille and Decombas, Marc},
  booktitle={Proceedings of the 20th International Conference on Content-based Multimedia Indexing},
  pages={61--70},
  year={2023}
}
```
