
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
get_slides(filepath, h, save_folder, zscore_threshold=4) #save_imgs=True to save the frames as individual images
```
