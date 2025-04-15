import jax.numpy as jnp
import numpy as np
from PIL import Image
from decord import VideoReader
from decord import cpu
import json

from phash import batch_phash, hash_dist

BATCH_SIZE = 64
DOWNSAMPLE = 10
FOLDER_PATH = "../demo/slides"

def binary_array_to_hex(arr):
	"""
	Function to make a hex string out of a binary array.
	"""
	bit_string = ''.join(str(b) for b in 1 * arr.flatten())
	width = int(jnp.ceil(len(bit_string) / 4))
	return '{:0>{width}x}'.format(int(bit_string, 2), width=width)

def compute_batch_hashes(vid_path):
    # batched computation of hashes in the video
    kwargs={"width": 64, "height":64}
    vr = VideoReader(vid_path, ctx=cpu(0), **kwargs)
    avg_fps = vr.get_avg_fps()
    hashes = []
    h_prev = None
    batch = []
    for i in range(0, len(vr), DOWNSAMPLE * BATCH_SIZE):
        ids = [id for id in range(i, min(i + DOWNSAMPLE * BATCH_SIZE, len(vr)), DOWNSAMPLE)]
        vr.seek(0)
        batch = jnp.array(vr.get_batch(ids).asnumpy())
        batch_h =  batch_phash(batch)
        for i in range(len(ids)):
            h = batch_h[i]
            if h_prev == None:
                h_prev=h
            hashes.append({"frame_id":ids[i], "timestamp": ids[i]/avg_fps, "hash": binary_array_to_hex(h), "distance": int(hash_dist(h, h_prev))})
            h_prev = h
    return hashes

def compute_zscore(hashes):
    # compute the zscore from the computed frame to frame hash distances
    distances = [h["distance"] for h in hashes]
    mean = np.mean(distances)
    std = np.std(distances)
    return [(h["distance"] - mean) / std for h in hashes]


def get_bounds(hashes, zscore_threshold=3):
    # get the boundaries between slides according to the hashes wrt the zscore threshold
    bounds = []
    i_start = 0

    zscores = compute_zscore(hashes)

    for i  in range(len(hashes)):
        if zscores[i] > zscore_threshold:
            # if the distance is multiple standard deviations above the observed mean, we detect a slide change
            bounds.append({"start": hashes[i_start]["timestamp"], "end": hashes[i-1]["timestamp"]})
            i_start=i

    bounds.append({"start": hashes[i_start]["timestamp"], "end": hashes[-1]["timestamp"]})
    return bounds

def get_slides(vid_path, hashes, zscore_threshold, save_folder=FOLDER_PATH, save_imgs=False, save_pdf=True, save_json=True):
    # extract the slides according to the hashes wrt the zscore threshold
    vr = VideoReader(vid_path, ctx=cpu(0))
    slideshow = []
    i_start = 0
    slides = []
    zscores = compute_zscore(hashes)
    for i  in range(len(hashes)):
        if zscores[i] > zscore_threshold:
            # if the distance is multiple standard deviations above the observed mean, we detect a slide change
            path=f'{save_folder}/{vid_path.split("/")[-1].split(".")[0]}_{i_start}_{i-1}.png'
            index_mid = int((i + i_start)/2)
            slides.append(Image.fromarray(vr[hashes[index_mid]["frame_id"]].asnumpy()))
            if save_imgs:
                slides[-1].save(path)
            slideshow.append({"slide": path, "start": i_start, "end": i-1})
            i_start=i
    path=f'{save_folder}/{vid_path.split("/")[-1].split(".")[0]}_{i_start}_{len(vr)-1}.png'
    if save_imgs:
        Image.fromarray(vr[-1].asnumpy()).save(path)
    slides.append(Image.fromarray(vr[-1].asnumpy()))
    if save_pdf:
        slides[0].save(
        f'{save_folder}/{vid_path.split("/")[-1].split(".")[0]}_slideshow.pdf', "PDF" ,resolution=100.0, save_all=True, append_images=slides[1:]
        )
    slideshow.append({"slide": path, "start": i_start, "end": len(vr)-1})
    if save_json:
        with open(f'{save_folder}/{vid_path.split("/")[-1].split(".")[0]}_slideshow.json', 'w') as f:
            json.dump(slideshow, f)


if __name__ == "__main__":
    filepath = "../demo/_YHDxyO9W8Q.mp4"
    h = compute_batch_hashes(filepath)
    bounds = get_bounds(h, zscore_threshold=4)
    print(bounds)
    get_slides(filepath, h, zscore_threshold=4)
