import json
from pdf2image import convert_from_path

def interleave(keyframes_timestamps, slides, transcript_segments):
    """
    returns a list that represent the interleaved sequence of the extracted slides and transcript segments
    the list follows the format used by huggingface processor for interleaved VLMs eg Qwen2-VL
    """
    content = []
    segment_i = 0
    for i, timestamp in enumerate(keyframes_timestamps):
        content.append({"type": "image", "image": slides[i]})
        text = ""
        start, end = timestamp[0], timestamp[1]
        while segment_i < len(transcript_segments) and end > transcript_segments[segment_i]["seek"] * 0.01:
            text += transcript_segments[segment_i]["text"]
            segment_i += 1
        if text != "":
            content.append({"type": "text", "text": text})
    if segment_i < len(transcript_segments):
        text = "".join([segment["text"] for segment in transcript_segments[segment_i:]])
        if text != "":
            content[-1]["text"] += text
    return content

def get_interleaved(transcript_file, slideshow_json_file, slideshow_pdf_file):
    with open(transcript_file, "r") as f:
        transcript = json.load(f)
    with open(slideshow_json_file, "r") as f:
        keyframes_info = json.load(f)
    keyframes_timestamps = [(kf["start"], kf["end"]) for kf in keyframes_info]
    slides = convert_from_path(slideshow_pdf_file)
    return interleave(keyframes_timestamps, slides, transcript["segments"])

if __name__ == "__main__":
    interleaved_content = get_interleaved("./../demo/transcript.json", "./../demo/slides/_YHDxyO9W8Q_slideshow.json", "./../demo/slides/_YHDxyO9W8Q_slideshow.pdf")
    print(interleaved_content)