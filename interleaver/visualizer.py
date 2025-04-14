
import gradio as gr
from PIL import Image
import io
import base64
from datasets import load_dataset

from interleaved import interleave


def create_interleaved_html(content, scale=0.4, max_width=600):
    """
    Creates an HTML string with interleaved images and text segments.
    The images are converted to base64 and embedded directly in the HTML.
    """
    html = []

    for j, segment in enumerate(content):
        # Add the image
        if segment["type"] == "image":
            img = segment["image"]
            img_width = int(img.width * scale)
            img_height = int(img.height * scale)
            if img_width > max_width:
                ratio = max_width / img_width
                img_width = max_width
                img_height = int(img_height * ratio)
            # Convert image to base64
            buffer = io.BytesIO()
            img.resize((img_width, img_height)).save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            html.append(f'<img src="data:image/png;base64,{img_str}" style="max-width: {max_width}px; display: block; margin: 20px auto;">')
            # Add the text segment after the image
        elif segment["type"] == "text":  
            html.append(f'<div style="white-space: pre-wrap;">{segment["text"]}</div>')
    return "".join(html)

def main():
    #App to visualize the interleaved documents, made with Gradio

    # Global variables to keep track of current document
    current_doc_index = 0
    annotations = []

    ds = load_dataset("gigant/tib-bench")["train"]

    choices = [f"{i} | {ds['title'][i]}" for i in range(len(ds))]

    def load_document(index):
        """Load a specific document from the dataset"""
        if 0 <= index < len(ds):
            doc = ds[index]
            content = interleave(doc["keyframes"]["timestamp"], doc["slides"], doc["transcript_segments"])
            return (
                doc["title"],
                doc["abstract"],
                create_interleaved_html(content, scale=0.7),
                choices[index],
            )
        return ("", "", "", "")

    def get_next_document():
        """Get the next document in the dataset"""
        global current_doc_index
        return choices[(current_doc_index + 1) % len(ds)]

    def get_prev_document():
        """Get the previous document in the dataset"""
        global current_doc_index
        return choices[(current_doc_index - 1) % len(ds)]

    def get_selected_document(arg):
        """Get the selected document from the dataset"""
        global current_doc_index
        index = int(arg.split(" | ")[0])
        current_doc_index = index
        return load_document(current_doc_index)

    theme = gr.themes.Ocean()

    with gr.Blocks(theme=theme) as demo:
        gr.Markdown("# Slide Presentation Visualization Tool")
        pres_selection_dd = gr.Dropdown(label="Presentation", value=choices[0], choices=choices)
        with gr.Row():
            with gr.Column():
                body = gr.HTML(max_height=400)

            with gr.Column():
                title = gr.Textbox(label="Title", interactive=False, max_lines=1)
                abstract = gr.Textbox(label="Abstract", interactive=False, max_lines=8)

        # Load first document
        title_val, abstract_val, body_val, choices_val = load_document(current_doc_index)
        title.value = title_val
        abstract.value = abstract_val
        body.value = body_val
        pres_selection_dd.value = choices_val

        pres_selection_dd.change(
            fn=get_selected_document,
            inputs=pres_selection_dd,
            outputs=[title, abstract, body, pres_selection_dd],
        )

        with gr.Row():
            prev_button = gr.Button("Previous Document")
            prev_button.click(fn=get_prev_document, inputs=[], outputs=[pres_selection_dd])
            next_button = gr.Button("Next Document")
            next_button.click(fn=get_next_document, inputs=[], outputs=[pres_selection_dd])

    demo.launch()

if __name__ == "__main__":
    main()