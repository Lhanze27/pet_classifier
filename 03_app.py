"""
Pet Breed Classifier - Gradio UI
Run: python 03_app.py
"""
import os
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import gradio as gr

# ----- GPU memory growth (same pattern as training scripts) -----
for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# ----- Load model + class names once -----
MODEL_PATH = 'outputs/models/final_model.keras'
CLASS_NAMES_PATH = 'outputs/class_names.txt'
IMG_SIZE = 224
LOW_CONF_THRESHOLD = 0.50

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"{MODEL_PATH} not found. Run 01_train_model.py first."
    )

print("Loading model...")
model = keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, 'r') as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

# Oxford-IIIT convention: capitalized first letter = cat, lowercase = dog
CAT_BREEDS = sorted([c for c in CLASS_NAMES if c[0].isupper()])
DOG_BREEDS = sorted([c for c in CLASS_NAMES if c[0].islower()])


def pretty(name: str) -> str:
    return name.replace('_', ' ').title()


def species_of(name: str) -> str:
    return "Cat" if name[0].isupper() else "Dog"


# ----- Prediction -----
def predict(img):
    if img is None:
        return {}, "", ""

    if isinstance(img, Image.Image):
        img = np.array(img)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]

    x = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    x = tf.cast(x, tf.float32)
    x = tf.expand_dims(x, 0)

    probs = model.predict(x, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:5]

    label_dict = {pretty(CLASS_NAMES[i]): float(probs[i]) for i in top_idx}

    top1_name = CLASS_NAMES[top_idx[0]]
    top1_conf = float(probs[top_idx[0]])
    species = species_of(top1_name)

    species_md = f"**Species** &nbsp;·&nbsp; {species}"

    if top1_conf < LOW_CONF_THRESHOLD:
        warning_md = (
            f"<div class='warn'>Low confidence ({top1_conf*100:.1f}%). "
            f"This may not be one of the 37 known breeds, or the framing / "
            f"lighting may be off.</div>"
        )
    else:
        warning_md = ""

    return label_dict, species_md, warning_md


# ----- Theme + CSS -----
theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#fbf6f2", c100="#f3e6db", c200="#e8cdb5", c300="#d9ae8a",
        c400="#c89167", c500="#b77752", c600="#9e5f3f", c700="#7f4a31",
        c800="#5f3724", c900="#3f2418", c950="#26160e",
    ),
    secondary_hue="stone",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#faf7f2",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#ece5db",
    block_radius="6px",
    button_primary_background_fill="#9e5f3f",
    button_primary_background_fill_hover="#7f4a31",
    button_primary_text_color="#ffffff",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');

.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
footer { display: none !important; }

#hero { padding: 28px 4px 8px 4px; border-bottom: 1px solid #ece5db; margin-bottom: 18px; }
#hero h1 {
    font-family: 'Fraunces', Georgia, serif !important;
    font-weight: 600;
    font-size: 38px;
    letter-spacing: -0.01em;
    color: #2a1d14;
    margin: 0 0 6px 0;
}
#hero .sub { color: #6b5b4c; font-size: 14px; max-width: 640px; line-height: 1.5; }
#hero .paw { color: #9e5f3f; }

#limit-banner {
    background: #fbf6f2;
    border: 1px solid #ece5db;
    border-left: 3px solid #9e5f3f;
    padding: 10px 14px;
    margin-bottom: 14px;
    font-size: 13px;
    color: #6b5b4c;
    border-radius: 4px;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: #6b5b4c !important;
    font-weight: 500 !important;
    padding: 10px 4px !important;
    margin-right: 22px !important;
}
.tab-nav button.selected {
    color: #2a1d14 !important;
    border-bottom-color: #9e5f3f !important;
}

.warn {
    background: #fdf5ef;
    border-left: 3px solid #c89167;
    padding: 10px 12px;
    color: #7f4a31;
    font-size: 13px;
    border-radius: 4px;
    margin-top: 6px;
}

#species-out { font-size: 15px; color: #2a1d14; padding: 6px 0; }

.breed-col h4 {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    color: #2a1d14;
    margin: 0 0 8px 0;
    font-size: 16px;
}
.breed-col ul { columns: 2; column-gap: 24px; padding-left: 16px; margin: 0; }
.breed-col li { font-size: 13px; color: #4a3b2e; line-height: 1.8; break-inside: avoid; }
"""


def breed_list_html(title, breeds, count):
    items = "".join(f"<li>{pretty(b)}</li>" for b in breeds)
    return (
        f"<div class='breed-col'>"
        f"<h4>{title} &nbsp;<span style='color:#9e5f3f;font-weight:500;'>({count})</span></h4>"
        f"<ul>{items}</ul></div>"
    )


# ----- UI -----
with gr.Blocks(title="Pet Breed Classifier") as demo:
    gr.HTML(
        "<div id='hero'>"
        "<h1><span class='paw'>·</span> Pet Breed Classifier</h1>"
        "<div class='sub'>Fine-grained breed recognition for cats and dogs, "
        "trained on the Oxford-IIIT Pet dataset using EfficientNetV2B0 "
        "with two-phase transfer learning.</div>"
        "</div>"
    )

    gr.HTML(
        "<div id='limit-banner'>"
        "Recognizes <b>37 breeds</b> only — 12 cat breeds and 25 dog breeds. "
        "Other animals, mixed breeds, or breeds outside this list will still "
        "produce a guess; rely on the confidence score and the low-confidence warning."
        "</div>"
    )

    with gr.Tabs():
        # ---- Tab 1: Snapshot / Upload ----
        with gr.Tab("Snapshot"):
            with gr.Row():
                with gr.Column(scale=6):
                    snap_input = gr.Image(
                        sources=["upload", "webcam", "clipboard"],
                        type="numpy",
                        label="Drop an image, paste, or snap from your webcam",
                        height=420,
                    )
                    snap_btn = gr.Button("Classify", variant="primary")
                with gr.Column(scale=4):
                    snap_species = gr.Markdown("", elem_id="species-out")
                    snap_label = gr.Label(num_top_classes=5, label="Top 5 breeds")
                    snap_warn = gr.HTML("")

            snap_btn.click(predict, inputs=snap_input,
                           outputs=[snap_label, snap_species, snap_warn])

        # ---- Tab 2: Live camera ----
        with gr.Tab("Live camera"):
            gr.Markdown(
                "Continuous prediction from your webcam. "
                "Hold a pet steady in frame — predictions update a few times per second."
            )
            with gr.Row():
                with gr.Column(scale=6):
                    live_input = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        label="Live feed",
                        height=420,
                    )
                with gr.Column(scale=4):
                    live_species = gr.Markdown("", elem_id="species-out")
                    live_label = gr.Label(num_top_classes=5, label="Top 5 breeds")
                    live_warn = gr.HTML("")

            live_input.stream(
                predict,
                inputs=live_input,
                outputs=[live_label, live_species, live_warn],
                stream_every=0.25,
            )

    with gr.Accordion("Supported breeds (what the model can classify)", open=False):
        with gr.Row():
            gr.HTML(breed_list_html("Cats", CAT_BREEDS, len(CAT_BREEDS)))
            gr.HTML(breed_list_html("Dogs", DOG_BREEDS, len(DOG_BREEDS)))


if __name__ == "__main__":
    demo.queue(max_size=1).launch(theme=theme, css=CSS)
