"""Small web demo.   python app.py   → http://127.0.0.1:7860"""

import gradio as gr

from generate import load, story

model, tok, device = load()


def run(prompt, temperature, top_k, top_p, max_tokens):
    text = story(model, tok, device, prompt, int(max_tokens), temperature, int(top_k), top_p)
    ids = tok.encode(prompt)
    pieces = " | ".join(tok.decode([i]) for i in ids)
    return text, f"{len(ids)} tokens: {pieces}"


with gr.Blocks(title="GPT from scratch") as demo:
    gr.Markdown(f"## GPT from scratch\nA {model.num_params() / 1e6:.1f}M-parameter transformer and BPE tokenizer, "
                "written from scratch and trained on TinyStories on a MacBook.")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox("Once upon a time, a little robot", label="Story start")
            temperature = gr.Slider(0.1, 1.5, 0.8, label="Temperature (randomness)")
            top_k = gr.Slider(0, 200, 50, step=1, label="Top-k (0 = off)")
            top_p = gr.Slider(0.1, 1.0, 0.95, label="Top-p / nucleus")
            max_tokens = gr.Slider(50, 500, 250, step=10, label="Max new tokens")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Story", lines=14)
            toks = gr.Textbox(label="How the tokenizer splits your prompt")
    btn.click(run, [prompt, temperature, top_k, top_p, max_tokens], [out, toks])

if __name__ == "__main__":
    demo.launch()
