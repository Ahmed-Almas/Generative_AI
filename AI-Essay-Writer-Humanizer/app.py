
"""Essay writer UI. Run with: python app.py"""

import gradio as gr
from graph import graph


def run(question: str, pdf_file: str | None):
    if not question or not question.strip():
        raise gr.Error(
            "Enter a request, e.g. 'Write an essay about dolphins in 50 words'."
        )

    try:
        result = graph.invoke({
            "question": question.strip(),
            "pdf_path": pdf_file or "",
        })

        return (
            result.get("original_essay", ""),
            result.get("humanized_essay", ""),
            result.get("final_word_count", 0),
            (
                f"Verdict: {result.get('detector_verdict', 'Unknown')}\n"
                f"Score: {result.get('detector_score', 'N/A')}\n"
                f"Confidence: {result.get('detector_confidence', 'N/A')}"
            ),
        )

    except Exception as e:
        raise gr.Error(f"Essay generation failed: {e}") from e


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(
            label="Essay request",
            placeholder="Write me an essay about dolphins in 50 words",
            lines=2,
        ),
        gr.File(
            label="Reference PDF (optional)",
            file_types=[".pdf"],
            type="filepath",
        ),
    ],
    outputs=[
        gr.Textbox(label="Original essay", lines=10),
        gr.Textbox(label="Humanized essay", lines=10),
        gr.Number(label="Humanized essay word count"),
        gr.Textbox(label="AI detector result", lines=4),
    ],
    title="Essay Writer and Humanizer",
    description=(
        "Generate an essay using web research or an uploaded PDF, "
        "rewrite it with ReverseGPT, and check it with your detector API."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
