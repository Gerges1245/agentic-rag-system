"""Gradio interface for the Agentic RAG System."""

import os
from pathlib import Path

import gradio as gr

from agentic_rag import agentic_answer
from csv_tools import list_csv_files
from document_tools import list_documents
from ingestion import ingest_document
from rag_core import COURSES


def _error_message(exc):
    message = str(exc)
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        message = message.replace(key, "[hidden]")
    return f"{type(exc).__name__}: {message}"


def upload_file(path, course_name):
    if not path:
        return "Please choose a file first."
    if course_name not in COURSES:
        return "Please select a course."
    try:
        result = ingest_document(path, COURSES[course_name])
        return (
            f"File added: {result['file']}\n"
            f"Processed chunks: {result['processed_chunks']}\n"
            f"New chunks: {result['new_chunks']}\n"
            f"Total chunks: {result['total_chunks']}\n\n"
            "Use the saved filename when asking for a document summary."
        )
    except Exception as exc:
        return "Upload failed: " + _error_message(exc)


def available_sources(course_name):
    if course_name not in COURSES:
        return "Please select a course."
    try:
        course = COURSES[course_name]
        documents = list_documents(course)
        csv_files = list_csv_files(course)
        lines = ["Indexed documents (PDF, DOCX, TXT):"]
        for item in documents["documents"]:
            lines.append(f"- {item['source']} ({item['type']}, {item['indexed_chunks']} chunks)")
        if not documents["documents"]:
            lines.append("- None")
        lines.append("\nUploaded CSV files:")
        for item in csv_files["files"]:
            lines.append(f"- {item['source']}")
        if not csv_files["files"]:
            lines.append("- None")
        if csv_files.get("truncated"):
            lines.append("Showing the first 100 CSV files.")
        return "\n".join(lines)
    except Exception as exc:
        return "Could not list sources: " + _error_message(exc)


def ask_question(question, course_name, file_type):
    if not question or not question.strip():
        return "Please enter a question.", "", "", ""
    if course_name not in COURSES:
        return "Please select a course.", "", "", ""
    try:
        result = agentic_answer(
            question=question.strip(), course=COURSES[course_name],
            file_type=file_type, max_steps=6,
        )
        answer = f"Status: {result['status']}\n\n{result['answer']}"
        plan = "\n".join(f"{index}. {step}" for index, step in enumerate(result['plan'], 1))
        actions = []
        for action in result["actions"]:
            actions.append(
                f"{action['step']}. {action['tool']} [{action['status']}]\n"
                f"   Arguments: {action['arguments']}\n"
                f"   Result: {action['summary']}"
            )
        action_text = "\n\n".join(actions) or "No tools were executed."
        action_text += f"\n\nTool steps: {result['tool_steps']} | Stop reason: {result['stop_reason']}"
        sources = []
        for source in result["sources"]:
            page = source.get("page", -1)
            page_text = str(page) if isinstance(page, int) and page >= 1 else "N/A"
            sources.append(
                f"[{source['citation_id']}] {source['source']} | Page: {page_text}\n"
                f"Quote: {source.get('supporting_quote', '')}"
            )
        return answer, plan or "No plan returned.", action_text, "\n\n".join(sources) or "No sources returned."
    except Exception as exc:
        return "Request failed: " + _error_message(exc), "", "", ""


CUSTOM_CSS = """
.gradio-container { max-width: 1440px !important; }
#hero {
    background: linear-gradient(120deg, #102a56 0%, #1254a5 60%, #168b9b 100%);
    border-radius: 22px; padding: 30px 34px; margin-bottom: 18px;
    box-shadow: 0 16px 38px rgba(16, 42, 86, .18);
}
#hero h1 { color: white !important; font-size: 2.05rem; margin: 5px 0 8px; }
#hero p { color: #dfedf9 !important; margin: 0; max-width: 760px; }
#hero .eyebrow { font-size: .72rem; letter-spacing: .16em; font-weight: 700; color: #a8e6f1 !important; }
.section-card { border: 1px solid var(--border-color-primary); border-radius: 18px; padding: 20px !important; background: var(--background-fill-primary); }
.section-heading { margin-bottom: 3px !important; }
.helper-copy { opacity: .75; font-size: .9rem; }
#ask-button { min-height: 48px; font-weight: 700; }
@media (max-width: 600px) { #hero { padding: 23px 20px; } #hero h1 { font-size: 1.6rem; } }
"""


def build_app():
    default_course = "Database Systems" if "Database Systems" in COURSES else next(iter(COURSES))
    with gr.Blocks(title="Agentic RAG System") as app:
        gr.HTML("""<div id="hero"><p class="eyebrow">KNOWLEDGE ASSISTANT</p>
            <h1>Agentic RAG System</h1>
            <p>Ask better questions. Explore your documents with answers backed by sources.</p></div>""")
        with gr.Row():
            course = gr.Dropdown(choices=list(COURSES), value=default_course, label="Course", scale=2)
            file_type = gr.Dropdown(choices=["All Files", "PDF", "DOCX", "CSV", "TXT"],
                                    value="All Files", label="File type", scale=1)

        with gr.Tab("Ask the agent"):
            with gr.Row():
                with gr.Column(scale=5, min_width=320, elem_classes="section-card"):
                    gr.Markdown("### Your question", elem_classes="section-heading")
                    gr.Markdown("Choose a course above, then ask about your materials.", elem_classes="helper-copy")
                    question = gr.Textbox(label="Question", lines=4,
                                          placeholder="What would you like to know?")
                    with gr.Row():
                        ask_button = gr.Button("Ask agent", variant="primary", elem_id="ask-button", scale=3)
                        clear_button = gr.Button("Clear", scale=1)
                    gr.Markdown("**Try an example**", elem_classes="helper-copy")
                    with gr.Row():
                        example_search = gr.Button("Explain a primary key", size="sm")
                        example_calculate = gr.Button("Calculate an expression", size="sm")
                    with gr.Accordion("Browse available sources", open=False):
                        gr.Markdown("Copy the saved filename when asking for a document summary.", elem_classes="helper-copy")
                        sources_button = gr.Button("Show available sources", size="sm")
                        inventory = gr.Textbox(label="Available sources", lines=6, interactive=False)

                with gr.Column(scale=7, min_width=320, elem_classes="section-card"):
                    gr.Markdown("### Answer & evidence", elem_classes="section-heading")
                    gr.Markdown("Review the answer alongside its supporting sources.", elem_classes="helper-copy")
                    answer = gr.Textbox(label="Answer", lines=7, interactive=False)
                    citations = gr.Textbox(label="Sources and supporting quotes", lines=7, interactive=False)
                    with gr.Accordion("Agent plan and actions", open=False):
                        plan = gr.Textbox(label="Agent plan", lines=4, interactive=False)
                        actions = gr.Textbox(label="Actions performed", lines=8, interactive=False)

            sources_button.click(available_sources, inputs=[course], outputs=[inventory])
            ask_button.click(ask_question, inputs=[question, course, file_type], outputs=[answer, plan, actions, citations])
            question.submit(ask_question, inputs=[question, course, file_type], outputs=[answer, plan, actions, citations])
            clear_button.click(lambda: ("", "", "", "", ""), outputs=[question, answer, plan, actions, citations])
            example_search.click(lambda: "What is a primary key?", outputs=[question])
            example_calculate.click(lambda: "Calculate (120 + 80) / 4.", outputs=[question])

        with gr.Tab("Upload documents"):
            with gr.Row():
                with gr.Column(scale=5, min_width=320, elem_classes="section-card"):
                    gr.Markdown("### Add to your knowledge base")
                    gr.Markdown("Your document will be added to the course selected above. Supported formats: PDF, DOCX, CSV, and TXT.", elem_classes="helper-copy")
                    upload = gr.File(label="Choose a document", file_types=[".pdf", ".docx", ".csv", ".txt"], type="filepath")
                    upload_button = gr.Button("Add document", variant="primary")
                with gr.Column(scale=7, min_width=320, elem_classes="section-card"):
                    gr.Markdown("### Upload result")
                    gr.Markdown("The saved filename is useful when asking for a summary or CSV analysis.", elem_classes="helper-copy")
                    upload_result = gr.Textbox(label="Status", lines=7, interactive=False)
            upload_button.click(upload_file, inputs=[upload, course], outputs=[upload_result])
    return app


if __name__ == "__main__":
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="cyan", neutral_hue="slate")
    build_app().queue().launch(theme=theme, css=CUSTOM_CSS)
