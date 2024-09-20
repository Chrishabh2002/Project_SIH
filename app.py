from flask import Flask, redirect
import gradio as gr
from huggingface_hub import InferenceClient
import PyPDF2
import threading

# Initialize Flask app
app = Flask(__name__)

# Gradio components and functions
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def sanitize_text(text):
    return text.encode("utf-8", "replace").decode("utf-8", "replace")

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    message = sanitize_text(message)
    system_message = sanitize_text(system_message)
    
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": sanitize_text(val[0])})
        if val[1]:
            messages.append({"role": "assistant", "content": sanitize_text(val[1])})

    messages.append({"role": "user", "content": message})

    response = ""

    try:
        for message in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = message.choices[0].delta.content
            response += token
            yield response
    except Exception as e:
        yield f"An error occurred: {str(e)}"

def process_pdf(pdf_file):
    if pdf_file is not None:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += sanitize_text(page.extract_text() or "")
            if pdf_text.strip():
                return pdf_text
            else:
                return "Could not extract text from the PDF."
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    else:
        return "No PDF uploaded."

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

pdf_demo = gr.Interface(
    process_pdf,
    inputs=gr.File(label="Upload PDF"),
    outputs=gr.Textbox(label="Extracted Text"),
)

combined_demo = gr.TabbedInterface([pdf_demo, demo], ["PDF Uploader", "Chatbot"])

# Gradio interface in a separate thread
def run_gradio():
    combined_demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

thread = threading.Thread(target=run_gradio)
thread.start()

# Flask routes
@app.route('/')
def index():
    return redirect("http://localhost:7860")

if __name__ == "__main__":
    app.run(port=5000, debug=True)
