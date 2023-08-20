import gradio as gr
import os

hf_token = os.environ['GRADIO_API_KEY']

iface = gr.load(name="Augustya/ai-email-subject-question-answering-generator", hf_token=hf_token, src="spaces")
iface.queue(api_open=False).launch(show_api=False)