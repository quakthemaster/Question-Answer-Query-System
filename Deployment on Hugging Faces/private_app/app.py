import re
import os
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

hf_token = os.environ['HF_ACCESS_KEY']

model_email =GPT2LMHeadModel.from_pretrained("Augustya/email-subject-generator",
                                       token=hf_token)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>', pad_token='<|pad|>', sep_token='<|sep|>')

model_qa = GPT2LMHeadModel.from_pretrained("Augustya/aiml-question-answering",
                                           token=hf_token)

def clean_subject(response):
    print(response)
    lst = response.split('<|sep|>')
    if (len(lst) >= 2):
        response = lst[1].replace("<|endoftext|>","")
    return response

def generate_subject(email:str):
    prompt = f"<|startoftext|> {email} <|sep|>"
    sample_outputs = model_email.generate(
        tokenizer(prompt, return_tensors='pt')['input_ids'],
        min_new_tokens = 4,
        max_new_tokens = 12,
        num_beams=5,
        early_stopping=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id)
    subject = clean_subject(tokenizer.decode(sample_outputs[0]))
    return subject

def clean_answer(response):
    lst = response.split('<|sep|>')
    if (len(lst) >= 2):
        response = lst[1].replace("<|endoftext|>","").replace("<|pad|>","").replace("<|startoftext|>","")
        response = response.split('___')[0]
    return response


def generate_answer(question: str):
    prompt = f"<|startoftext|> {question} <|sep|>"
    sample_outputs = model_qa.generate(
        tokenizer(prompt, return_tensors='pt')['input_ids'],
        min_new_tokens = 60, 
        max_new_tokens = 200,
        penalty_alpha=0.6, 
        top_k=4, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id)
    answer = clean_answer(tokenizer.decode(sample_outputs[0]))
    return answer

theme = gr.themes.ThemeClass.from_hub("freddyaboulton/dracula_revamped")
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# AI Based Email Subject and QA Generator")
    with gr.Tab("Email Subject Generation"):
        gr.Markdown("To generate subject for your email, copy/paste or type email in the box and click on Generate Subject")
        with gr.Row():
            with gr.Column(scale=2):
                text_input_email = gr.Textbox(lines=8, label="Email")
                text_output_subject = gr.Textbox(label="Generated Subject")
            with gr.Column(scale=1):
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        ## Team 3
                        ```

                        Ayush Mahansaria
                        Anil Sharma
                        Hema Passi
                        Sanidhya Vjiyawat
                        ```
                        ## Model
                        ```
                        Fine tuned gpt2-medium

                        ```
                        ## Training Dataset
                        ```
                        AESLC
                        ```

                        """)
        with gr.Row():
            with gr.Column(scale=2):
                btn_generate_subject = gr.Button("Generate Subject")
            with gr.Column():
                gr.Markdown(
                    """
                    """)
    with gr.Tab("Answer Generation"):
        with gr.Column(scale=1):
            gr.Markdown("To generate answer for your question, copy/paste or type question in the box and click on Generate Answer")
            with gr.Row():
                with gr.Column(scale=2):
                    text_input_question = gr.Textbox(label="Question")
                    text_output_answer = gr.Textbox(lines=8, label="Generated Answer")
                with gr.Column(scale=1):
                    with gr.Column(scale=1):
                        gr.Markdown(
                            """
                            ## Team 3
                            ```

                            Ayush Mahansaria
                            Anil Sharma
                            Hema Passi
                            Sanidhya Vjiyawat
                            ```
                            ## Model
                            ```
                            Fine tuned gpt2-medium

                            ```
                            ## Training Dataset
                            ```
                            Custom
                            ```

                            """)
            with gr.Row():
                with gr.Column(scale=2):
                    btn_generated_answer = gr.Button("Generate Answer")
                with gr.Column():
                    gr.Markdown(
                        """
                        """)
    btn_generate_subject.click(generate_subject, inputs=text_input_email, outputs=text_output_subject)

    btn_generated_answer.click(generate_answer, inputs=text_input_question, outputs=text_output_answer)

demo.launch()
