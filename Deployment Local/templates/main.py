import uvicorn
from typing import Annotated
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path


app = FastAPI(
    title="Email Subject Generator",
    description="A simple API using GPT-2 fine-tuned model to generate email subject",
    version="beta 0.1",
)

templates = Jinja2Templates(directory="templates")
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.absolute() / "templates/static"),
    name="static",
)

model =GPT2LMHeadModel.from_pretrained("model/results/output/")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>', pad_token='<|pad|>', sep_token='<|sep|>')

model.resize_token_embeddings(len(tokenizer))


def clean_response(response):
    print(response)
    lst = response.split('<|sep|>')
    if (len(lst) >= 2):
        response = lst[1].replace("<|endoftext|>","")
    return response

@app.get("/", response_class=HTMLResponse)
async def main(request:Request):
    return templates.TemplateResponse("index.html",{"request": request})


@app.post("/subject", response_class=HTMLResponse)
async def generate_response(request:Request, email: Annotated[str, Form()]='None'):
    prompt = f"<|startoftext|> {email} <|sep|>"
    sample_outputs = model.generate(
        tokenizer(prompt, return_tensors='pt')['input_ids'],
        min_new_tokens = 4,
        max_new_tokens = 12,
        num_beams=5,
        early_stopping=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id)
    subject = clean_response(tokenizer.decode(sample_outputs[0]))
    return templates.TemplateResponse("index.html", {"request":request, "subject": subject,"email": email})

