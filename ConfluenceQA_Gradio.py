import gradio as gr
from ConfluenceQA import ConfluenceQA

token = "xyz..."


client = ConfluenceQA(config = {
    "confluence_url":"http://localhost:8090",
    "api_key": token,
    "space_key":"PD",
    "persist_directory":"./confluence_docs",
    "model_name":"gpt-4o-mini",
})


def q_and_a(question):
    print("question",question)
    client.vector_db_confluence_docs(True)
    answer = client.answer_confluence(question)
    print("answer", answer)
    return answer

gr.Interface(fn=q_and_a, inputs="text", outputs="text").launch()