import gradio as gr
import requests

def get_model_response(input_text):
    response = requests.post("http://ollama:11434", json={"input": input_text})
    return response.json().get("output", "Error")

iface = gr.Interface(fn=get_model_response, inputs="text", outputs="text").launch()
