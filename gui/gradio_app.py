import gradio as gr
import requests
import os
import logging

# 静音 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

API_URL_CHAT = "http://127.0.0.1:5000/v1/chat/completions"
API_URL_MASK = "http://127.0.0.1:5000/v1/mask/inference"

def query_chat(messages, model="default-model", max_tokens=200, temperature=0.7, top_p=1.0):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    try:
        response = requests.post(API_URL_CHAT, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        return answer
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def query_mask(input_data, model="default-mask-model", **params):
    payload = {
        "model": model,
        "input": input_data,
        "params": params
    }
    try:
        response = requests.post(API_URL_MASK, json=payload)
        response.raise_for_status()
        data = response.json()
        output = data["output"]
        return output
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def main():
    with gr.Blocks() as demo:
        gr.Markdown("LLM Inference Demo")

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column():
                    model = gr.Textbox(label="Model", value="default-model")
                    max_tokens = gr.Slider(label="Max Tokens", minimum=50, maximum=1000, step=50, value=200)
                    temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.7)
                    top_p = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, step=0.1, value=1.0)

                with gr.Column():
                    chatbot = gr.Chatbot()
                    user_input = gr.Textbox(label="Your Message")

            def respond(message, chat_history, model, max_tokens, temperature, top_p):
                print("Generating response...")
                messages = [{"role": "system", "content": "你是一个有帮助的助手。"}] + \
                           [{"role": "user", "content": m[0]} for m in chat_history] + \
                           [{"role": "assistant", "content": m[1]} for m in chat_history] + \
                           [{"role": "user", "content": message}]
                
                answer = query_chat(messages, model, max_tokens, temperature, top_p)
                chat_history.append((message, answer))
                return "", chat_history

            user_input.submit(respond, 
                              inputs=[user_input, chatbot, model, max_tokens, temperature, top_p], 
                              outputs=[user_input, chatbot])

        with gr.Tab("Mask Inference"):
            with gr.Row():
                model_mask = gr.Textbox(label="Mask Model", value="default-mask-model")
                input_data = gr.Textbox(label="Input Data")
                param1 = gr.Textbox(label="Param 1")
                param2 = gr.Textbox(label="Param 2")
                # 添加更多参数根据需要

            mask_output = gr.Textbox(label="Mask Output")

            def run_mask(model, input_data, param1, param2):
                print("Generating mask inference output...")
                params = {
                    "param1": param1,
                    "param2": param2
                    # 添加更多参数
                }
                output = query_mask(input_data, model, **params)
                return output

            run_button = gr.Button("Run Mask Inference")
            run_button.click(run_mask, 
                             inputs=[model_mask, input_data, param1, param2], 
                             outputs=mask_output)

    return demo

if __name__ == "__main__":
    print("Loading GUI...")
    demo = main()
    demo.launch(server_name="0.0.0.0", server_port=7860)
