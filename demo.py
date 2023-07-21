import gradio as gr
from chatgpt_api import RagChatGPT

chat = RagChatGPT()

demo = gr.Interface(
    fn=chat.reply,
    title="RAG ChatGPT(차량 비상시 응급조치)",
    inputs=gr.Textbox(lines=2, placeholder="질문을 입력하세요..."),
    outputs="text",
)
demo.launch()
