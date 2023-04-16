import uuid
import gradio as gr
import re
from diffusers.utils import load_image
import requests
from awesome_chat import chat_huggingface

all_messages = []
OPENAI_KEY = ""

def add_message(content, role):
    message = {"role":role, "content":content}
    all_messages.append(message)

def extract_medias(message):
    image_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(jpg|jpeg|tiff|gif|png)")
    image_urls = []
    for match in image_pattern.finditer(message):
        if match.group(0) not in image_urls:
            image_urls.append(match.group(0))

    audio_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(flac|wav)")
    audio_urls = []
    for match in audio_pattern.finditer(message):
        if match.group(0) not in audio_urls:
            audio_urls.append(match.group(0))

    video_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(mp4)")
    video_urls = []
    for match in video_pattern.finditer(message):
        if match.group(0) not in video_urls:
            video_urls.append(match.group(0))

    return image_urls, audio_urls, video_urls

def set_openai_key(openai_key):
    global OPENAI_KEY
    OPENAI_KEY = openai_key
    return OPENAI_KEY

def add_text(messages, message):
    if len(OPENAI_KEY) == 0 or not OPENAI_KEY.startswith("sk-"):
        return messages, "Please set your OpenAI API key first."
    add_message(message, "user")
    messages = messages + [(message, None)]
    image_urls, audio_urls, video_urls = extract_medias(message)

    for image_url in image_urls:
        if not image_url.startswith("http"):
            image_url = "public/" + image_url
        image = load_image(image_url)
        name = f"public/images/{str(uuid.uuid4())[:4]}.jpg" 
        image.save(name)
        messages = messages + [((f"{name}",), None)]
    for audio_url in audio_urls:
        if not audio_url.startswith("http"):
            audio_url = "public/" + audio_url
        ext = audio_url.split(".")[-1]
        name = f"public/audios/{str(uuid.uuid4()[:4])}.{ext}"
        response = requests.get(audio_url)
        with open(name, "wb") as f:
            f.write(response.content)
        messages = messages + [((f"{name}",), None)]
    for video_url in video_urls:
        if not video_url.startswith("http"):
            video_url = "public/" + video_url
        ext = video_url.split(".")[-1]
        name = f"public/audios/{str(uuid.uuid4()[:4])}.{ext}"
        response = requests.get(video_url)
        with open(name, "wb") as f:
            f.write(response.content)
        messages = messages + [((f"{name}",), None)]
    return messages, ""

def bot(messages):
    if len(OPENAI_KEY) == 0 or not OPENAI_KEY.startswith("sk-"):
        return messages
    message = chat_huggingface(all_messages, OPENAI_KEY, "openai")["message"]
    image_urls, audio_urls, video_urls = extract_medias(message)
    add_message(message, "assistant")
    messages[-1][1] = message
    for image_url in image_urls:
        messages = messages + [((None, (f"public/{image_url}",)))]
    for audio_url in audio_urls:
        messages = messages + [((None, (f"public/{audio_url}",)))]
    for video_url in video_urls:
        messages = messages + [((None, (f"public/{video_url}",)))]
    return messages

with gr.Blocks() as demo:
    gr.Markdown("<h2><center>HuggingGPT (Dev)</center></h2>")
    with gr.Row():
        openai_api_key = gr.Textbox(
            show_label=False,
            placeholder="Set your OpenAI API key here and press Enter",
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=500)

    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Enter text and press enter. The url of the multimedia resource must contain the extension name.",
        ).style(container=False)

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )
    openai_api_key.submit(set_openai_key, [openai_api_key], [openai_api_key])

    gr.Examples(
        examples=["Given a collection of image A: /examples/a.jpg, B: /examples/b.jpg, C: /examples/c.jpg, please tell me how many zebras in these picture?",
                    "Please generate a canny image based on /examples/f.jpg",
                    "show me a joke and an image of cat",
                    "what is in the examples/a.jpg",
                    "generate a video and audio about a dog is running on the grass",
                    "based on the /examples/a.jpg, please generate a video and audio",
                    "based on pose of /examples/d.jpg and content of /examples/e.jpg, please show me a new image",
                    ],
        inputs=txt
    )

demo.launch()