import io
from diffusers.utils import load_image
from huggingface_hub.inference_api import InferenceApi

def image_to_bytes(img_url):
    img_byte = io.BytesIO()
    load_image(img_url).save(img_byte, format="jpeg")
    img_data = img_byte.getvalue()
    return img_data

inference = InferenceApi("lambdalabs/sd-image-variations-diffusers", token="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")
result = inference(data=image_to_bytes("https://raw.githubusercontent.com/justinpinkney/stable-diffusion/main/assets/im-vars-thin.jpg"))
print(result)