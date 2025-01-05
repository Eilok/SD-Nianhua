import gradio as gr
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from peft import PeftConfig, PeftModel, set_peft_model_state_dict
import cv2
from PIL import Image

def image_to_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    edges_image = Image.fromarray(edges)

    prompt = "a traditional Chinese New Year painting style of a warrior figure, vibrant colors, detailed lines"
    generated_image = pipe(
        prompt, 
        image=edges_image, 
        num_inference_steps=120
    ).images[0]

    return generated_image

lora_path = "./post_model/AdaLora/epoch10"
# autodl path
# control_path = "../autodl-tmp/models--lllyasviel--control_v11p_sd15_canny/snapshots/115a470d547982438f70198e353a921996e2e819"
# SD_path = "../autodl-tmp/runwayml--stable-diffusion/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1"

# local path
SD_path = r'C:\Users\admin\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\f03de327dd89b501a01da37fc5240cf4fdba85a1'
control_path = r'C:\Users\admin\.cache\huggingface\hub\models--lllyasviel--control_v11p_sd15_canny\snapshots\115a470d547982438f70198e353a921996e2e819'

# load ControlNet and Stable Diffusion
controlnet = ControlNetModel.from_pretrained(control_path)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    SD_path, controlnet=controlnet
).to("cuda")
# pipe.load_lora_weights(lora_path)
fine_model = PeftModel.from_pretrained(pipe.unet, lora_path)
set_peft_model_state_dict(pipe.unet, fine_model.state_dict())
pipe.unet = fine_model.model
pipe.safety_checker = None  # 禁用安全检查器


demo = gr.Interface(
    fn=image_to_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="pil"),
    title="木版年画生成",
    description="请上传一张人物肖像来生成对应的木版年画",
)
demo.launch(share=True)