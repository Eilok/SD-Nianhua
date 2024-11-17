The project aims to fine-tune the stable diffusion model using Lora method to generate the Chinese New Year pictures from the portrait photos. \
Two model was used in the project:
- sd-controlnet-canny
- stable-diffusion-v1-5

We can download them from huggingface website:

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

sd_path = 'runwayml/stable-diffusion-v1-5'
controlnet_canny_path = 'lllyasviel/sd-controlnet-canny'

controlnet = ControlNetModel.from_pretrained(controlnet_canny_path)
pipe =StableDiffusionControlNetPipeline.from_pretrained(
    sd_path, controlnet=controlnet
)
```
The original data can be found in [here](None), which was collected from the database of Chinese wooden New Year pictures.
