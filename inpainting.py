from PIL import Image
import io
import cv2
import base64
import requests

# A1111 URL
url = "https://db19289ab29abb4f96.gradio.live"

# Read Image in RGB order
img = cv2.imread('apple.png')
base_img = cv2.imread('mask.png')

# Encode into PNG and send to ControlNet
retval, bytes = cv2.imencode('.png', img)
encoded_image = base64.b64encode(bytes).decode('utf-8')

base_retval, base_bytes = cv2.imencode('.png', base_img)
encoded_baseimage = base64.b64encode(base_bytes).decode('utf-8')

# print(encoded_baseimage)


# # A1111 payload
payload = {
    "init_images" : [
        encoded_image
    ],
    "mask": encoded_baseimage,
    "batch_size": 1,
    "cfg_scale": 23,
    "resize_mode": 2,
    "inpainting_mask_invert": 1,
    "inpainting_fill":1,
    "inpaint_full_res": 0,
    "refiner_checkpoint": "v1-5-pruned-emaonly.safetensors [6ce0161689]",
    "prompt": 'an orange',
    "negative_prompt": "",
    "width": 512,
    "height": 512,
    "steps": 20,
    "sampler_index": "DPM++ 2M Karras",

}


# # Trigger Generation
response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)

# Read results
r = response.json()
result = r['images'][0]
image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
image.save('output123.png')