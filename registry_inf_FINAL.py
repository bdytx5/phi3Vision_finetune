import weave
import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import base64
from pathlib import Path
import wandb 

# Initialize Weights & Biases run
run = wandb.init(project='burberry-product-price-prediction')
artifact = run.use_artifact('byyoung3/model-registry/phi3-v-burberry:v0', type='model')
artifact_dir = artifact.download()
print(f"Artifact downloaded to: {artifact_dir}")

model_id = "microsoft/Phi-3-vision-128k-instruct" 

try:
    model = AutoModelForCausalLM.from_pretrained(
        artifact_dir, 
        torch_dtype=torch.float16, 
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
except Exception as e:
    print(f"Error loading model or processor: {e}")
    raise

# Ensure the model is on the correct device
device = 'cuda'
model.to(device)

# Function to convert image to data URL
EXT_TO_MIMETYPE = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.svg': 'image/svg+xml'
}

def image_to_data_url(image: Image.Image, ext: str) -> str:
    ext = ext.lower()
    if ext not in EXT_TO_MIMETYPE:
        ext = '.jpg'  # Default to .jpg if extension is not recognized
    mimetype = EXT_TO_MIMETYPE[ext]
    buffered = BytesIO()
    image_format = 'JPEG' if ext in ['.jpg', '.jpeg'] else ext.replace('.', '').upper()
    image.save(buffered, format=image_format)
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    data_url = f"data:{mimetype};base64,{encoded_string}"
    return data_url

# Function to run inference on a single image
@weave.op()
def run_inference(image_url: str) -> dict:
    try:
        prompt = "<|user|>\n<|image_1|>What is shown in this image?<|end|><|assistant|>\n"        
        # Load image
        image = Image.open(requests.get(image_url, stream=True).raw)
        ext = Path(image_url).suffix

        # Convert image to data URL
        data_url = image_to_data_url(image, ext)

        inputs = processor(prompt, [image], return_tensors="pt").to(device)
        generation_args = { 
            "max_new_tokens": 500, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

        # Remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        return {
            "predicted_text": response_text,
            "image_data_url": data_url
        }

    except Exception as e:
        print(f"Error during inference: {e}")
        raise

# Initialize Weave project
weave.init('burberry-product-price-prediction')

# Example usage
image_url = "https://assets.burberry.com/is/image/Burberryltd/1C09D316-7A71-472C-8877-91CEFBDB268A?$BBY_V3_SL_1$&wid=1501&hei=1500"
try:
    result = run_inference(image_url)
    print("Predicted Text:", result['predicted_text'])
    print("Image Data URL:", result['image_data_url'])
except Exception as e:
    print(f"Error running inference: {e}")