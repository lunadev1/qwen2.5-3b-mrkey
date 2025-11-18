"""
RunPod Serverless Handler für Keyword Generation
"""

import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Global model loading (wird nur einmal beim Cold Start geladen)
print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, "lunadev1/mrkey")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
print("✅ Model loaded!")

def handler(event):
    """
    RunPod Handler Function
    Input: {"input": {"product_title": "...", "temperature": 0.7}}
    """
    try:
        # Extract input
        product_title = event["input"]["product_title"]
        temperature = event["input"].get("temperature", 0.7)
        
        # Format prompt
        messages = [{"role": "user", "content": product_title}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract keywords from output
        try:
            assistant_response = result.split("assistant")[-1].strip()
            import json
            keywords = json.loads(assistant_response)
        except:
            keywords = []
        
        return {
            "keywords": keywords,
            "raw_output": result
        }
    
    except Exception as e:
        return {"error": str(e)}

# Start RunPod serverless
runpod.serverless.start({"handler": handler})
