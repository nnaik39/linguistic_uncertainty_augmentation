# Adapted from https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075

from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration, AutoTokenizer
import torch
import numpy as np
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# Just put a question in the collab-lens about compute_transition_scores?

#processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
#model = Blip2ForConditionalGeneration.from_pretrained(
#    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, load_in_8bit=True
#)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

loss_fct = nn.NLLLoss(reduction="none", ignore_index=model.config.pad_token_id)

print("Ready...")

urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://farm5.staticflickr.com/4142/4740402062_383881e305_z.jpg",
]

for url in urls:
    image = Image.open(requests.get(url, stream=True).raw)
    
    prompt = "Question: Is there a cat in this image? Answer: "
    print("Prompt: ", prompt)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    outputs = model.generate(**inputs, output_scores=True, return_dict_in_generate=True, max_new_tokens=100)   

    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=False
    )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        # | token | token string | logits | probability
        print(f"| {tok:5d} | {processor.tokenizer.decode(tok):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2%}")
