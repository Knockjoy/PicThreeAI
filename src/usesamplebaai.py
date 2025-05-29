# %%
from PIL import Image
from transformers import AutoModel, AutoConfig
from transformers import CLIPImageProcessor, pipeline, CLIPTokenizer
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

image_path = "./Apple.png"
model_name_or_path = "BAAI/EVA-CLIP-8B" # or /path/to/local/EVA-CLIP-8B
image_size = 224

processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

# use image processor with conig
# processor = CLIPImageProcessor(size={"shortest_edge":image_size}, do_center_crop=True, crop_size=image_size)

## you can also directly use the image processor by torchvision
## squash
# processor = T.Compose(
#     [
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
#     ]
# )
## shortest
## processor = T.Compose(
#     [
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
#         T.CenterCrop(image_size),
#         T.ToTensor(),
#         T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
#     ]
# )

model = AutoModel.from_pretrained(
    model_name_or_path, 
    torch_dtype=torch.float16,
    trust_remote_code=True).to('cpu').eval()

image = Image.open(image_path)
captions = ["a diagram", "a dog", "a cat"]
tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)
input_ids = tokenizer(captions,  return_tensors="pt", padding=True).input_ids.to('cpu')
input_pixels = processor(images=image, return_tensors="pt", padding=True).pixel_values.to('cpu')

with torch.no_grad():#torch.cuda.amp.autocast()
    image_features = model.encode_image(input_pixels)
    text_features = model.encode_text(input_ids)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

label_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(f"Label probs: {label_probs}")

# %%
