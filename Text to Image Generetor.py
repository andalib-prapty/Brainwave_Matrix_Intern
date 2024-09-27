import torch
from diffusers import StableDiffusionPipeline as sdp
from transformers import set_seed
import tkinter as tk
from PIL import ImageTk, Image
import io


class CFG:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_max_length = 12


# Load the Stable Diffusion model with correct precision for CPU/GPU
image_gen_model = sdp.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=CFG.dtype, variant="fp16" if CFG.device == "cuda" else None
)
image_gen_model = image_gen_model.to(CFG.device)


def generate_image(prompt, model):
    # Generate an image from the given text prompt
    with torch.autocast(CFG.device):
        image = model(
            prompt, num_inference_steps=CFG.image_gen_steps,
            generator=CFG.generator,
            guidance_scale=CFG.image_gen_guidance_scale
        ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image


def display_image(image):
    # Convert the image to a format Tkinter can display
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    img = Image.open(io.BytesIO(img_byte_arr))
    img_tk = ImageTk.PhotoImage(img)

    return img_tk


# Tkinter GUI Application
class ImageGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text-to-Image Generator")

        # Prompt input label
        self.label = tk.Label(root, text="Enter a prompt:")
        self.label.pack()

        # Text entry field for the prompt
        self.entry = tk.Entry(root, width=50)
        self.entry.pack()

        # Button to trigger image generation
        self.button = tk.Button(root, text="Generate Image", command=self.generate_image)
        self.button.pack()

        # Label to display the generated image
        self.image_label = tk.Label(root)
        self.image_label.pack()

    def generate_image(self):
        # Get prompt from user input
        prompt = self.entry.get()

        # Generate image using the provided prompt
        generated_image = generate_image(prompt, image_gen_model)

        # Convert and display the image in Tkinter
        img_tk = display_image(generated_image)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep reference to avoid garbage collection


# Main application loop
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGeneratorApp(root)
    root.mainloop()
