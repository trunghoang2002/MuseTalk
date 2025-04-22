from diffusers import AutoencoderKL

model_path = "models/sd-vae"
vae = AutoencoderKL.from_pretrained(model_path)
print(vae)