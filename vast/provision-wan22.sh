#!/bin/bash
# Vast.ai provisioning script for the ComfyUI template: downloads Wan 2.2 14B
# (text-to-video + image-to-video), the LightX2V 4-step LoRAs, text encoder and VAE.
# Use: set the template's PROVISIONING_SCRIPT variable to the raw URL of this file.
# Needs about 70 GB of disk (rent with 150 GB). Already-downloaded files are skipped.
set -uo pipefail

COMFY="${WORKSPACE:-/workspace}/ComfyUI"
MODELS="$COMFY/models"
BASE=https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files

FILES=(
  text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
  vae/wan_2.1_vae.safetensors
  diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
  diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors
  loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors
  loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors
  diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
  diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
  loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
  loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors
)

echo "Wan 2.2: downloading models into $MODELS"
failed=0
for f in "${FILES[@]}"; do
  if [ -s "$MODELS/$f" ]; then echo "  ok      $f"; continue; fi
  mkdir -p "$MODELS/$(dirname "$f")"
  echo "  get     $f"
  # Download to .part first, so ComfyUI never sees a half-finished file.
  if wget -q -c -O "$MODELS/$f.part" "$BASE/$f"; then
    mv "$MODELS/$f.part" "$MODELS/$f"
  else
    echo "  FAILED  $f"; failed=1
  fi
done

if [ "$failed" = 0 ]; then
  echo "Wan 2.2: all models ready. In ComfyUI: Templates > Video > Wan 2.2 14B, set enable_turbo_mode = true."
else
  echo "Wan 2.2: some downloads failed; restart the instance or run this script again."
fi
