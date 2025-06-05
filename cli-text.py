#!/usr/bin/env python3
import argparse
import os
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")


def load_model(model_name, force_cpu=False):
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine the best device
    device = None
    torch_dtype = torch.float32

    if force_cpu:
        device = "cpu"
        print("Using CPU (forced)")
    else:
        # Try to use the best available device
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
            print("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            try:
                # Test MPS availability
                torch.tensor([1.0], device="mps")
                device = "mps"
                torch_dtype = torch.float16
                print("Using MPS (Apple Silicon GPU)")
            except Exception as e:
                print(f"MPS failed ({e}), falling back to CPU")
                device = "cpu"
        else:
            device = "cpu"
            print("Using CPU")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )

    # Move model to device
    model = model.to(device)

    return model, tokenizer, device


def improve_prompt(prompt, model, tokenizer, device, max_tokens=512):
    """Improve the given prompt"""
    full_prompt = f"Improve this prompt to make it more effective and detailed:\n\n{prompt}\n\nImproved prompt:"

    # Tokenize
    inputs = tokenizer(
        full_prompt, return_tensors="pt", truncation=True, max_length=max_tokens
    )

    # Move inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and extract improvement
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Improved prompt:" in response:
        improved = response.split("Improved prompt:")[-1].strip()
    else:
        improved = response[len(full_prompt) :].strip()

    return improved


def main():
    parser = argparse.ArgumentParser(
        description="Simple prompt improver using LLaMA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("prompt", help="The prompt to improve")
    parser.add_argument(
        "--model", "-m", default="meta-llama/Llama-2-7b-chat-hf", help="Model to use"
    )
    parser.add_argument(
        "--max-tokens", "-t", type=int, default=512, help="Max tokens for improvement"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Only output the improved prompt"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()

    # Force CPU and disable GPU if requested
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        torch.set_default_device("cpu")

    try:
        print("Loading model...")
        model, tokenizer, device = load_model(args.model, force_cpu=args.cpu)

        print("Improving prompt...")
        improved = improve_prompt(
            args.prompt, model, tokenizer, device, args.max_tokens
        )
        print(f"\nOriginal: {args.prompt}")
        print(f"Improved: {improved}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have access to the model on HuggingFace")


if __name__ == "__main__":
    main()
