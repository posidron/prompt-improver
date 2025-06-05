import argparse
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# Set transformers verbosity to error level to suppress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Suppress warnings
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")

def load_prompter():
    """Load the Promptist model and tokenizer"""
    prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return prompter_model, tokenizer

def generate_prompt(plain_text, model, tokenizer, max_new_tokens=75, temperature=0.0, early_stopping=True):
    """Generate optimized prompt from input text"""
    # Tokenize with attention mask to avoid warnings
    inputs = tokenizer(plain_text.strip() + " Rephrase:", return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    eos_id = tokenizer.eos_token_id
    
    # Adjust generation parameters based on temperature
    do_sample = temperature > 0.0
    
    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        max_new_tokens=max_new_tokens, 
        num_beams=8 if not do_sample else 1,
        num_return_sequences=8 if not do_sample else 1,
        eos_token_id=eos_id, 
        pad_token_id=eos_id,
        early_stopping=early_stopping
    )
    
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    result = output_texts[0].replace(plain_text + " Rephrase:", "").strip()
    return result

def main():
    parser = argparse.ArgumentParser(description="Optimize prompts using Promptist")
    parser.add_argument("prompt", help="The prompt to optimize")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output the result")
    parser.add_argument("--max-tokens", "-t", type=int, default=75, 
                       help="Maximum number of new tokens to generate (default: 75)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature for more variation (default: 0.0 for deterministic)")
    parser.add_argument("--no-early-stopping", action="store_true",
                       help="Disable early stopping to potentially generate longer prompts")
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("Loading Promptist model...")
    
    model, tokenizer = load_prompter()
    
    if not args.quiet:
        print("Generating optimized prompt...")
    
    optimized_prompt = generate_prompt(
        args.prompt, model, tokenizer, 
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        early_stopping=not args.no_early_stopping
    )
    
    if args.quiet:
        print(optimized_prompt)
    else:
        print(f"\nOriginal: {args.prompt}")
        print(f"Optimized: {optimized_prompt}")

if __name__ == "__main__":
    main()

