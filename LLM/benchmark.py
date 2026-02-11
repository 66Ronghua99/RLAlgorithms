import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from reward_function import extract_xml_answer, extract_hash_answer, correctness_reward_func, strict_format_reward_func, soft_format_reward_func, xmlcount_reward_func, get_answer
import torch
import re
import json

def benchmark(model_name: str, lora_path: str = None, limit: int = None, output_file: str = "benchmark_results.jsonl"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    if lora_path:
        print(f"Loading LoRA from: {lora_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload() # Optional, for speed
    
    # Load GSM8K
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(limit))

    # Prepare prompts
    SYSTEM_PROMPT = """Answer the following math problem. 
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
    
    prompts = []
    ground_truths = []
    data_items = []
    
    for example in ds:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example['question']}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
        ground_truths.append(extract_hash_answer(example['answer']))
        data_items.append(example)

    print(f"Generating for {len(prompts)} examples...")
    
    correct = 0
    total = len(ground_truths)
    results = []
    
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,
                temperature=0.0, # Greedy
                do_sample=False
            )
        generated_ids = output_ids[0][len(inputs.input_ids[0]):]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        extracted = get_answer(generated_text)
        gt = ground_truths[i]


        is_correct = False
        try:
            # Normalize and compare
            if extracted == gt:
                is_correct = True
            elif gt and extracted:
                 # Remove commas and try float conversion
                 ext_clean = extracted.replace(',', '')
                 gt_clean = gt.replace(',', '')
                 if float(ext_clean) == float(gt_clean):
                     is_correct = True
        except:
            pass
            
        if is_correct:
            correct += 1

        # Calculate rewards
        # The reward functions expect a list of completions and potentially other args.
        # Structure for reward functions: completions=[[{"content": generated_text}]] (simulating message list structure if needed, or just text depending on impl)
        # Looking at reward_function.py, they expect `completions` where each item can be a list (messages) or string.
        # correctness_reward_func expects `prompts`, `completions`, `answer`
        
        completion_struct = [{"content": generated_text}] # Passing as list of dicts to match probable chat format expectation or just raw text.
        # Let's check reward_function.py usage.
        # It handles `c[0]['content']` if list. So we pass `[[{"content": generated_text}]]` to be safe if it iterates.
        # But looking at valid inputs: `completions` is iterated. `c` is an item.
        # If we pass `[generated_text]`, `c` is string.
        # `responses = [c[0]['content'] if isinstance(c, list) else c for c in completions]`
        # So passing `[generated_text]` is fine.
        
        # correctness_reward_func needs `prompts` (for logging question), `completions`, `answer`
        # and it logs things. It expects lists.
        # We process one by one here, so lists of length 1.
        
        r_correctness_list = correctness_reward_func(prompts=[[{"content": data_items[i]['question']}]], completions=[generated_text], answer=[gt])
        r_strict = strict_format_reward_func(completions=[generated_text])[0]
        r_soft = soft_format_reward_func(completions=[generated_text])[0]
        r_xml = xmlcount_reward_func(completions=[generated_text])[0]
        
        # correctness_reward_func returns a list of rewards.
        r_correctness = r_correctness_list[0]

        result_item = {
            "question": data_items[i]['question'],
            "ground_truth": gt,
            "original_answer": data_items[i]['answer'], # The full answer string from dataset
            "generated_text": generated_text,
            "extracted_answer": extracted,
            "is_correct": is_correct,
            "rewards": {
                "correctness": r_correctness,
                "strict_format": r_strict,
                "soft_format": r_soft,
                "xml_count": r_xml
            }
        }
        results.append(result_item)
        
        if i < 5: # Debug print first 5
            print(f"Q: {data_items[i]['question']}")
            print(f"Gen: {generated_text}")
            print(f"Extracted: {extracted}, GT: {gt}, Correct: {is_correct}")
            print(f"Rewards: {result_item['rewards']}\n")

    accuracy = correct / total
    print(f"Final Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_file", type=str, default="benchmark_results.jsonl")
    args = parser.parse_args()
    benchmark(args.model_name, args.lora_path, args.limit, args.output_file)
