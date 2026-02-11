
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from build_sft_dataset import format_gsm8k

def train_sft(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "outputs/Qwen-SFT",
    learning_rate: float = 2e-5,
    max_steps: int = 1000,
):
    print(f"Loading model: {model_name}")
    attn_impl = "sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and format dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    
    train_dataset = dataset["train"].map(format_gsm8k)
    eval_dataset = dataset["test"].map(format_gsm8k) # This is our internal validation set
    
    SYSTEM_PROMPT = """Answer the following math problem. 
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

    def formatting_prompts_func(example):
        output_texts = []
        # Check if batched
        questions = example['question']
        answers = example['answer']
        
        if isinstance(questions, str):
            questions = [questions]
            answers = [answers]
            
        for q, a in zip(questions, answers):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a} 
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
            
        if isinstance(example['question'], str):
            return output_texts[0]
        return output_texts



    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        eval_strategy="steps",
        eval_steps=50,
        logging_first_step=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        args=training_args,
    )

    print("Starting SFT training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    
    if args.dry_run:
        train_sft(max_steps=10)
    else:
        train_sft(max_steps=args.max_steps)
