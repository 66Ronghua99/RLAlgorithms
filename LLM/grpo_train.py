import torch
from datasets import load_dataset, Dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from reward_function import xmlcount_reward_func, soft_format_reward_func, strict_format_reward_func, int_reward_func, correctness_reward_func

def train_grpo(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "outputs/Qwen-GRPO",
    sft_model_path: str = None,
    learning_rate: float = 1e-6, # Lower LR for RL usually
    beta: float = 0.04, # KL penalty coefficient
    max_steps: int = 1000,
    run_name: str = "grpo_gsm8k",
):
    if sft_model_path:
        print(f"Using SFT model from: {sft_model_path}")
        model_name = sft_model_path # Override model_name if SFT path provided

    print(f"Loading model: {model_name}")
    
    # Qwen 2.5 0.5B is small, we can load in float16 or bfloat16
    # Use 'sdpa' (PyTorch 2.0+) if available, else 'eager'
    attn_impl = "sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation=attn_impl
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.05,
        bias="none",
    )
    
    # Load Dataset
    print("Loading GSM8K dataset...")
    # To ensure no data pollution, we take the official 'train' split and split it internally
    # into train (e.g. 95%) and validation (5%). We leave the official 'test' split completely unseen.
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Format dataset for GRPO
    # GRPOTrainer expects specific format usually or we can process it in `training_step`.
    # But usually it expects a list of prompts.
    
    SYSTEM_PROMPT = """Answer the following math problem. 
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
    
    def format_params(examples):
        prompts = []
        answers = []
        for q, a in zip(examples['question'], examples['answer']):
            # Chat format
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            answers.append(a)
        return {"prompt": prompts, "answer": answers}
        
    train_dataset = train_dataset.map(format_params, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_params, batched=True, remove_columns=eval_dataset.column_names)
    
    # Filter dataset for debugging if needed
    # dataset = dataset.select(range(100))

    # GRPO Config
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=learning_rate,
        per_device_train_batch_size=4, # Adjust based on VRAM. 0.5B is tiny. 
        # With GRPO we generate G completions. So effective BS is batch_size * num_generations.
        num_generations=4, # number of generations per prompt
        max_prompt_length=256,
        max_completion_length=512, # CoT can be long
        num_train_epochs=1,
        max_steps=max_steps,        
        save_steps=100,
        logging_steps=10,
        gradient_accumulation_steps=1,
        beta=beta, # KL penalty
    )
    
    # Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer, # pass tokenizer as processing_class
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--sft_model_path", type=str, default=None, help="Path to SFT model checkpoint")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    
    if args.dry_run:
        train_grpo(max_steps=10, run_name="grpo_gsm8k_dryrun", sft_model_path=args.sft_model_path)
    else:
        train_grpo(max_steps=args.max_steps, learning_rate=args.learning_rate, sft_model_path=args.sft_model_path)
