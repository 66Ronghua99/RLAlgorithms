
from datasets import load_dataset

def format_gsm8k(example):
    # GSM8K answer format: "Reasoning... #### Answer"
    answer_raw = example['answer']
    if "####" not in answer_raw:
        return None
    
    reasoning, answer = answer_raw.split("####")
    formatted_answer = f"<reasoning>\n{reasoning.strip()}\n</reasoning>\n<answer>\n{answer.strip()}\n</answer>"
    
    return {
        "question": example['question'],
        "answer": formatted_answer
    }

def main():
    ds = load_dataset("openai/gsm8k", "main")
    # Apply formatting
    ds = ds.map(format_gsm8k)
    # Filter out None if any (though GSM8K is clean usually)
    
    # We can just return the dataset or save it to disk
    print("Example entry:")
    print(ds['train'][0])
    
    return ds

if __name__ == "__main__":
    main()
