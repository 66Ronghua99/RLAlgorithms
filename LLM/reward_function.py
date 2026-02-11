
import re

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def extract_boxed_answer(text: str) -> str | None:
    # Look for \boxed{...} which is common in math datasets
    pattern = r"\\boxed\{([^}]+)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    return None

def extract_last_number(text: str) -> str | None:
    # Find all numbers
    # patterns for integers or floats
    # This is a fallback and might be noisy
    pattern = r"-?\d+(?:\.\d+)?"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    return None

def get_answer(text: str) -> str:
    # Priority: XML > Boxed > Last number
    if "<answer>" in text and "</answer>" in text:
        return extract_xml_answer(text)
    
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
        
    last_num = extract_last_number(text)
    if last_num:
        return last_num
        
    return ""

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [c[0]['content'] if isinstance(c, list) else c for c in completions]
    q = prompts[0][-1]['content'] if isinstance(prompts[0], list) else prompts[0]
    extracted_responses = [get_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}", f"\nAnswer:\n{answer[0]}", '-'*20)
    
    # Check correctness
    rewards = []
    for r, a in zip(extracted_responses, answer):
        is_correct = False
        if r == a:
            is_correct = True
        else:
             try:
                 if float(r.replace(',', '')) == float(a.replace(',', '')):
                     is_correct = True
             except:
                 pass
        rewards.append(2.0 if is_correct else 0.0)
    return rewards

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [c[0]['content'] if isinstance(c, list) else c for c in completions]
    extracted_responses = [get_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [c[0]['content'] if isinstance(c, list) else c for c in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c[0]['content'] if isinstance(c, list) else c for c in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n<answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    responses = [c[0]['content'] if isinstance(c, list) else c for c in completions]
    return [count_xml(r) for r in responses]
