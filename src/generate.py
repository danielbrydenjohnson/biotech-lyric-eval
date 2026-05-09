import json
from pathlib import Path
from typing import Callable, Dict, List

from tqdm.notebook import tqdm


def build_generation_prompt(prompt_record: Dict[str, str]) -> str:
    """
    Build the generation prompt sent to each model.

    The input is one prompt record from prompts/prompts.json.
    The output is a complete instruction string for the model.
    """
    task_prompt = prompt_record["prompt"]
    notes = prompt_record.get("notes", "")

    return f"""
You are writing original song lyrics for a creative AI evaluation.

TASK:
{task_prompt}

CONTEXT:
This is part of an eval comparing language models on creative writing at the intersection of biotechnology and music.

WHAT TO PRODUCE:
Write complete song lyrics for the task above.

REQUIREMENTS:
- Commit strongly to the requested genre or format.
- Use accurate biotechnology or life science references.
- Make the lyrics specific, not generic.
- Use rhyme, rhythm, structure, and memorable lines.
- Avoid bland motivational science lyrics.
- Avoid explaining the song. Only output the lyrics.
- Do not include commentary before or after the lyrics.
- If the prompt involves a pathogen, write in a fictional, non-instructional, non-operational way. Do not provide instructions for culturing, engineering, spreading, or evading detection.

HELPFUL NOTES:
{notes}

LENGTH:
Aim for roughly 24 to 40 lines.
""".strip()


def save_json(records: List[Dict[str, str]], output_path: Path) -> None:
    """
    Save records to JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def load_existing_results(output_path: Path) -> List[Dict[str, str]]:
    """
    Load existing generation results if the file exists.
    """
    if not output_path.exists():
        return []

    with output_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def result_key(record: Dict[str, str]) -> tuple:
    """
    Create a unique key for one prompt/model generation.
    """
    return (record["prompt_id"], record["model"])


def run_generation(
    prompts: List[Dict[str, str]],
    model_calls: Dict[str, Callable[[str], str]],
    output_path: Path,
    max_tokens: int = 1200,
    overwrite: bool = False,
) -> List[Dict[str, str]]:
    """
    Generate lyric outputs for every prompt and every model.

    Results are saved progressively to output_path after every call.

    If output_path already exists and overwrite=False, existing prompt/model results
    are skipped. This allows safe resuming after crashes.
    """
    existing_results = [] if overwrite else load_existing_results(output_path)
    completed_keys = {result_key(record) for record in existing_results}

    results = existing_results.copy()
    total_expected = len(prompts) * len(model_calls)

    if existing_results and not overwrite:
        print(f"Loaded {len(existing_results)} existing results from: {output_path}")
        print("Existing prompt/model pairs will be skipped.")

    for prompt_record in tqdm(prompts, desc="Prompts"):
        generation_prompt = build_generation_prompt(prompt_record)

        for model_name, call_model in tqdm(
            model_calls.items(),
            total=len(model_calls),
            desc=f"Models for {prompt_record['id']}",
            leave=False,
        ):
            current_key = (prompt_record["id"], model_name)

            if current_key in completed_keys:
                print(f"Skipping {prompt_record['id']} with {model_name}: already exists")
                continue

            print(f"Generating {prompt_record['id']} with {model_name}")

            try:
                output_text = call_model(
                    generation_prompt,
                    max_tokens=max_tokens,
                )
                success = True
                error_message = ""

            except Exception as error:
                output_text = ""
                success = False
                error_message = str(error)
                print(f"FAILED {prompt_record['id']} with {model_name}: {error_message}")

            result = {
                "prompt_id": prompt_record["id"],
                "category": prompt_record["category"],
                "category_short": prompt_record["category_short"],
                "prompt": prompt_record["prompt"],
                "model": model_name,
                "generation_prompt": generation_prompt,
                "output_text": output_text,
                "success": success,
                "error_message": error_message,
            }

            results.append(result)
            completed_keys.add(current_key)

            save_json(results, output_path)
            print(f"Saved progress: {len(results)}/{total_expected} records")

    if len(results) != total_expected:
        print(f"Warning: expected {total_expected} records but got {len(results)}")

    print(f"Generation run complete. Results saved to: {output_path}")

    return results