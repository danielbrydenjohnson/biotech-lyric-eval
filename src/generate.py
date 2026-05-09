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

HELPFUL NOTES:
{notes}

LENGTH:
Aim for roughly 24 to 40 lines.
""".strip()


def run_generation(
    prompts: List[Dict[str, str]],
    model_calls: Dict[str, Callable[[str], str]],
    output_path: Path,
    max_tokens: int = 1200,
    overwrite: bool = False,
) -> List[Dict[str, str]]:
    """
    Generate lyric outputs for every prompt and every model.

    Results are saved to output_path as JSON.

    If output_path already exists and overwrite=False, the existing file is loaded
    instead of making more API calls.
    """
    if output_path.exists() and not overwrite:
        print(f"Generation file already exists. Loading existing results from: {output_path}")
        with output_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    results = []
    total_calls = len(prompts) * len(model_calls)

    for prompt_record in tqdm(prompts, desc="Prompts"):
        generation_prompt = build_generation_prompt(prompt_record)

        for model_name, call_model in tqdm(
            model_calls.items(),
            total=len(model_calls),
            desc=f"Models for {prompt_record['id']}",
            leave=False,
        ):
            print(f"Generating {prompt_record['id']} with {model_name}")

            output_text = call_model(
                generation_prompt,
                max_tokens=max_tokens,
            )

            results.append(
                {
                    "prompt_id": prompt_record["id"],
                    "category": prompt_record["category"],
                    "category_short": prompt_record["category_short"],
                    "prompt": prompt_record["prompt"],
                    "model": model_name,
                    "generation_prompt": generation_prompt,
                    "output_text": output_text,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} generations to: {output_path}")

    if len(results) != total_calls:
        print(f"Warning: expected {total_calls} generations but got {len(results)}")

    return results