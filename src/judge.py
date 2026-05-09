import json
import random
import re
from pathlib import Path
from typing import Callable, Dict, List

from pydantic import ValidationError
from tqdm.notebook import tqdm

from src.schemas import Judgement


VERSION_LABELS = ["A", "B", "C", "D", "E", "F"]


def extract_json(text: str) -> Dict:
    """
    Extract a JSON object from a model response.

    Some models return clean JSON.
    Some wrap JSON in ```json fences.
    Some add a sentence before or after.

    This function tries to recover the JSON object.
    """
    text = text.strip()

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in judge response.")

    return json.loads(text[start:end + 1])


def validate_ranking(judgement: Judgement, expected_labels: List[str]) -> None:
    """
    Ensure the ranking and score labels contain exactly the expected labels once each.
    """
    ranking = judgement.ranking

    if sorted(ranking) != sorted(expected_labels):
        raise ValueError(
            f"Invalid ranking. Expected labels {expected_labels}, got {ranking}"
        )

    if len(ranking) != len(set(ranking)):
        raise ValueError(f"Invalid ranking. Duplicate labels found: {ranking}")

    score_labels = list(judgement.scores.keys())

    if sorted(score_labels) != sorted(expected_labels):
        raise ValueError(
            f"Invalid score labels. Expected labels {expected_labels}, got {score_labels}"
        )


def build_scores_template(labels: List[str]) -> str:
    """
    Build the example JSON scores object for the expected labels.
    """
    template = {
        label: {
            "genre_fidelity": 1,
            "scientific_accuracy": 1,
            "lyrical_craft": 1,
            "cleverness": 1,
            "commitment": 1,
        }
        for label in labels
    }

    return json.dumps(template, indent=4)


def build_anonymous_judge_prompt(
    prompt_text: str,
    labelled_outputs: Dict[str, str],
) -> str:
    """
    Build the anonymous-mode judging prompt.

    labelled_outputs maps anonymous labels A/B/C/D to generated lyrics.
    """
    versions_text = "\n\n".join(
        f"VERSION {label}:\n{output}"
        for label, output in labelled_outputs.items()
    )

    expected_labels = list(labelled_outputs.keys())
    scores_template = build_scores_template(expected_labels)

    return f"""
You are evaluating song lyrics written for the following prompt:

PROMPT:
{prompt_text}

Below are versions written by different language models. The model identities are hidden.

Judge each version on five criteria, scoring each from 1 to 5:

1. Genre fidelity: Does it actually sound like the requested genre or format?
2. Scientific accuracy: Are the biotechnology or life science references correct and meaningfully used?
3. Lyrical craft: Rhyme, rhythm, structure, and whether it scans as lyrics.
4. Cleverness: Wordplay, double meanings, surprising lines, or inventive framing.
5. Commitment: Did the writer commit to the bit, or did it play safe?

Then rank the versions from best to worst.

{versions_text}

Return ONLY valid JSON matching this exact structure:

{{
  "scores": {scores_template},
  "ranking": {json.dumps(expected_labels)},
  "reasoning": "Brief explanation of the judgement."
}}

Important:
- The ranking must be ordered from best to worst.
- Use each version label exactly once in the ranking.
- Do not include markdown.
- Do not include commentary outside the JSON.
""".strip()


def build_labelled_judge_prompt(
    prompt_text: str,
    model_outputs: Dict[str, str],
) -> str:
    """
    Build the labelled-mode judging prompt.

    model_outputs maps model names to generated lyrics.
    """
    versions_text = "\n\n".join(
        f"VERSION {model_name}:\n{output}"
        for model_name, output in model_outputs.items()
    )

    labels_list = list(model_outputs.keys())
    scores_template = build_scores_template(labels_list)

    return f"""
You are evaluating song lyrics written for the following prompt:

PROMPT:
{prompt_text}

Below are versions written by different language models. The model identities are shown.

Judge each version on five criteria, scoring each from 1 to 5:

1. Genre fidelity: Does it actually sound like the requested genre or format?
2. Scientific accuracy: Are the biotechnology or life science references correct and meaningfully used?
3. Lyrical craft: Rhyme, rhythm, structure, and whether it scans as lyrics.
4. Cleverness: Wordplay, double meanings, surprising lines, or inventive framing.
5. Commitment: Did the writer commit to the bit, or did it play safe?

Then rank the versions from best to worst.

{versions_text}

Return ONLY valid JSON matching this exact structure:

{{
  "scores": {scores_template},
  "ranking": {json.dumps(labels_list)},
  "reasoning": "Brief explanation of the judgement."
}}

Important:
- The ranking must be ordered from best to worst.
- Use each model name exactly once in the ranking.
- Do not include markdown.
- Do not include commentary outside the JSON.
""".strip()


def judge_once(
    judge_name: str,
    judge_call: Callable[[str], str],
    judge_prompt: str,
    expected_labels: List[str],
    max_tokens: int = 1500,
    retries: int = 2,
) -> Judgement:
    """
    Call one judge model and validate its structured response.
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            raw_response = judge_call(judge_prompt, max_tokens=max_tokens)
            parsed = extract_json(raw_response)
            judgement = Judgement.model_validate(parsed)
            validate_ranking(judgement, expected_labels)
            return judgement

        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            last_error = error
            print(f"{judge_name} judgement parse failed on attempt {attempt}: {error}")

    raise ValueError(f"{judge_name} failed to return valid judgement: {last_error}")


def save_json(records: List[Dict], output_path: Path) -> None:
    """
    Save judgement records to JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def load_existing_results(output_path: Path) -> List[Dict]:
    """
    Load existing judgement results if the file exists.
    """
    if not output_path.exists():
        return []

    with output_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt_groups(generations: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group generation records by prompt_id.
    """
    groups = {}

    for record in generations:
        groups.setdefault(record["prompt_id"], []).append(record)

    return groups


def run_judging(
    generations: List[Dict],
    judge_calls: Dict[str, Callable[[str], str]],
    output_path: Path,
    mode: str,
    overwrite: bool = False,
    random_seed: int = 42,
) -> List[Dict]:
    """
    Run model judging over generated lyrics.

    mode must be either:
    - "anonymous"
    - "labelled"

    Results are saved progressively after every judge call.
    """
    if mode not in {"anonymous", "labelled"}:
        raise ValueError("mode must be 'anonymous' or 'labelled'")

    rng = random.Random(random_seed)

    existing_results = [] if overwrite else load_existing_results(output_path)
    completed_keys = {
        (record["prompt_id"], record["judge_model"], record["mode"])
        for record in existing_results
    }

    results = existing_results.copy()
    prompt_groups = build_prompt_groups(generations)

    total_expected = len(prompt_groups) * len(judge_calls)

    if existing_results and not overwrite:
        print(f"Loaded {len(existing_results)} existing judgements from: {output_path}")
        print("Existing prompt/judge/mode records will be skipped.")

    for prompt_id, prompt_records in tqdm(prompt_groups.items(), desc=f"Judging {mode}"):
        prompt_text = prompt_records[0]["prompt"]

        successful_records = [
            record for record in prompt_records
            if record.get("success", True) and record.get("output_text", "").strip()
        ]

        if len(successful_records) < 2:
            print(f"Skipping {prompt_id}: fewer than 2 successful outputs")
            continue

        if mode == "anonymous":
            shuffled_records = successful_records.copy()
            rng.shuffle(shuffled_records)

            label_to_record = {
                label: record
                for label, record in zip(VERSION_LABELS[:len(shuffled_records)], shuffled_records)
            }

            labelled_outputs = {
                label: record["output_text"]
                for label, record in label_to_record.items()
            }

            expected_labels = list(labelled_outputs.keys())
            judge_prompt = build_anonymous_judge_prompt(prompt_text, labelled_outputs)

            label_mapping = {
                label: record["model"]
                for label, record in label_to_record.items()
            }

        else:
            model_outputs = {
                record["model"]: record["output_text"]
                for record in successful_records
            }

            expected_labels = list(model_outputs.keys())
            judge_prompt = build_labelled_judge_prompt(prompt_text, model_outputs)
            label_mapping = {model_name: model_name for model_name in expected_labels}

        for judge_name, judge_call in tqdm(
            judge_calls.items(),
            total=len(judge_calls),
            desc=f"Judges for {prompt_id}",
            leave=False,
        ):
            current_key = (prompt_id, judge_name, mode)

            if current_key in completed_keys:
                print(f"Skipping {prompt_id} judged by {judge_name} in {mode}: already exists")
                continue

            print(f"Judging {prompt_id} with {judge_name} in {mode} mode")

            try:
                judgement = judge_once(
                    judge_name=judge_name,
                    judge_call=judge_call,
                    judge_prompt=judge_prompt,
                    expected_labels=expected_labels,
                )

                success = True
                error_message = ""
                judgement_dict = judgement.model_dump()

            except Exception as error:
                success = False
                error_message = str(error)
                judgement_dict = {}

                print(f"FAILED judging {prompt_id} with {judge_name}: {error_message}")

            result = {
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "judge_model": judge_name,
                "mode": mode,
                "success": success,
                "error_message": error_message,
                "label_mapping": label_mapping,
                "judgement": judgement_dict,
            }

            results.append(result)
            completed_keys.add(current_key)

            save_json(results, output_path)
            print(f"Saved progress: {len(results)}/{total_expected} judgement records")

    print(f"Judging run complete. Results saved to: {output_path}")

    return results