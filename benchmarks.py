"""SPECTER — Benchmark runner: prompt loading, Ollama streaming measurement, scoring."""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("specter.benchmarks")

PROMPTS_DIR = Path(__file__).parent / "prompts"

PROMPT_SETS = {
    "intelligence": "intelligence.json",
    "tactical": "tactical.json",
    "coding": "coding.json",
    "reasoning": "reasoning.json",
}


def load_prompt_set(name: str) -> list[dict]:
    """Load a prompt set by name."""
    filename = PROMPT_SETS.get(name)
    if not filename:
        raise ValueError(f"Unknown prompt set: {name}")
    path = PROMPTS_DIR / filename
    with open(path) as f:
        return json.load(f)


def save_prompt_set(name: str, prompts: list[dict]):
    """Save a prompt set to disk."""
    filename = PROMPT_SETS.get(name)
    if not filename:
        raise ValueError(f"Unknown prompt set: {name}")
    path = PROMPTS_DIR / filename
    with open(path, "w") as f:
        json.dump(prompts, f, indent=2)
        f.write("\n")


def list_prompt_sets() -> list[dict]:
    """List available prompt sets with metadata."""
    result = []
    for name, filename in PROMPT_SETS.items():
        path = PROMPTS_DIR / filename
        try:
            with open(path) as f:
                prompts = json.load(f)
            result.append({
                "id": name,
                "name": name.replace("_", " ").title(),
                "count": len(prompts),
                "categories": list({p.get("category", "general") for p in prompts}),
            })
        except Exception as e:
            logger.warning(f"Failed to load prompt set {name}: {e}")
    return result


def add_prompt(set_name: str, prompt: dict) -> dict:
    """Add a prompt to a set. Auto-generates ID if not provided."""
    prompts = load_prompt_set(set_name)
    if "id" not in prompt:
        prefix = set_name[:5]
        max_num = 0
        for p in prompts:
            try:
                num = int(p["id"].split("-")[-1])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                pass
        prompt["id"] = f"{prefix}-{max_num + 1:02d}"
    if not prompt.get("name"):
        prompt["name"] = prompt["id"]
    prompts.append(prompt)
    save_prompt_set(set_name, prompts)
    return prompt


def update_prompt(set_name: str, prompt_id: str, updates: dict) -> dict | None:
    """Update a prompt in a set."""
    prompts = load_prompt_set(set_name)
    for i, p in enumerate(prompts):
        if p["id"] == prompt_id:
            updates.pop("id", None)
            prompts[i].update(updates)
            save_prompt_set(set_name, prompts)
            return prompts[i]
    return None


def delete_prompt(set_name: str, prompt_id: str) -> bool:
    """Delete a prompt from a set."""
    prompts = load_prompt_set(set_name)
    new_prompts = [p for p in prompts if p["id"] != prompt_id]
    if len(new_prompts) == len(prompts):
        return False
    save_prompt_set(set_name, new_prompts)
    return True


def score_response(prompt_meta: dict, response_text: str) -> dict | None:
    """Score a response against expected outputs if available."""
    scores = {}

    # Entity extraction scoring
    expected_entities = prompt_meta.get("expected_entities")
    if expected_entities:
        lines = response_text.strip().split("\n")
        table_rows = sum(1 for l in lines if "|" in l and not l.strip().startswith("|-"))
        scores["entity_count"] = min(table_rows, expected_entities)
        scores["entity_expected"] = expected_entities
        scores["entity_pct"] = round(scores["entity_count"] / expected_entities * 100, 1)

    # Expected answer scoring (exact substring match)
    expected_answer = prompt_meta.get("expected_answer")
    if expected_answer:
        answer_lower = expected_answer.lower()
        response_lower = response_text.lower()
        scores["correct"] = answer_lower in response_lower
        scores["expected_answer"] = expected_answer

    # General metrics
    scores["response_length"] = len(response_text)
    scores["word_count"] = len(response_text.split())

    return scores if scores else None


class BenchmarkResult:
    """Result of a single benchmark prompt run."""

    def __init__(self, model: str, prompt_set: str, prompt_meta: dict):
        self.model = model
        self.prompt_set = prompt_set
        self.prompt_id = prompt_meta["id"]
        self.prompt_name = prompt_meta["name"]
        self.category = prompt_meta.get("category", "general")
        self.ttft_ms: float = 0
        self.tokens_per_sec: float = 0
        self.total_tokens: int = 0
        self.thinking_tokens: int = 0
        self.total_time_ms: float = 0
        self.response_text: str = ""
        self.thinking_text: str = ""
        self.scores: dict | None = None
        self.error: str | None = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_set": self.prompt_set,
            "prompt_id": self.prompt_id,
            "prompt_name": self.prompt_name,
            "category": self.category,
            "ttft_ms": round(self.ttft_ms, 1),
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "total_tokens": self.total_tokens,
            "thinking_tokens": self.thinking_tokens,
            "total_time_ms": round(self.total_time_ms, 1),
            "response_text": self.response_text,
            "thinking_text": self.thinking_text,
            "response_length": len(self.response_text),
            "scores": self.scores,
            "error": self.error,
        }
