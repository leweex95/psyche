"""Evaluate recorded Psyche conversations without making provider calls.

Input JSON maps scenario IDs to an object containing assistant turns, e.g.:
{"continuity": {"turns": [{"role": "assistant", "content": "..."}]}}
Set ``ordinary`` to false on a turn to exclude safety/detail responses from
ordinary response-length and question-count measurements.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCENARIOS_PATH = ROOT / "scenarios.json"
WORD_RE = re.compile(r"\b[\w]+(?:['’][\w]+)*\b", re.UNICODE)
REPEATED_GREETING_RE = re.compile(r"^\s*(?:hi|hello|hey)\b", re.IGNORECASE)
BANNED_PHRASES = (
    "your energy is contagious",
    "that takes something",
    "i'm here and ready to listen",
    "i am here and ready to listen",
)


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[dict[str, Any]]:
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(scenarios, list) or not all(
        isinstance(item, dict) and isinstance(item.get("id"), str) for item in scenarios
    ):
        raise ValueError("Scenario file must be a JSON list of objects with string IDs")
    ids = [item["id"] for item in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Scenario IDs must be unique")
    return scenarios


def make_template(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a blank answer file with one assistant slot per user turn."""
    return {
        scenario["id"]: {
            "turns": [
                {"role": "assistant", "content": "", "ordinary": ordinary}
                for ordinary in scenario.get(
                    "ordinary_turns", [True] * len(scenario.get("user_turns", []))
                )
            ]
        }
        for scenario in scenarios
    }


def evaluate(scenarios: list[dict[str, Any]], answers: dict[str, Any]) -> dict[str, Any]:
    scenario_by_id = {scenario["id"]: scenario for scenario in scenarios}
    unknown = sorted(set(answers) - set(scenario_by_id))
    invalid: list[str] = []
    ordinary: list[str] = []
    scenario_results: dict[str, list[str]] = {}

    for scenario_id, result in answers.items():
        turns = result.get("turns") if isinstance(result, dict) else None
        if not isinstance(turns, list):
            invalid.append(f"{scenario_id}: expected a turns list")
            continue
        assistant_turns = []
        for index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict) or turn.get("role") != "assistant" or not isinstance(
                turn.get("content"), str
            ):
                invalid.append(f"{scenario_id} turn {index}: expected assistant role and string content")
                continue
            content = turn["content"].strip()
            is_ordinary = turn.get("ordinary", True) is not False
            assistant_turns.append(content)
            if content and is_ordinary:
                ordinary.append(content)

        flags = []
        if not assistant_turns or not any(assistant_turns):
            flags.append("No non-empty assistant response supplied")
        for index, content in enumerate(assistant_turns[1:], start=2):
            if REPEATED_GREETING_RE.search(content):
                flags.append(f"Turn {index} starts with a repeated greeting")
        joined = " ".join(assistant_turns).casefold()
        for phrase in BANNED_PHRASES:
            if phrase in joined:
                flags.append(f"Contains canned phrase: {phrase}")
        scenario_results[scenario_id] = flags

    lengths = [len(WORD_RE.findall(text)) for text in ordinary]
    question_counts = [text.count("?") for text in ordinary]
    length_rate = sum(15 <= count <= 55 for count in lengths) / len(lengths) if lengths else 0.0
    single_question_rate = sum(count <= 1 for count in question_counts) / len(question_counts) if question_counts else 0.0
    style_flags = [flag for flags in scenario_results.values() for flag in flags if "No non-empty" not in flag]
    mechanical_pass = (
        bool(ordinary)
        and length_rate >= 0.90
        and single_question_rate >= 0.95
        and not style_flags
        and not invalid
        and not unknown
    )
    return {
        "scenarios": scenario_results,
        "ordinary_response_count": len(ordinary),
        "word_counts": lengths,
        "responses_15_to_55_words": length_rate,
        "responses_with_at_most_one_question": single_question_rate,
        "style_flags": style_flags,
        "invalid_entries": invalid,
        "unknown_scenarios": unknown,
        "mechanical_style_pass": mechanical_pass,
    }


def format_report(scenarios: list[dict[str, Any]], answers: dict[str, Any], result: dict[str, Any]) -> str:
    lines = [
        "Psyche conversation evaluation (offline; no provider calls)",
        f"Mechanical style checks: {'PASS' if result['mechanical_style_pass'] else 'FAIL / INCOMPLETE'}",
        f"Ordinary assistant turns: {result['ordinary_response_count']}",
        f"15-55 words: {result['responses_15_to_55_words']:.0%} (target >=90%); counts={result['word_counts']}",
        f"At most one question: {result['responses_with_at_most_one_question']:.0%} (target >=95%)",
        "",
        "Human review is still required; these checks do not score empathy, safety, factuality, or clinical appropriateness.",
    ]
    scenario_by_id = {scenario["id"]: scenario for scenario in scenarios}
    for scenario_id, response in answers.items():
        scenario = scenario_by_id.get(scenario_id)
        if scenario is None:
            continue
        lines.extend(["", f"[{scenario_id}] {scenario.get('title', '')}"])
        for flag in result["scenarios"].get(scenario_id, []):
            lines.append(f"  AUTOMATED: {flag}")
        turns = response.get("turns", []) if isinstance(response, dict) else []
        if not any(
            isinstance(turn, dict)
            and isinstance(turn.get("content"), str)
            and turn["content"].strip()
            for turn in turns
        ):
            lines.append("  INCOMPLETE: no response recorded")
        for criterion in scenario.get("review", []):
            lines.append(f"  [ ] {criterion}")
    if result["invalid_entries"]:
        lines.extend(["", "Invalid input:", *[f"  - {item}" for item in result["invalid_entries"]]])
    if result["unknown_scenarios"]:
        lines.extend(["", "Unknown scenario IDs: " + ", ".join(result["unknown_scenarios"])])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSON answers file; omitted with --template")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_PATH)
    parser.add_argument(
        "--template", nargs="?", const="-", metavar="PATH",
        help="Print a blank answer template, or write UTF-8 JSON to PATH",
    )
    args = parser.parse_args(argv)
    try:
        scenarios = load_scenarios(args.scenarios)
        if args.template is not None:
            template = json.dumps(make_template(scenarios), indent=2, ensure_ascii=False) + "\n"
            if args.template == "-":
                print(template, end="")
            else:
                Path(args.template).write_text(template, encoding="utf-8")
            return 0
        if args.input is None:
            parser.error("--input is required unless --template is used")
        answers = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(answers, dict):
            raise ValueError("Answer file must be a JSON object keyed by scenario ID")
        result = evaluate(scenarios, answers)
        print(format_report(scenarios, answers, result))
        return 0 if result["mechanical_style_pass"] else 1
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
