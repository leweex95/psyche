import json

from evals.runner import evaluate, load_scenarios, main, make_template


def test_curated_scenarios_have_unique_ids_and_review_criteria():
    scenarios = load_scenarios()
    assert len(scenarios) >= 6
    assert all(scenario["review"] for scenario in scenarios)
    assert all(len(make_template([scenario])[scenario["id"]]["turns"]) == len(scenario["user_turns"])
               for scenario in scenarios)
    assert make_template(scenarios)["specific_distress"]["turns"][0]["ordinary"] is False


def test_evaluator_measures_only_recorded_ordinary_turns_and_flags_style_issues():
    scenarios = [
        {"id": "sample", "title": "Sample", "review": ["Human judgment"]},
    ]
    in_range = "These are fifteen counted words in this ordinary response that should pass the length check."
    answers = {
        "sample": {
            "turns": [
                {"role": "assistant", "content": in_range},
                {"role": "assistant", "content": "Hi again. Your energy is contagious? What else?"},
                {"role": "assistant", "content": "A longer safety response is intentionally excluded from ordinary metrics.", "ordinary": False},
            ]
        }
    }

    result = evaluate(scenarios, answers)

    assert result["ordinary_response_count"] == 2
    assert result["responses_15_to_55_words"] == 0.5
    assert result["responses_with_at_most_one_question"] == 0.5
    assert not result["mechanical_style_pass"]
    assert any("repeated greeting" in flag.lower() for flag in result["scenarios"]["sample"])
    assert any("canned phrase" in flag.lower() for flag in result["scenarios"]["sample"])


def test_evaluator_does_not_equate_mechanical_pass_with_qualitative_review():
    scenarios = [{"id": "sample", "title": "Sample", "review": ["Check response quality"]}]
    answer = "I can see why that specific situation is on your mind. What feels most important to focus on today?"
    result = evaluate(scenarios, {"sample": {"turns": [{"role": "assistant", "content": answer}]}})

    assert result["mechanical_style_pass"]
    assert result["scenarios"]["sample"] == []


def test_template_is_machine_readable_json():
    scenarios = [{"id": "two", "user_turns": ["first", "second"]}]
    template = make_template(scenarios)
    assert len(template["two"]["turns"]) == 2
    assert json.loads(json.dumps(template))["two"]["turns"][0]["role"] == "assistant"


def test_template_command_writes_explicit_utf8_file(tmp_path):
    output = tmp_path / "answers.json"
    assert main(["--template", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["specific_distress"]
