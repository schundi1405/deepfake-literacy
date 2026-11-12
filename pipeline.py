
from model import evaluate_response
from nlp_classifier import predict_strategies


def evaluate_participant(
    prediction,
    explanation,
    confidence,
    actual_label,
    scenario_context
):
    # Detect reasoning strategies
    strategies = predict_strategies(explanation)

    # Evaluate using Gemini
    evaluation = evaluate_response(
        prediction=prediction,
        explanation=explanation,
        confidence=confidence,
        actual_label=actual_label,
        scenario_context=scenario_context
    )

    # Combine both results
    return {
        "detected_strategies": strategies,
        "rubric_scores": evaluation["scores"],
        "total_score": evaluation["total_score"],
        "strengths": evaluation["strengths"],
        "improvements": evaluation["improvements"],
        "feedback": evaluation["feedback"]
    }


# Test our complete pipeline
if __name__ == "__main__":

    result = evaluate_participant(
        prediction="fake",

        explanation=(
            "The lips don't match the audio, "
            "but I would check the original video."
        ),

        confidence=80,

        actual_label="fake",

        scenario_context=(
            "Synthetic test scenario. The video is "
            "AI-generated, with a verified lip-sync mismatch."
        )
    )

    import json
    print(json.dumps(result, indent=2))
