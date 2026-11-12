
import json
from model import evaluate_response

# Example participant response
result = evaluate_response(
    prediction="fake",

    explanation=(
        "I think this video is fake because the person's "
        "mouth doesn't match the audio. The lighting "
        "also looks strange. I'm not completely sure, "
        "so I would look for the original video."
    ),

    confidence=80,

    actual_label="fake",

    scenario_context=(
        "This is a synthetic test scenario involving "
        "an AI-generated video. Human reviewers have "
        "confirmed a lip-sync mismatch and "
        "inconsistent lighting."
    )
)

# Display the AI evaluation
print(json.dumps(result, indent=2))
