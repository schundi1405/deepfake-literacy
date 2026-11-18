
import streamlit as st
from pathlib import Path
import random
from pipeline import evaluate_participant

st.set_page_config(
    page_title="Deepfake Literacy Tool",
    page_icon="🔍"
)

st.title("🔍 Can You Spot the Deepfake?")
st.write("Look at the image and decide whether it is real or AI-manipulated.")

image_dir = Path("data/sample_images")
images = list(image_dir.glob("*.jpg"))

if not images:
    st.error("No images found! Run load_dataset.py first.")
    st.stop()

if "remaining_images" not in st.session_state:
    st.session_state.remaining_images = random.sample(images, len(images))

if "current_image" not in st.session_state:
    st.session_state.current_image = st.session_state.remaining_images.pop()

if "submitted" not in st.session_state:
    st.session_state.submitted = False
image_path = st.session_state.current_image

st.image(str(image_path), width=400)

st.subheader("Your Analysis")

prediction = st.radio(
    "Is this image real or fake?",
    ["Real", "Fake"],
    index=None,
    key="prediction"
)

explanation = st.text_area(
    "Explain your reasoning:",
    placeholder="What clues helped you make your decision?",
    key="explanation"
)

confidence = st.slider(
    "How confident are you?",
    min_value=0,
    max_value=100,
    value=50,
    key="confidence"
)


if st.button("Submit Answer", disabled=st.session_state.submitted):
    if prediction is None or not explanation.strip():
        st.warning("Please select an answer and explain your reasoning.")
    else:
        st.session_state.submitted = True

        actual_label = "real" if image_path.name.startswith("real_") else "fake"

        scenario_context = (
            f"The dataset labels this image as {actual_label}. "
            "No specific visual manipulation clues have been independently verified."
        )

        with st.spinner("Generating personalized feedback..."):
            result = evaluate_participant(
                prediction=prediction.lower(),
                explanation=explanation,
                confidence=confidence,
                actual_label=actual_label,
                scenario_context=scenario_context
            )
            
            st.session_state.result = result

        st.success("Your analysis is complete!")

        st.subheader("Your Personalized Feedback")
        st.write(result["feedback"])

        st.subheader("Your Rubric Score")
        st.metric("Total Score", f'{result["total_score"]}/15')

        
        st.subheader("Strengths")
        strengths = result["strengths"]

        if isinstance(strengths, str):
            st.write("✅", strengths)
        else:
            for strength in strengths:
                st.write("✅", strength)

        st.subheader("Areas for Improvement")
        improvements = result["improvements"]

        if isinstance(improvements, str):
            st.write("💡", improvements)
        else:
            for improvement in improvements:
                st.write("💡", improvement)


        st.subheader("Reasoning Strategies Detected")
        st.json(result["detected_strategies"])


if st.session_state.submitted:

    if st.session_state.remaining_images:
        if st.button("➡️ Next Image"):
            st.session_state.current_image = (
                st.session_state.remaining_images.pop()
            )
            st.session_state.submitted = False
            st.session_state.pop("result", None)

            for key in ["prediction", "explanation", "confidence"]:
                st.session_state.pop(key, None)




            
            st.rerun()

    else:
        st.success("🎉 You've completed all 6 practice images!")

        if st.button("Practice Again"):
            st.session_state.remaining_images = random.sample(
                images, len(images)
            )
            st.session_state.current_image = (
                st.session_state.remaining_images.pop()
            )
            st.session_state.submitted = False
            st.session_state.pop("result", None)

            for key in ["prediction", "explanation", "confidence"]:
                            st.session_state.pop(key, None)
            st.rerun()
