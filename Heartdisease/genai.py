
from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

def generate_explanation(prediction, patient_data):

    prediction = int(prediction)

    if prediction == 0:
        result = "No heart disease was predicted by the KNN model."
    else:
        result = "Heart disease was predicted by the KNN model."

    prompt = (
        "Rewrite this in simple English: "
        + result
        + " KNN makes predictions by comparing patient information "
        "with similar examples in its training data."
    )

    response = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False
    )

    ai_text = response[0]["generated_text"].strip()

    return f"""
    ML Prediction: {prediction}

    Result: {result}

    How KNN works:
    KNN compares the patient's information with similar records
    from the training dataset and uses nearby records to make a prediction.

    AI-generated explanation:
    {ai_text}

    Important:
    This is an educational prediction, not a medical diagnosis.
    Consult a qualified doctor for medical evaluation.
    """
