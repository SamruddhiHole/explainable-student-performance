"""
Natural Language Explanation Module.
Generates teacher/student friendly natural language explanations based STRICTLY
on verified model predictions and SHAP feature contributions.

Architecture:
  1. Primary: Local Qwen2.5-3B-Instruct via Ollama API (http://localhost:11434)
  2. Fallback: Deterministic rule-based Python template verbalizer
CRITICAL RULE: The LLM NEVER predicts or alters scores; it solely verbalizes verified XAI factors.
"""

import json
from typing import Dict, Any, Optional
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b-instruct"


def check_ollama_available(url: str = "http://localhost:11434/api/tags", timeout: float = 1.5) -> bool:
    """Checks if a local Ollama instance is actively running and responding."""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def generate_template_explanation(
    prediction_result: Dict[str, Any],
    shap_explanation: Dict[str, Any]
) -> str:
    """
    Deterministic rule-based Python verbalizer fallback.
    Guarantees 100% reliable, factual explanations without external dependencies or cloud APIs.
    """
    cat = prediction_result.get("predicted_category", "Medium")
    conf = prediction_result.get("confidence", 0.0)
    score = prediction_result.get("predicted_score", 0.0)
    risk_desc = prediction_result.get("risk_info", {}).get("level", "Moderate")

    pos_factors = shap_explanation.get("positive_factors", [])
    neg_factors = shap_explanation.get("negative_factors", [])

    top_pos_str = ""
    if pos_factors:
        pos_names = [f"{f['feature_name']} ({f['actual_value']}, +{f['shap_value']:.2f})" for f in pos_factors[:2]]
        top_pos_str = f"The primary positive drivers elevating this student's standing are {', and '.join(pos_names)}."

    top_neg_str = ""
    if neg_factors:
        neg_names = [f"{f['feature_name']} ({f['actual_value']}, {f['shap_value']:.2f})" for f in neg_factors[:2]]
        top_neg_str = f"Conversely, performance is held back mainly by {', and '.join(neg_names)}."

    # Recommendation synthesis
    if cat == "Low":
        action_advice = (
            "Recommended Intervention: Immediate academic counseling is advised. Focus on improving "
            "attendance consistency, completing outstanding assignments, and attending peer remedial sessions."
        )
    elif cat == "Medium":
        action_advice = (
            "Recommended Guidance: The student is maintaining a passing trajectory. Allocating 3-4 additional "
            "study hours weekly and preparing strategically for internal assessments will help elevate them into High Performance."
        )
    else:
        action_advice = (
            "Recommended Guidance: Outstanding academic consistency! Encourage participation in honors seminars, "
            "peer mentoring, and advanced competitive problem-solving."
        )

    explanation = (
        f"Academic Assessment Summary:\n"
        f"The student is classified into the '{cat} Performance' category with {conf:.1f}% confidence "
        f"(Estimated Final Score: {score:.1f}/100), placing them in the '{risk_desc}' tier.\n\n"
        f"Key Explainable Insights:\n"
        f"{top_pos_str}\n"
        f"{top_neg_str}\n\n"
        f"{action_advice}"
    )

    return explanation.strip()


def generate_llm_explanation(
    prediction_result: Dict[str, Any],
    shap_explanation: Dict[str, Any],
    ollama_model: str = DEFAULT_MODEL,
    use_llm_if_available: bool = True
) -> Dict[str, Any]:
    """
    Verbalizes verified prediction and SHAP outputs using local Qwen2.5-3B via Ollama.
    Automatically falls back to template verbalizer if Ollama is unavailable.

    Returns:
      {
        "explanation": str,
        "source": "ollama_qwen" or "rule_based_template",
        "model_used": str
      }
    """
    if not use_llm_if_available or not check_ollama_available():
        return {
            "explanation": generate_template_explanation(prediction_result, shap_explanation),
            "source": "rule_based_template",
            "model_used": "Python Deterministic Template"
        }

    # Construct prompt with strictly verified data
    cat = prediction_result.get("predicted_category", "Medium")
    conf = prediction_result.get("confidence", 0.0)
    score = prediction_result.get("predicted_score", 0.0)
    pos_factors = [f"{f['feature_name']}: {f['actual_value']} (SHAP +{f['shap_value']:.2f})" for f in shap_explanation.get("positive_factors", [])[:3]]
    neg_factors = [f"{f['feature_name']}: {f['actual_value']} (SHAP {f['shap_value']:.2f})" for f in shap_explanation.get("negative_factors", [])[:3]]

    prompt = (
        f"You are an academic mentor assistant. Verbalize the following verified machine learning prediction and SHAP analysis "
        f"into a 3-4 sentence constructive teacher summary for a student report. "
        f"DO NOT invent facts or change numbers. Rely only on the verified data below:\n\n"
        f"- Predicted Performance: {cat}\n"
        f"- Prediction Confidence: {conf}%\n"
        f"- Estimated Score: {score}/100\n"
        f"- Positive Drivers: {', '.join(pos_factors) if pos_factors else 'None'}\n"
        f"- Negative Constraints: {', '.join(neg_factors) if neg_factors else 'None'}\n\n"
        f"Provide a polite, structured, and actionable paragraph."
    )

    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "max_tokens": 250
        }
    }

    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=6.0)
        if resp.status_code == 200:
            llm_text = resp.json().get("response", "").strip()
            if llm_text:
                return {
                    "explanation": llm_text,
                    "source": "ollama_qwen",
                    "model_used": f"Ollama ({ollama_model})"
                }
    except Exception as e:
        print(f"Ollama call failed or timed out: {e}")

    # Fallback if call fails or times out
    return {
        "explanation": generate_template_explanation(prediction_result, shap_explanation),
        "source": "rule_based_template",
        "model_used": "Python Deterministic Template (Fallback)"
    }
