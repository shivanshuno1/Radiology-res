import os
import json
import re
from typing import Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel
from huggingface_hub import InferenceClient


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found in .env. "
        "Add HF_TOKEN=your_huggingface_token to the .env file."
    )


# ============================================================
# QWEN CONFIGURATION
# ============================================================

QWEN_MODEL = "Qwen/Qwen3-8B"
QWEN_PROVIDER = "nscale"

client = InferenceClient(
    provider=QWEN_PROVIDER,
    api_key=HF_TOKEN,
)

T = TypeVar("T", bound=BaseModel)


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text: str) -> dict | list:
    """
    Extract JSON from Qwen's response.

    Handles:
    1. Pure JSON
    2. JSON inside ```json ... ```
    3. JSON surrounded by additional text
    """

    if not text:
        raise ValueError("Qwen returned EMPTY text.")

    text = text.strip()

    # --------------------------------------------------------
    # CASE 1: Response is already valid JSON
    # --------------------------------------------------------

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # CASE 2: JSON inside Markdown code block
    # --------------------------------------------------------

    match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:

        candidate = match.group(1).strip()

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            pass

    # --------------------------------------------------------
    # CASE 3: Find JSON object
    # --------------------------------------------------------

    start_object = text.find("{")
    end_object = text.rfind("}")

    if (
        start_object != -1
        and end_object != -1
        and end_object > start_object
    ):

        candidate = text[
            start_object:end_object + 1
        ]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            pass

    # --------------------------------------------------------
    # CASE 4: Find JSON array
    # --------------------------------------------------------

    start_array = text.find("[")
    end_array = text.rfind("]")

    if (
        start_array != -1
        and end_array != -1
        and end_array > start_array
    ):

        candidate = text[
            start_array:end_array + 1
        ]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            pass

    # --------------------------------------------------------
    # FAILURE
    # --------------------------------------------------------

    raise ValueError(
        "Qwen returned invalid JSON.\n\n"
        "RAW RESPONSE:\n"
        f"{text}"
    )


# ============================================================
# QWEN STRUCTURED GENERATION
# ============================================================

def generate_qwen_structured(
    *,
    prompt: str,
    system_instruction: str,
    schema: Type[T],
) -> T:

    print("\n==========================================")
    print("[Qwen] Building request...")
    print("==========================================")

    # --------------------------------------------------------
    # Generate Pydantic JSON schema
    # --------------------------------------------------------

    schema_json = json.dumps(
        schema.model_json_schema(),
        indent=2,
    )

    # --------------------------------------------------------
    # Strict system prompt
    # --------------------------------------------------------

    system_prompt = f"""
{system_instruction}

============================================================
STRICT OUTPUT CONTRACT
============================================================

You are being used as a structured JSON generation engine.

You MUST follow these rules:

1. Return ONLY ONE valid JSON object.

2. Do NOT return a JSON array as the top-level response.

3. Do NOT provide reasoning.

4. Do NOT provide analysis.

5. Do NOT explain your decisions.

6. Do NOT repeat the user's prompt.

7. Do NOT repeat the dictation.

8. Do NOT repeat the template.

9. Do NOT include Markdown.

10. Do NOT use ```json.

11. Do NOT write any text before the JSON.

12. Do NOT write any text after the JSON.

13. The response MUST be complete valid JSON.

14. Follow the supplied schema exactly.

15. Stop immediately after the final closing brace.

16. Do not output unnecessary fields.

17. Do not output comments inside the JSON.

18. Do not output trailing commas.

19. Keep strings concise.

20. If the schema contains a "findings" field, the top-level
    structure MUST be:

    {{
        "findings": [...]
    }}

============================================================
EXPECTED JSON SCHEMA
============================================================

{schema_json}

============================================================
END OUTPUT CONTRACT
============================================================
"""

    # --------------------------------------------------------
    # Messages
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # --------------------------------------------------------
    # Debug information
    # --------------------------------------------------------

    print("[Qwen] Model:", QWEN_MODEL)
    print("[Qwen] Provider:", QWEN_PROVIDER)
    print("[Qwen] Temperature: 0.0")
    print("[Qwen] Max tokens: 8192")
    print("[Qwen] Sending request...")

    # --------------------------------------------------------
    # API REQUEST
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=4500,
        )

    except Exception as e:

        print("\n==========================================")
        print("[Qwen] REQUEST FAILED")
        print("==========================================")

        print(
            "Exception type:",
            type(e).__name__,
        )

        print(
            "Exception:",
            repr(e),
        )

        raise

    # --------------------------------------------------------
    # Response received
    # --------------------------------------------------------

    print("\n==========================================")
    print("[Qwen] Request completed.")
    print("==========================================")

    # --------------------------------------------------------
    # Check choices
    # --------------------------------------------------------

    if not response.choices:

        raise ValueError(
            "Qwen returned no choices."
        )

    # --------------------------------------------------------
    # Get first message
    # --------------------------------------------------------

    message = response.choices[0].message

    # --------------------------------------------------------
    # Finish reason
    # --------------------------------------------------------

    finish_reason = getattr(
        response.choices[0],
        "finish_reason",
        None,
    )

    print(
        "[Qwen] Finish reason:",
        finish_reason,
    )

    # --------------------------------------------------------
    # Token usage
    # --------------------------------------------------------

    usage = getattr(
        response,
        "usage",
        None,
    )

    if usage:

        print(
            "[Qwen] Prompt tokens:",
            getattr(
                usage,
                "prompt_tokens",
                None,
            ),
        )

        print(
            "[Qwen] Completion tokens:",
            getattr(
                usage,
                "completion_tokens",
                None,
            ),
        )

        print(
            "[Qwen] Total tokens:",
            getattr(
                usage,
                "total_tokens",
                None,
            ),
        )

    # --------------------------------------------------------
    # Warn if output was truncated
    # --------------------------------------------------------

    if finish_reason == "length":

        print("\n==========================================")
        print("[Qwen] WARNING: OUTPUT WAS TRUNCATED")
        print("==========================================")

        raise ValueError(
            "Qwen output was truncated because the maximum "
            "token limit was reached."
        )

    # --------------------------------------------------------
    # Content
    # --------------------------------------------------------

    content = message.content

    print("\n[Qwen] CONTENT:")
    print(repr(content))

    # --------------------------------------------------------
    # Validate content
    # --------------------------------------------------------

    if content is None:

        raise ValueError(
            "Qwen returned content=None."
        )

    if not content.strip():

        raise ValueError(
            "Qwen returned an EMPTY STRING."
        )

    # --------------------------------------------------------
    # Extract JSON
    # --------------------------------------------------------

    data = extract_json(content)

    # --------------------------------------------------------
    # Defensive top-level list handling
    # --------------------------------------------------------

    if isinstance(data, list):

        schema_fields = getattr(
            schema,
            "model_fields",
            {},
        )

        if "findings" in schema_fields:

            print(
                "\n[Qwen] WARNING:"
                " Model returned a top-level list."
            )

            print(
                "[Qwen] Converting list to "
                "{'findings': [...]}."
            )

            data = {
                "findings": data
            }

        else:

            raise ValueError(
                "Qwen returned a JSON array, but the expected "
                "schema requires a JSON object."
            )

    # --------------------------------------------------------
    # Ensure top-level JSON object
    # --------------------------------------------------------

    if not isinstance(data, dict):

        raise ValueError(
            "Qwen returned JSON, but it is neither "
            "an object nor an accepted list."
        )

    # --------------------------------------------------------
    # Print parsed JSON
    # --------------------------------------------------------

    print("\n==========================================")
    print("[Qwen] Parsed JSON")
    print("==========================================")

    print(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # Pydantic validation
    # --------------------------------------------------------

    try:

        result = schema.model_validate(data)

    except Exception as e:

        print("\n==========================================")
        print("[Qwen] PYDANTIC VALIDATION FAILED")
        print("==========================================")

        print(
            "Exception type:",
            type(e).__name__,
        )

        print(
            "Exception:",
            repr(e),
        )

        raise

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print("\n==========================================")
    print("[Qwen] Pydantic validation successful.")
    print("==========================================")

    return result


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    class TestResult(BaseModel):
        answer: str

    print("\n================================")
    print("QWEN DIRECT TEST")
    print("================================")

    result = generate_qwen_structured(
        system_instruction=(
            "Answer the user's question concisely."
        ),
        prompt=(
            "What is machine learning? "
            "Answer in one sentence."
        ),
        schema=TestResult,
    )

    print("\n================================")
    print("FINAL RESULT")
    print("================================")

    print(result)

    print("\nANSWER:")
    print(result.answer)