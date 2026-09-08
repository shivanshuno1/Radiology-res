# src/gemini_retry.py

import time
import traceback
from typing import Any, Callable, Type

from pydantic import BaseModel

from qwen_model import generate_qwen_structured


# ============================================================
# CONFIGURATION
# ============================================================

MAX_GEMINI_ATTEMPTS = 3


# ============================================================
# RETRYABLE GEMINI ERRORS
# ============================================================

def is_retryable_error(error: Exception) -> bool:
    """
    Returns True when the Gemini error is likely temporary.

    These errors are worth retrying before falling back to Qwen.
    """

    text = str(error).upper()

    retry_keywords = [
        "429",
        "500",
        "502",
        "503",
        "504",

        "RESOURCE_EXHAUSTED",
        "RATE LIMIT",
        "TOO MANY REQUESTS",

        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",

        "TIMEOUT",
        "TIMED OUT",

        "SERVICE UNAVAILABLE",
        "INTERNAL SERVER ERROR",
        "BAD GATEWAY",
        "GATEWAY TIMEOUT",
    ]

    return any(
        keyword in text
        for keyword in retry_keywords
    )


# ============================================================
# GEMINI + QWEN FALLBACK
# ============================================================

def generate_content_with_retry(
    generate_content: Callable[..., Any],
    *,
    max_attempts: int = MAX_GEMINI_ATTEMPTS,

    fallback_prompt: str | None = None,
    fallback_system_instruction: str | None = None,
    fallback_schema: Type[BaseModel] | None = None,

    **kwargs: Any,
) -> Any:
    """
    Primary model:
        Gemini

    Fallback model:
        Qwen3-8B through Hugging Face

    Behaviour:

        Gemini success
            -> return Gemini response

        Gemini temporary failure
            -> retry Gemini

        Gemini permanent failure
            -> immediately use Qwen

        Gemini exhausted retries
            -> use Qwen

        Qwen failure
            -> raise the actual Qwen error
    """

    last_gemini_error: Exception | None = None

    # ========================================================
    # 1. GEMINI PRIMARY
    # ========================================================

    for attempt in range(1, max_attempts + 1):

        try:

            print(
                f"[Gemini] Attempt "
                f"{attempt}/{max_attempts}"
            )

            response = generate_content(
                **kwargs
            )

            print(
                "[Gemini] Success."
            )

            return response

        except Exception as error:

            last_gemini_error = error

            print(
                f"[Gemini] Failed: {error}"
            )

            # ------------------------------------------------
            # TEMPORARY ERROR
            # ------------------------------------------------

            if is_retryable_error(error):

                if attempt < max_attempts:

                    wait_time = 2 ** (
                        attempt - 1
                    )

                    print(
                        "[Gemini] Temporary error."
                    )

                    print(
                        f"[Gemini] Retrying in "
                        f"{wait_time} second(s)..."
                    )

                    time.sleep(
                        wait_time
                    )

                    continue

                # --------------------------------------------
                # Maximum attempts reached
                # --------------------------------------------

                print(
                    "[Gemini] Maximum retry attempts "
                    "reached."
                )

                break

            # ------------------------------------------------
            # PERMANENT / UNKNOWN ERROR
            #
            # IMPORTANT:
            #
            # We DO NOT raise here.
            #
            # We go directly to Qwen.
            # ------------------------------------------------

            print(
                "[Gemini] Non-retryable error."
            )

            print(
                "[Gemini] Switching to Qwen..."
            )

            break

    # ========================================================
    # 2. VALIDATE FALLBACK CONFIGURATION
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "[Fallback] Gemini unavailable."
    )

    print(
        "[Fallback] Preparing Qwen3-8B..."
    )

    print(
        "=========================================="
    )

    if fallback_prompt is None:

        raise RuntimeError(
            "Gemini failed, but fallback_prompt "
            "was not provided."
        ) from last_gemini_error

    if fallback_system_instruction is None:

        raise RuntimeError(
            "Gemini failed, but "
            "fallback_system_instruction "
            "was not provided."
        ) from last_gemini_error

    if fallback_schema is None:

        raise RuntimeError(
            "Gemini failed, but fallback_schema "
            "was not provided."
        ) from last_gemini_error

    # ========================================================
    # 3. QWEN FALLBACK
    # ========================================================

    try:

        print(
            "[Fallback] Calling Qwen3-8B..."
        )

        result = generate_qwen_structured(

            prompt=fallback_prompt,

            system_instruction=(
                fallback_system_instruction
            ),

            schema=fallback_schema,
        )

        print(
            "[Fallback] Qwen3-8B succeeded."
        )

        return result

    # ========================================================
    # 4. QWEN ERROR
    # ========================================================

    except Exception as qwen_error:

        print(
            "\n=========================================="
        )

        print(
            "[Fallback] Qwen3-8B FAILED."
        )

        print(
            "=========================================="
        )

        print(
            f"Exception type: "
            f"{type(qwen_error).__name__}"
        )

        print(
            f"Exception: {qwen_error}"
        )

        print(
            "\nFull Qwen traceback:"
        )

        traceback.print_exc()

        # --------------------------------------------
        # IMPORTANT:
        #
        # Preserve the REAL Qwen exception.
        # --------------------------------------------

        raise RuntimeError(
            "Both Gemini and Qwen failed.\n\n"
            f"Gemini error:\n"
            f"{last_gemini_error}\n\n"
            f"Qwen error:\n"
            f"{qwen_error}"
        ) from qwen_error