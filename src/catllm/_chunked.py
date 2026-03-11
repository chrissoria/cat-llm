"""
Chunked classification for CatLLM.

When users have large category lists, this module splits them into smaller
chunks, runs a separate LLM call per chunk with local 1..N numbering, and
merges the results back into global numbering so downstream code
(aggregate_results, build_output_dataframes) sees a single merged JSON dict.

Each chunk automatically gets a temporary "Other" catch-all category appended
(unless one is already present in the chunk). This gives the LLM an escape
hatch for ambiguous responses, improving classification accuracy. The "Other"
column is dropped before merging back to global keys, so the final output
only contains the user's real categories.
"""

import json
import math

from .text_functions import (
    build_json_schema,
    extract_json,
    validate_classification_json,
    ollama_two_step_classify,
)
from ._category_analysis import has_other_category


def run_chunked_classification(
    *,
    client,
    cfg,
    item,
    categories,
    categories_str,
    example_json,
    json_schema,
    cove_original_task,
    effective_creativity,
    use_json_schema,
    survey_question,
    survey_question_context,
    examples_text,
    chain_of_thought,
    context_prompt,
    step_back_prompt,
    stepback_insights,
    chain_of_verification,
    thinking_budget,
    max_retries,
    multi_label,
    categories_per_call,
    add_unified_other=False,
    formatter_fallback_fn,
    # Mode-specific
    is_pdf_mode,
    is_image_mode,
    pdf_mode=None,
    pdf_dpi=150,
    input_description="",
    # Prompt builders (passed in to avoid circular imports)
    build_text_prompt_fn=None,
    build_pdf_prompt_fn=None,
    build_image_prompt_fn=None,
    google_multimodal_fn=None,
    prepare_page_data_fn=None,
    prepare_image_data_fn=None,
    build_cove_prompts_fn=None,
    run_cove_fn=None,
):
    """
    Run chunked classification for one item across category chunks.

    Splits the full category list into chunks of `categories_per_call`,
    runs one LLM call per chunk, and merges results with key remapping.

    Returns:
        tuple: (json_result_str, error) — same contract as a single LLM call
    """
    # Build chunks: list of (chunk_categories, global_offset)
    chunks = []
    for start in range(0, len(categories), categories_per_call):
        chunk_cats = categories[start : start + categories_per_call]
        chunks.append((chunk_cats, start))

    merged_json = {}
    chunk_other_values = []  # Track per-chunk "Other" values for unification

    for chunk_cats, global_offset in chunks:
        # Add temporary "Other" catch-all if the chunk doesn't already have one.
        # This gives the LLM an escape hatch for ambiguous responses, improving
        # accuracy. The "Other" key is dropped before merging to global keys.
        added_other = False
        num_real_cats = len(chunk_cats)
        if not has_other_category(chunk_cats):
            chunk_cats_for_call = list(chunk_cats) + ["Other"]
            added_other = True
        else:
            chunk_cats_for_call = chunk_cats

        # Build chunk-local prompt components (with "Other" if added)
        chunk_categories_str = "\n".join(
            f"{j+1}. {cat}" for j, cat in enumerate(chunk_cats_for_call)
        )
        chunk_example_json = json.dumps(
            {str(j + 1): "0" for j in range(len(chunk_cats_for_call))}, indent=2
        )
        chunk_json_schema = (
            build_json_schema(
                chunk_cats_for_call,
                include_additional_properties=(cfg["provider"] != "google"),
            )
            if use_json_schema
            else None
        )

        # Rebuild CoVe task for this chunk if CoVe enabled
        chunk_cove_task = ""
        if chain_of_verification:
            if multi_label:
                cove_categorize = "into the following categories"
                cove_json = 'Provide your answer in JSON format where the category number is the key and "1" if present, "0" if not.'
            else:
                cove_categorize = "into the single most appropriate category"
                cove_json = 'Provide your answer in JSON format where the category number is the key. Assign "1" to the single best matching category and "0" to all others.'
            chunk_cove_task = f"""{survey_question_context}
Categorize survey responses {cove_categorize}:
{chunk_categories_str}
{cove_json}"""

        # Run one LLM call for this chunk (with "Other" included)
        chunk_result, chunk_error = _run_single_chunk_call(
            client=client,
            cfg=cfg,
            item=item,
            chunk_cats=chunk_cats_for_call,
            chunk_categories_str=chunk_categories_str,
            chunk_json_schema=chunk_json_schema,
            chunk_example_json=chunk_example_json,
            chunk_cove_task=chunk_cove_task,
            effective_creativity=effective_creativity,
            survey_question=survey_question,
            survey_question_context=survey_question_context,
            examples_text=examples_text,
            chain_of_thought=chain_of_thought,
            context_prompt=context_prompt,
            step_back_prompt=step_back_prompt,
            stepback_insights=stepback_insights,
            chain_of_verification=chain_of_verification,
            thinking_budget=thinking_budget,
            max_retries=max_retries,
            multi_label=multi_label,
            formatter_fallback_fn=formatter_fallback_fn,
            is_pdf_mode=is_pdf_mode,
            is_image_mode=is_image_mode,
            pdf_mode=pdf_mode,
            pdf_dpi=pdf_dpi,
            input_description=input_description,
            build_text_prompt_fn=build_text_prompt_fn,
            build_pdf_prompt_fn=build_pdf_prompt_fn,
            build_image_prompt_fn=build_image_prompt_fn,
            google_multimodal_fn=google_multimodal_fn,
            prepare_page_data_fn=prepare_page_data_fn,
            prepare_image_data_fn=prepare_image_data_fn,
            build_cove_prompts_fn=build_cove_prompts_fn,
            run_cove_fn=run_cove_fn,
        )

        if chunk_error:
            return (json.dumps(merged_json) if merged_json else '{"1":"e"}', chunk_error)

        # Remap chunk-local keys (1..N) to global keys, dropping "Other"
        try:
            chunk_parsed = json.loads(chunk_result)
        except (json.JSONDecodeError, TypeError):
            return ('{"1":"e"}', f"Failed to parse chunk result: {chunk_result}")

        # The "Other" key (if added) is the last one: str(num_real_cats + 1)
        other_local_key = str(num_real_cats + 1) if added_other else None

        for local_key, value in chunk_parsed.items():
            # Capture the temporary "Other" value, don't merge it
            if local_key == other_local_key:
                try:
                    chunk_other_values.append(int(value))
                except (ValueError, TypeError):
                    chunk_other_values.append(0)
                continue
            try:
                global_key = str(global_offset + int(local_key))
                merged_json[global_key] = value
            except (ValueError, TypeError):
                # Non-numeric key — skip (shouldn't happen with proper schemas)
                pass

    # Unified "Other": if all real categories are 0 but at least one chunk's
    # "Other" fired, the response genuinely doesn't fit any category.
    if add_unified_other:
        real_sum = sum(
            int(v) for v in merged_json.values()
            if str(v).strip() in ("0", "1")
        )
        other_sum = sum(chunk_other_values)
        unified_other = "1" if real_sum == 0 and other_sum > 0 else "0"
        merged_json[str(len(categories) + 1)] = unified_other

    return (json.dumps(merged_json), None)


def _run_single_chunk_call(
    *,
    client,
    cfg,
    item,
    chunk_cats,
    chunk_categories_str,
    chunk_json_schema,
    chunk_example_json,
    chunk_cove_task,
    effective_creativity,
    survey_question,
    survey_question_context,
    examples_text,
    chain_of_thought,
    context_prompt,
    step_back_prompt,
    stepback_insights,
    chain_of_verification,
    thinking_budget,
    max_retries,
    multi_label,
    formatter_fallback_fn,
    is_pdf_mode,
    is_image_mode,
    pdf_mode,
    pdf_dpi,
    input_description,
    build_text_prompt_fn,
    build_pdf_prompt_fn,
    build_image_prompt_fn,
    google_multimodal_fn,
    prepare_page_data_fn,
    prepare_image_data_fn,
    build_cove_prompts_fn,
    run_cove_fn,
):
    """
    Run one LLM call for one chunk of categories on one item.

    Returns:
        tuple: (json_result_str, error)
    """
    thinking_providers = ("google", "openai", "anthropic", "huggingface", "huggingface-together")

    # =================================================================
    # PDF MODE
    # =================================================================
    if is_pdf_mode and isinstance(item, tuple):
        pdf_path, page_index, page_label = item

        page_data = prepare_page_data_fn(
            pdf_path=pdf_path,
            page_index=page_index,
            page_label=page_label,
            pdf_mode=pdf_mode,
            provider=cfg["provider"],
            pdf_dpi=pdf_dpi,
        )

        if page_data.get("error"):
            return ('{"1":"e"}', page_data["error"])

        messages = build_pdf_prompt_fn(
            page_data=page_data,
            categories_str=chunk_categories_str,
            input_description=input_description,
            provider=cfg["provider"],
            pdf_mode=pdf_mode,
            chain_of_thought=chain_of_thought,
            context_prompt=context_prompt,
            step_back_prompt=step_back_prompt,
            stepback_insights=stepback_insights,
            model_name=cfg["model"],
            example_json=chunk_example_json,
            multi_label=multi_label,
        )

        if cfg["provider"] == "google":
            reply, error = google_multimodal_fn(
                client=client,
                messages=messages,
                json_schema=chunk_json_schema,
                creativity=effective_creativity,
                thinking_budget=thinking_budget,
                max_retries=max_retries,
            )
        else:
            reply, error = client.complete(
                messages=messages,
                json_schema=chunk_json_schema,
                creativity=effective_creativity,
                thinking_budget=thinking_budget if cfg["provider"] in thinking_providers else None,
                max_retries=max_retries,
            )

        if error:
            return ('{"1":"e"}', error)

        json_result = extract_json(reply)
        json_result = formatter_fallback_fn(json_result, reply, chunk_cats)
        return (json_result, None)

    # =================================================================
    # IMAGE MODE
    # =================================================================
    elif is_image_mode and isinstance(item, tuple):
        image_path, image_label = item

        image_data = prepare_image_data_fn(image_path, image_label)

        if image_data.get("error"):
            return ('{"1":"e"}', image_data["error"])

        messages = build_image_prompt_fn(
            image_data=image_data,
            categories_str=chunk_categories_str,
            input_description=input_description,
            provider=cfg["provider"],
            chain_of_thought=chain_of_thought,
            context_prompt=context_prompt,
            step_back_prompt=step_back_prompt,
            stepback_insights=stepback_insights,
            model_name=cfg["model"],
            example_json=chunk_example_json,
            multi_label=multi_label,
        )

        if cfg["provider"] == "google":
            reply, error = google_multimodal_fn(
                client=client,
                messages=messages,
                json_schema=chunk_json_schema,
                creativity=effective_creativity,
                thinking_budget=thinking_budget,
                max_retries=max_retries,
            )
        else:
            reply, error = client.complete(
                messages=messages,
                json_schema=chunk_json_schema,
                creativity=effective_creativity,
                thinking_budget=thinking_budget if cfg["provider"] in thinking_providers else None,
                max_retries=max_retries,
            )

        if error:
            return ('{"1":"e"}', error)

        json_result = extract_json(reply)
        json_result = formatter_fallback_fn(json_result, reply, chunk_cats)
        return (json_result, None)

    # =================================================================
    # TEXT MODE
    # =================================================================
    else:
        response_text = item

        if cfg["use_two_step"]:  # Ollama
            json_result, error = ollama_two_step_classify(
                client=client,
                response_text=response_text,
                categories=chunk_cats,
                categories_str=chunk_categories_str,
                survey_question=survey_question,
                creativity=effective_creativity,
                max_retries=max_retries,
            )
            if not error:
                json_result = formatter_fallback_fn(json_result, json_result, chunk_cats)
            return (json_result, error)
        else:
            messages = build_text_prompt_fn(
                response_text=response_text,
                categories_str=chunk_categories_str,
                survey_question_context=survey_question_context,
                examples_text=examples_text,
                chain_of_thought=chain_of_thought,
                context_prompt=context_prompt,
                step_back_prompt=step_back_prompt,
                stepback_insights=stepback_insights,
                model_name=cfg["model"],
                multi_label=multi_label,
            )
            reply, error = client.complete(
                messages=messages,
                json_schema=chunk_json_schema,
                creativity=effective_creativity,
                thinking_budget=thinking_budget if cfg["provider"] in thinking_providers else None,
                max_retries=max_retries,
            )
            if error:
                return ('{"1":"e"}', error)

            json_result = extract_json(reply)
            json_result = formatter_fallback_fn(json_result, reply, chunk_cats)

            # Run Chain of Verification if enabled
            if chain_of_verification:
                step2, step3, step4 = build_cove_prompts_fn(
                    chunk_cove_task, response_text
                )
                json_result = run_cove_fn(
                    client=client,
                    initial_reply=json_result,
                    step2_prompt=step2,
                    step3_prompt=step3,
                    step4_prompt=step4,
                    json_schema=chunk_json_schema,
                    creativity=effective_creativity,
                    max_retries=max_retries,
                )
                json_result = formatter_fallback_fn(json_result, json_result, chunk_cats)

            return (json_result, None)
