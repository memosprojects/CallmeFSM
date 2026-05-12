import json

import numpy as np

from src.models import DecoderFunction, DecoderInput, FunctionCallResult


def create_initial_prompt(prompt_text: str, decoder_input: DecoderInput) -> str:
    escaped_prompt = json.dumps(prompt_text)[1:-1]
    functions_info = "\n\n".join(f.routing_block for f in decoder_input.functions)

    rules = (
        "1. Output ONLY valid JSON. No explanation, no extra text.\n"
        "2. Regex patterns: use the simplest pattern that matches exactly what the request describes.\n"
        "   - Examples: \\d+ for digits, [aeiouAEIOU] for vowels, \\bword\\b for whole words.\n"
        "   - When matching individual characters from a set (e.g. vowels), use character class notation [] and include both lowercase and uppercase: [aeiouAEIOU].\n"
        "   - Character classes [] are allowed. NO capturing groups (), NO alternation |, NO trailing ? or * quantifiers, NO regex flags.\n"
        "3. Replacement values: output the exact replacement character or word exactly once. Do NOT repeat it. If the request says 'with asterisks' or 'with *', use '*'. If the request says 'with X' where X is a symbol name, use the symbol character itself.\n"
        "4. Integer parameters: whole number only, no decimal point (e.g., 4, 23).\n"
        "5. Float/number parameters: always include a decimal point (e.g., 16.0, 0.5).\n"
        "6. String parameters: copy the exact value from the request verbatim, character by character."
    )

    return (
        f"Available functions:\n{functions_info}\n\n"
        f"Rules:\n{rules}\n\n"
        f"User request: {prompt_text}\n\n"
        f"JSON output: {{\"prompt\": \"{escaped_prompt}\", \"name\": \""
    )


def get_public_vocab(model) -> dict[int, str]:
    dummy_logits = model.get_logits_from_input_ids([0])
    return {i: model.decode([i]) for i in range(len(dummy_logits))}


def run_dynamic_routing(
    model, init_prompt: str, allowed_names: list[str], vocab: dict[int, str],
    verbose: bool = False,
) -> tuple[str, list[int]]:
    try:
        input_ids: list[int] = model.encode(init_prompt).tolist()[0]
        current_name = ""

        if verbose:
            print("\n[routing]", end="", flush=True)

        while True:
            logits_np = np.array(model.get_logits_from_input_ids(input_ids))
            masked_logits = np.full(logits_np.shape, -np.inf)

            for token_id, token_text in vocab.items():
                if '"' in token_text or ',' in token_text:
                    continue
                potential = current_name + token_text
                if any(name.startswith(potential) for name in allowed_names):
                    masked_logits[token_id] = logits_np[token_id]

            if np.all(masked_logits == -np.inf) or current_name in allowed_names:
                break

            next_id = int(np.argmax(masked_logits))
            token_text = vocab[next_id]
            current_name += token_text
            input_ids.append(next_id)

            if verbose:
                score = float(logits_np[next_id])
                print(f" '{token_text}'({score:.2f})", end="", flush=True)

        if verbose:
            print(f"\n[routing] → selected: '{current_name.strip()}'")

        input_ids.extend(model.encode('"').tolist()[0])
        return current_name.strip(), input_ids
    except Exception:
        return "fn_unknown", []


def _is_naked_quote(text: str) -> bool:
    return text.replace('\\"', '').count('"') > 0


def _sample_from_logits(logits: np.ndarray, temperature: float) -> int:
    scaled = logits / temperature
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def run_parameter_extraction(
    model,
    input_ids: list[int],
    selected_name: str,
    func_lookup: dict[str, DecoderFunction],
    vocab: dict[int, str],
    verbose: bool = False,
    temperature: float = 1.0,
) -> str:
    try:
        input_ids.extend(model.encode(', "parameters": {').tolist()[0])
        params_list = func_lookup[selected_name].parameters

        for i, param in enumerate(params_list):
            key_prefix = f'"{param.name}": '
            if param.type_name == "string":
                key_prefix += '"'
            input_ids.extend(model.encode(key_prefix).tolist()[0])

            if verbose:
                print(f"\n[param: {param.name} ({param.type_name})]", end="", flush=True)

            step = 0
            limit = 200

            while step < limit:
                logits_np = np.array(model.get_logits_from_input_ids(input_ids))
                masked_logits = np.full(logits_np.shape, -np.inf)

                for tid, text in vocab.items():
                    if param.type_name == "integer":
                        clean = text.strip()
                        if clean and all(c in "0123456789-" for c in clean) and not any(c in text for c in '"{}[] ,'):
                            masked_logits[tid] = logits_np[tid]
                    elif param.type_name == "number":
                        clean = text.strip()
                        if clean and all(c in "0123456789.-" for c in clean) and not any(c in text for c in '"{}[] ,'):
                            masked_logits[tid] = logits_np[tid]
                    else:
                        if not _is_naked_quote(text):
                            masked_logits[tid] = logits_np[tid]

                if np.all(masked_logits == -np.inf):
                    if verbose:
                        print(" [no valid tokens]", flush=True)
                    break

                raw_top_id = int(np.argmax(logits_np))
                raw_top_text = vocab[raw_top_id]

                stop_signal = False
                if param.type_name == "string":
                    if raw_top_text.strip() == '"' or raw_top_text.startswith('",') or raw_top_text.startswith('"}'):
                        stop_signal = True
                elif param.type_name == "integer":
                    if not any(c in raw_top_text for c in "0123456789-") and raw_top_text.strip():
                        stop_signal = True
                else:
                    if not any(c in raw_top_text for c in "0123456789.-") and raw_top_text.strip():
                        stop_signal = True

                if stop_signal and step > 0:
                    if verbose:
                        print(f" [stop: '{raw_top_text}']", flush=True)
                    break

                next_id = _sample_from_logits(masked_logits, temperature)
                token_text = vocab[next_id]
                input_ids.append(next_id)
                step += 1

                if verbose:
                    score = float(logits_np[next_id])
                    print(f" '{token_text}'({score:.2f})", end="", flush=True)

            sep = '"' if param.type_name == "string" else ""
            if i < len(params_list) - 1:
                sep += ", "
            if sep:
                input_ids.extend(model.encode(sep).tolist()[0])

        input_ids.extend(model.encode("}}").tolist()[0])
        return model.decode(input_ids)
    except Exception:
        return "{}"


def _fix_json_backslashes(s: str) -> str:
    valid_after = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            if s[i + 1] in valid_after:
                result.append('\\')
                result.append(s[i + 1])
                i += 2
            else:
                result.append('\\\\')
                i += 1
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


def _post_process_parameters(result_data: dict, fn_def: DecoderFunction) -> None:
    for param in fn_def.parameters:
        val = result_data["parameters"].get(param.name)
        if val is None:
            continue
        if param.type_name == "integer":
            result_data["parameters"][param.name] = int(float(val))
        elif param.type_name == "number":
            result_data["parameters"][param.name] = float(val)


def post_process_result(
    generated_raw: str, selected_name: str, func_lookup: dict[str, DecoderFunction]
) -> FunctionCallResult | None:
    start_idx = generated_raw.rfind('{"prompt":')
    if start_idx == -1:
        return None

    json_candidate = generated_raw[start_idx:]
    last_brace = json_candidate.rfind('}')
    if last_brace == -1:
        return None

    result_data = json.loads(_fix_json_backslashes(json_candidate[:last_brace + 1]))

    fn_def = func_lookup.get(selected_name)
    if fn_def and "parameters" in result_data:
        _post_process_parameters(result_data, fn_def)

    return FunctionCallResult.model_validate(result_data)


def decode_prompt(
    model, prompt_text: str, decoder_input: DecoderInput, vocab: dict[int, str],
    verbose: bool = False,
    temperature: float = 1.0,
) -> FunctionCallResult | None:
    init_prompt = create_initial_prompt(prompt_text, decoder_input)
    allowed_names = [f.name for f in decoder_input.functions]
    func_lookup = decoder_input.function_by_name

    selected_name, updated_ids = run_dynamic_routing(model, init_prompt, allowed_names, vocab, verbose=verbose)
    generated_raw = run_parameter_extraction(model, updated_ids, selected_name, func_lookup, vocab, verbose=verbose, temperature=temperature)
    return post_process_result(generated_raw, selected_name, func_lookup)
