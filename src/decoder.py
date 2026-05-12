import json

import numpy as np

from src.models import DecoderFunction, DecoderInput, FunctionCallResult

_REGEX_VALID_ESCAPES = set('dwsDWSBb\\^$.|+*?[]{}()nrtavf0123456789')


def _regex_next_state(state: str, c: str) -> str:
    if state == 'init':
        if c == '\\':
            return 'escape'
        if c == '[':
            return 'char_class_start'
        return 'invalid'
    if state == 'after_quant':
        if c in '()|*?+"':
            return 'invalid'
        if c == '\\':
            return 'escape'
        if c == '[':
            return 'char_class_start'
        return 'has_atom'
    if state == 'has_atom':
        if c in '()|*?"':
            return 'invalid'
        if c == '+':
            return 'after_quant'
        if c == '\\':
            return 'escape'
        if c == '[':
            return 'char_class_start'
        return 'has_atom'
    if state == 'escape':
        return 'has_atom' if c in _REGEX_VALID_ESCAPES else 'invalid'
    if state == 'char_class_start':
        if c in ('^', ']', '"', '\\'):
            return 'invalid'
        return 'char_class'
    if state == 'char_class':
        if c in ('"', '\\'):
            return 'invalid'
        if c == ']':
            return 'has_atom'
        return 'char_class'
    return 'invalid'


def _regex_state_of(s: str) -> str:
    state = 'init'
    for c in s:
        state = _regex_next_state(state, c)
        if state == 'invalid':
            return 'invalid'
    return state


def _is_valid_regex_continuation(state: str, token: str) -> bool:
    for c in token:
        state = _regex_next_state(state, c)
        if state == 'invalid':
            return False
    return True


def _is_complete_regex(s: str) -> bool:
    return bool(s) and _regex_state_of(s) in ('has_atom', 'after_quant')


def create_initial_prompt(
    prompt_text: str, decoder_input: DecoderInput
) -> str:
    escaped_prompt = json.dumps(prompt_text)[1:-1]
    functions_info = "\n\n".join(
        f.routing_block for f in decoder_input.functions
    )

    rules = (
        "1. Output ONLY valid JSON. No explanation, no extra text.\n"
        "2. Regex patterns: use the simplest pattern that matches exactly what the request describes.\n"  # noqa: E501
        "   - Examples: \\d+ for digits, [aeiouAEIOU] for vowels, \\bword\\b for whole words.\n"  # noqa: E501
        "   - When matching individual characters from a set (e.g. vowels), use character class notation [] and include both lowercase and uppercase: [aeiouAEIOU].\n"  # noqa: E501
        "   - Character classes [] are allowed. NO capturing groups (), NO alternation |, NO trailing ? or * quantifiers, NO regex flags.\n"  # noqa: E501
        "3. Replacement values: output the exact replacement character or word exactly once. Do NOT repeat it. If the request says 'with asterisks' or 'with *', use '*'. If the request says 'with X' where X is a symbol name, use the symbol character itself.\n"  # noqa: E501
        "4. Integer parameters: whole number only, no decimal point (e.g., 4, 23).\n"  # noqa: E501
        "5. Float/number parameters: always include a decimal point (e.g., 16.0, 0.5).\n"  # noqa: E501
        "6. String parameters: copy the exact value from the request verbatim, character by character."  # noqa: E501
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

            if (
                np.all(masked_logits == -np.inf)
                or current_name in allowed_names
            ):
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
                print(
                    f"\n[param: {param.name} ({param.type_name})]",
                    end="", flush=True
                )

            partial = ""
            partial_state = 'init'
            just_closed_class_flag = False
            step = 0
            limit = 200

            while step < limit:
                if param.name == "regex" and step > 0:
                    if just_closed_class_flag:
                        if verbose:
                            print(" [stop: class closed]", flush=True)
                        break
                    if partial_state == 'after_quant':
                        if verbose:
                            print(" [stop: after_quant]", flush=True)
                        break

                raw = model.get_logits_from_input_ids(input_ids)
                logits_np = np.array(raw)
                masked_logits = np.full(logits_np.shape, -np.inf)

                if param.name == "regex":
                    for tid, text in vocab.items():
                        stripped = text.strip()
                        if stripped and _is_valid_regex_continuation(
                            partial_state, stripped
                        ):
                            masked_logits[tid] = logits_np[tid]
                elif param.type_name == "integer":
                    for tid, text in vocab.items():
                        clean = text.strip()
                        if (
                            clean
                            and all(c in "0123456789-" for c in clean)
                            and not any(c in text for c in '"{}[] ,')
                        ):
                            masked_logits[tid] = logits_np[tid]
                elif param.type_name == "number":
                    for tid, text in vocab.items():
                        clean = text.strip()
                        if (
                            clean
                            and all(c in "0123456789.-" for c in clean)
                            and not any(c in text for c in '"{}[] ,')
                        ):
                            masked_logits[tid] = logits_np[tid]
                else:
                    for tid, text in vocab.items():
                        if not _is_naked_quote(text):
                            masked_logits[tid] = logits_np[tid]

                if np.all(masked_logits == -np.inf):
                    if verbose:
                        print(" [no valid tokens]", flush=True)
                    break

                raw_top_id = int(np.argmax(logits_np))
                raw_top_text = vocab[raw_top_id]

                stop_signal = False
                if param.name == "regex":
                    if _is_complete_regex(partial) and (
                        raw_top_text.strip() == '"'
                        or raw_top_text.startswith('",')
                        or raw_top_text.startswith('"}')
                    ):
                        stop_signal = True
                elif param.type_name == "string":
                    if (
                        raw_top_text.strip() == '"'
                        or raw_top_text.startswith('",')
                        or raw_top_text.startswith('"}')
                    ):
                        stop_signal = True
                elif param.type_name == "integer":
                    if (
                        not any(c in raw_top_text for c in "0123456789-")
                        and raw_top_text.strip()
                    ):
                        stop_signal = True
                else:
                    if (
                        not any(c in raw_top_text for c in "0123456789.-")
                        and raw_top_text.strip()
                    ):
                        stop_signal = True

                if stop_signal and step > 0:
                    if verbose:
                        print(f" [stop: '{raw_top_text}']", flush=True)
                    break

                next_id = _sample_from_logits(masked_logits, temperature)
                token_text = vocab[next_id]
                input_ids.append(next_id)

                if param.name == "regex":
                    just_closed_class_flag = False
                    for c in token_text.strip():
                        prev = partial_state
                        partial_state = _regex_next_state(partial_state, c)
                        if prev == 'char_class' and partial_state == 'has_atom':
                            just_closed_class_flag = True
                    partial += token_text.strip()

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


def _post_process_parameters(
    result_data: dict, fn_def: DecoderFunction
) -> None:
    for param in fn_def.parameters:
        val = result_data["parameters"].get(param.name)
        if val is None:
            continue
        if param.type_name == "integer":
            result_data["parameters"][param.name] = int(float(val))
        elif param.type_name == "number":
            result_data["parameters"][param.name] = float(val)
        elif param.type_name == "string" and isinstance(val, str):
            result_data["parameters"][param.name] = val.strip()


def post_process_result(
    generated_raw: str,
    selected_name: str,
    func_lookup: dict[str, DecoderFunction],
) -> FunctionCallResult | None:
    start_idx = generated_raw.rfind('{"prompt":')
    if start_idx == -1:
        return None

    json_candidate = generated_raw[start_idx:]
    last_brace = json_candidate.rfind('}')
    if last_brace == -1:
        return None

    fixed = _fix_json_backslashes(json_candidate[:last_brace + 1])
    result_data = json.loads(fixed)

    fn_def = func_lookup.get(selected_name)
    if fn_def and "parameters" in result_data:
        _post_process_parameters(result_data, fn_def)

    return FunctionCallResult.model_validate(result_data)


def decode_prompt(
    model,
    prompt_text: str,
    decoder_input: DecoderInput,
    vocab: dict[int, str],
    verbose: bool = False,
    temperature: float = 1.0,
) -> FunctionCallResult | None:
    init_prompt = create_initial_prompt(prompt_text, decoder_input)
    allowed_names = [f.name for f in decoder_input.functions]
    func_lookup = decoder_input.function_by_name

    selected_name, updated_ids = run_dynamic_routing(
        model, init_prompt, allowed_names, vocab, verbose=verbose
    )
    generated_raw = run_parameter_extraction(
        model, updated_ids, selected_name, func_lookup, vocab,
        verbose=verbose, temperature=temperature
    )
    return post_process_result(generated_raw, selected_name, func_lookup)
