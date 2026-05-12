from pathlib import Path

from src.models import (
    DecoderFunction,
    DecoderInput,
    DecoderParameter,
    FunctionDefinition,
    ParameterSpec,
    PromptItem,
)
from src.loader import parse_function_definitions, parse_prompt_items


def _normalize_type(type_name: str) -> tuple[str, str]:
    mapping = {
        "string": ("string", "str"),
        "number": ("number", "float"),
        "integer": ("integer", "int"),
        "boolean": ("boolean", "bool"),
        "object": ("object", "dict"),
        "array": ("array", "list"),
    }
    if type_name not in mapping:
        raise ValueError(f"Unsupported parameter type: {type_name}")
    return mapping[type_name]


def _build_parameter(name: str, spec: ParameterSpec, index: int) -> DecoderParameter:
    json_type, python_type = _normalize_type(spec.type)
    return DecoderParameter(
        name=name,
        type_name=spec.type,
        json_type=json_type,
        python_type=python_type,
        order_index=index,
    )


def _build_schema_block(parameters: list[DecoderParameter]) -> str:
    if not parameters:
        return "    no parameters"
    return "\n".join(f"    {p.name}: {p.type_name}" for p in parameters)


def _build_routing_block(
    name: str, description: str, parameters: list[DecoderParameter], return_type: str
) -> str:
    return (
        f"Function name: {name}\n"
        f"Description: {description}\n"
        f"Parameters:\n{_build_schema_block(parameters)}\n"
        f"Returns: {return_type}"
    )


def _build_decoder_function(fn: FunctionDefinition) -> DecoderFunction:
    parameters = [
        _build_parameter(name, spec, i)
        for i, (name, spec) in enumerate(fn.parameters.items())
    ]
    return_type = fn.returns.type
    return DecoderFunction(
        name=fn.name,
        description=fn.description,
        parameters=parameters,
        return_type=return_type,
        routing_block=_build_routing_block(fn.name, fn.description, parameters, return_type),
        parameter_schema_block=_build_schema_block(parameters),
    )


def build_decoder_input(
    functions: list[FunctionDefinition], prompts: list[PromptItem]
) -> DecoderInput:
    decoder_functions = [_build_decoder_function(fn) for fn in functions]
    return DecoderInput(
        functions=decoder_functions,
        function_by_name={fn.name: fn for fn in decoder_functions},
        prompts=prompts,
    )


def load_decoder_input(functions_path: Path, prompts_path: Path) -> DecoderInput:
    functions = parse_function_definitions(functions_path)
    prompts = parse_prompt_items(prompts_path)
    return build_decoder_input(functions, prompts)
