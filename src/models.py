from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


SUPPORTED_TYPES = {"string", "number", "boolean", "integer", "object", "array"}


class PromptItem(BaseModel):
    """Single natural-language prompt entry from the input test file."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Reject empty or whitespace-only prompts."""
        if not value.strip():
            raise ValueError("Prompt must not be empty or whitespace only.")
        return value


class ParameterSpec(BaseModel):
    """Schema definition for a single function parameter."""

    model_config = ConfigDict(extra="forbid")

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Ensure the parameter type is supported."""
        if value not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported parameter type '{value}'. "
                f"Supported types: {sorted(SUPPORTED_TYPES)}"
            )
        return value


class ReturnSpec(BaseModel):
    """Schema definition for a function return value."""

    model_config = ConfigDict(extra="forbid")

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Ensure the return type is supported."""
        if value not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported return type '{value}'. "
                f"Supported types: {sorted(SUPPORTED_TYPES)}"
            )
        return value


class FunctionDefinition(BaseModel):
    """Definition of a callable function available to the LLM."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: dict[str, ParameterSpec]
    returns: ReturnSpec

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Reject empty or whitespace-only function names."""
        if not value.strip():
            raise ValueError(
                "Function name must not be empty or whitespace only."
            )
        return value

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        """Reject empty or whitespace-only descriptions."""
        if not value.strip():
            raise ValueError(
                "Function description must not be empty or whitespace only."
            )
        return value

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls,
        value: dict[str, ParameterSpec],
    ) -> dict[str, ParameterSpec]:
        """Reject empty parameter names."""
        for key in value:
            if not key.strip():
                raise ValueError(
                    "Parameter names must not be empty or whitespace only."
                )
        return value


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]


class DecoderParameter(BaseModel):
    """Normalized parameter metadata used by the decoder."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type_name: str
    json_type: str
    python_type: str
    order_index: int


class DecoderFunction(BaseModel):
    """Normalized function metadata used by the decoder."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: list[DecoderParameter] = Field(default_factory=list)
    return_type: str
    routing_block: str
    parameter_schema_block: str


class DecoderInput(BaseModel):
    """Complete decoder-ready input for the function calling pipeline."""

    model_config = ConfigDict(extra="forbid")

    functions: list[DecoderFunction]
    function_by_name: dict[str, DecoderFunction]
    prompts: list[PromptItem]