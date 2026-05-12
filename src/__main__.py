import argparse
import json
from pathlib import Path

from llm_sdk.llm_sdk import Small_LLM_Model

from src.builder import load_decoder_input
from src.decoder import decode_prompt, get_public_vocab


def main():
    parser = argparse.ArgumentParser(description="Call Me Maybe — constrained LLM function-calling engine")
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    parser.add_argument("--verbose", action="store_true", help="Print token-by-token generation progress")
    parser.add_argument("--temperature", type=float, default=1.5, help="Sampling temperature for parameter extraction (default: 1.5)")
    args = parser.parse_args()

    try:
        model = Small_LLM_Model("Qwen/Qwen3-0.6B", device="cpu")
        decoder_input = load_decoder_input(
            Path(args.functions_definition),
            Path(args.input),
        )
        vocab = get_public_vocab(model)

        final_outputs = []
        for prompt_item in decoder_input.prompts:
            try:
                print(f"Processing: {prompt_item.prompt}")
                result = decode_prompt(model, prompt_item.prompt, decoder_input, vocab, verbose=args.verbose, temperature=args.temperature)
                if result:
                    final_outputs.append(result.model_dump())
                    print("Done.")
            except Exception as e:
                print(f"Error: {e}")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(final_outputs, f, indent=4)
        print("\nAll prompts processed successfully.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
