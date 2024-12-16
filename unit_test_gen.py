import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
import torch
import os

# Load configuration from a PTFE config file
def load_config(config_path="config.json"):
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as file:
        return json.load(file)

# Load the model with LoRA and bitsandbytes quantization
def load_llm(config):
    """
    Load a large language model with LoRA fine-tuning and bitsandbytes quantization.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Load model with 4-bit quantization using bitsandbytes
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        device_map="auto",
        quantization_config=bnb.nn.quantization.QuantizationConfig(bits=4),
        torch_dtype=torch.float16
    )

    # Apply LoRA fine-tuning if enabled
    if config.get("use_lora", False):
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            target_modules=config["target_modules"],
            lora_dropout=config["lora_dropout"],
            bias=config["bias"],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    # Set up the HuggingFace pipeline
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=generator)

# Main script
if __name__ == "__main__":
    # Load configuration
    config = load_config("config.json")

    # Initialize the LLM with config
    llm = load_llm(config)

    # Define a prompt template for unit test generation
    prompt_template = PromptTemplate(
        input_variables=["function_code"],
        template=(
            "You are an AI assistant that generates Python unit tests for the given function."
            " The tests should have inline documentation explaining each test case clearly."
            "\n\nFunction:\n" 
            "{function_code}\n\n"
            "Please provide the complete unit test code with inline comments."
        ),
    )

    # Define a LangChain LLM chain for generating unit tests
    unit_test_chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )

    # Function to generate unit tests
    def generate_unit_tests(function_code: str) -> str:
        """
        Generates unit tests with inline documentation for a given Python function.

        Args:
            function_code (str): The Python function code to generate tests for.

        Returns:
            str: Generated unit test code.
        """
        unit_tests = unit_test_chain.run(function_code=function_code)
        return unit_tests

    # Example function provided by the user
    user_function = """
    def add(a, b):
        """Returns the sum of two numbers."""
        return a + b

    def subtract(a, b):
        """Returns the difference of two numbers."""
        return a - b
    """

    # Generate unit tests
    generated_tests = generate_unit_tests(user_function)

    # Save the generated tests to a file
    output_file = config["output_file"]
    with open(output_file, "w") as f:
        f.write(generated_tests)

    print(f"Generated unit tests saved to {output_file}.")

    # Print the generated tests
    print("\nGenerated Unit Tests:\n")
    print(generated_tests)
