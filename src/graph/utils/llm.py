import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


def _llm_env_var(key: str, model_prefix: str = None):
    """
    Get the value of the specified environment variable.

    Args:
        key (str): The name of the environment variable.
        model_prefix (str, optional): The prefix to be added to the environment variable name. Defaults to None.

    Returns:
        str: The value of the environment variable.

    Raises:
        ValueError: If the environment variable is not set.
    """
    if model_prefix:
        env_key = f"{model_prefix}_{key}"
    else:
        env_key = key

    value = os.environ.get(env_key)
    if value is None:
        raise ValueError(f"Environment variable '{env_key}' is not set")

    return value

def azure_llm(model_prefix: str = None, **kwargs):
    """
    Create an instance of AzureChatOpenAI for language model (LLM) tasks.

    Args:
        model_prefix (str, optional): The prefix to be added to the environment variable names. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to AzureChatOpenAI.

    Returns:
        AzureChatOpenAI: An instance of AzureChatOpenAI for LLM tasks.
    """
    if model_prefix:
        model_prefix = model_prefix.upper()
    else:
        model_prefix = os.getenv("DEFAULT_MODEL_PREFIX", "GPT4")

    if "verbose" in kwargs:
        verbose = kwargs.pop("verbose")
    else:
        verbose = os.environ.get("VERBOSE", "0") == "1"

    # o1 and o3 models require specific API version
    api_version = _llm_env_var("AZURE_OPENAI_API_VERSION", model_prefix)

    # Check if this is a reasoning model
    is_reasoning_model = model_prefix in ["O3_MINI", "O4_MINI", "O3", "O4"]
    
    if is_reasoning_model:
        # O3-mini and O4-mini use standard chat completions API, not Responses API
        # They just don't support certain parameters
        unsupported_params = ['temperature', 'top_p', 'presence_penalty', 'frequency_penalty', 
                            'logprobs', 'top_logprobs', 'logit_bias']
        for param in unsupported_params:
            kwargs.pop(param, None)
        
        # Ensure max_completion_tokens is set for reasoning models
        if 'max_completion_tokens' not in kwargs:
            kwargs['max_completion_tokens'] = 100000

    return AzureChatOpenAI(
        openai_api_key=_llm_env_var("AZURE_OPENAI_API_KEY", model_prefix),
        openai_api_version=api_version,
        deployment_name=_llm_env_var("AZURE_DEPLOYMENT_NAME", model_prefix),
        azure_endpoint=_llm_env_var("AZURE_OPENAI_ENDPOINT", model_prefix),
        model_version=_llm_env_var("AZURE_OPENAI_MODEL_NAME", model_prefix),
        model=_llm_env_var("AZURE_OPENAI_MODEL_NAME", model_prefix),
        verbose=verbose,
        **kwargs,
    )


def azure_embeddings(**kwargs):
    """
    Create an instance of AzureOpenAIEmbeddings for embeddings tasks.

    Args:
        **kwargs: Additional keyword arguments to be passed to AzureOpenAIEmbeddings.

    Returns:
        AzureOpenAIEmbeddings: An instance of AzureOpenAIEmbeddings for embeddings tasks.
    """
    return AzureOpenAIEmbeddings(
        openai_api_key=_llm_env_var("AZURE_OPENAI_API_KEY", "EMBEDDINGS"),
        openai_api_version=_llm_env_var("AZURE_OPENAI_API_VERSION", "EMBEDDINGS"),
        azure_endpoint=_llm_env_var("AZURE_OPENAI_ENDPOINT", "EMBEDDINGS"),
        deployment=_llm_env_var("AZURE_OPENAI_DEPLOYMENT", "EMBEDDINGS"),
        **kwargs,
    )