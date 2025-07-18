import litellm
import time

def create_api_call(
        model,
        parallel_tool_calls,
        tool_choice,
        temperature,
        max_tokens,
        additional_drop_params,
        delay
):
    """Factory for creating the api_call function."""
    return lambda messages, tools: (
        time.sleep(delay),  # Delay before each call
        litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_drop_params=list(additional_drop_params)
        ))[1]