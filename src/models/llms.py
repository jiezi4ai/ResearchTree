import time
import ollama
import numpy as np

import os
from google import genai
from google.genai import types

def llm_gen(api_key, model_name, qa_prompt, sys_prompt=None, temperature=0.3):
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=sys_prompt,
        temperature=temperature)
    response = client.models.generate_content(
        model=model_name, 
        contents=qa_prompt,
        config=config)
    return response.text

def llm_image_gen(api_key, model_name, qa_prompt, pil_images, sys_prompt=None, temperature=0.3):
    """q&a with images
    Args:
        pil_images:
            import PIL.Image
            image = PIL.Image.open('/path/to/image.png')
    """

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=sys_prompt,
        temperature=temperature)

    response = client.models.generate_content(
        model=model_name,  #　"gemini-2.0-flash-exp",
        contents=[qa_prompt]+pil_images,
        config=config)

    return response.text

def llm_gen_w_retry(api_key, model_name, qa_prompt, sys_prompt=None, temperature=0.3, max_retries=3, initial_delay=1):
    """
    Wraps the llm_gen_w_images function to enable retries on RESOURCE_EXHAUSTED errors.

    Args:
        api_key: API key for the LLM service.
        model_name: Name of the LLM model to use.
        qa_prompt: Question and answer prompt for the LLM.
        pil_images: List of PIL.Image objects.
        temperature: Temperature for LLM response generation.
        max_retries: Maximum number of retries in case of error.
        initial_delay: Initial delay in seconds before the first retry.

    Returns:
        str: The text response from the LLM, or None if max retries are exceeded and still error.
    """
    retries = 0
    delay = initial_delay

    while retries <= max_retries:
        try:
            return llm_gen(api_key, model_name, qa_prompt, sys_prompt, temperature)
        except Exception as e:
            if e.code == 429:
                if retries < max_retries:
                    retries += 1
                    print(f"Rate limit exceeded. Retrying in {delay} seconds (Retry {retries}/{max_retries})...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff for delay
                else:
                    print(f"Max retries reached.  Raising the last exception.")
                    return None # raise  # Re-raise the last exception if max retries are exhausted
            else:
                print(f"Error Code: {e.code} Error Message: {e.message}")
                return None
                # raise  # Re-raise other ClientErrors (not related to resource exhaustion)

    return None # Should not reach here in normal cases as exception is re-raised or value is returned in try block

def llm_image_gen_w_retry(api_key, model_name, qa_prompt, pil_images, sys_prompt=None, temperature=0.3, max_retries=3, initial_delay=1):
    """
    Wraps the llm_gen_w_images function to enable retries on RESOURCE_EXHAUSTED errors.

    Args:
        api_key: API key for the LLM service.
        model_name: Name of the LLM model to use.
        qa_prompt: Question and answer prompt for the LLM.
        pil_images: List of PIL.Image objects.
        sys_prompt: Optional system prompt for the LLM.
        temperature: Temperature for LLM response generation.
        max_retries: Maximum number of retries in case of error.
        initial_delay: Initial delay in seconds before the first retry.

    Returns:
        str: The text response from the LLM, or None if max retries are exceeded and still error.
    """
    retries = 0
    delay = initial_delay

    while retries <= max_retries:
        try:
            return llm_image_gen(api_key, model_name, qa_prompt, pil_images, sys_prompt, temperature)
        except Exception as e:
            if e.code == 429:
                if retries < max_retries:
                    retries += 1
                    print(f"Rate limit exceeded. Retrying in {delay} seconds (Retry {retries}/{max_retries})...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff for delay
                else:
                    print(f"Max retries reached.  Raising the last exception.")
                    return None # raise  # Re-raise the last exception if max retries are exhausted
            else:
                print(f"Error Code: {e.code} Error Message: {e.message}")
                return None
                # raise  # Re-raise other ClientErrors (not related to resource exhaustion)

    return None # Should not reach here in normal cases as exception is re-raised or value is returned in try block