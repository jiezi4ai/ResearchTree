import os
import time
import numpy as np
import asyncio 
import PIL.Image 
from typing import List

# Import Google AI specific libraries
from google import genai
from google.genai import types 


# --- Original Synchronous Functions (preserved) ---
def llm_gen(api_key, model_name, qa_prompt, sys_prompt=None, temperature=0.3):
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=sys_prompt,
        temperature=temperature)
    try:
        response = client.models.generate_content(
            model=model_name, 
            contents=qa_prompt,
            config=config)
        return response.text
    except Exception as e:
        print(f"Error during synchronous llm_gen: {e}")
        raise e 


def llm_image_gen(api_key, model_name, qa_prompt, pil_images: List[PIL.Image.Image], sys_prompt=None, temperature=0.3):
    """q&a with images (synchronous)
    Args:
        pil_images: List of PIL.Image objects.
    """
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=sys_prompt,
        temperature=temperature)
    try:
        response = client.models.generate_content(
            model=model_name,  #　"gemini-2.0-flash-exp",
            contents=[qa_prompt]+pil_images,
            config=config)
        # Basic safety check
        return response.text
    except Exception as e:
        print(f"Error during synchronous llm_image_gen: {e}")
        raise e # Re-raise

def llm_gen_w_retry(api_key, model_name, qa_prompt, sys_prompt=None, temperature=0.3, max_retries=3, initial_delay=1):
    """ Synchronous llm_gen with retry logic """
    retries = 0
    delay = initial_delay

    while retries <= max_retries:
        try:
            # Call the base synchronous function
            return llm_gen(api_key, model_name, qa_prompt, sys_prompt, temperature)
        # Handle potential direct ValueErrors from safety checks etc. in llm_gen
        except ValueError as e:
            print(f"Sync: Non-retryable error during generation: {e}")
            return None # Return None for non-retryable errors
        except Exception as e: # Catch other potential exceptions from the API call
            # Attempt to check status code if available, otherwise treat as non-retryable
            status_code = getattr(e, 'code', None) or getattr(e, 'status_code', None) # Check common attributes
            if status_code == 429: # Check if it looks like a rate limit error anyway
                 if retries < max_retries:
                     retries += 1
                     print(f"Sync: Rate limit inferred ({status_code}). Retrying in {delay} seconds (Retry {retries}/{max_retries})...")
                     time.sleep(delay)
                     delay *= 2
                 else:
                     print(f"Sync: Max retries reached after inferred rate limit error. Returning None.")
                     return None
            else:
                 print(f"Sync: An unexpected error occurred: {e} (Code: {status_code})")
                 return None 

    return None


def llm_image_gen_w_retry(api_key, model_name, qa_prompt, pil_images: List[PIL.Image.Image], sys_prompt=None, temperature=0.3, max_retries=3, initial_delay=1):
    """ Synchronous llm_image_gen with retry logic """
    retries = 0
    delay = initial_delay

    while retries <= max_retries:
        try:
            # Call the base synchronous function
            return llm_image_gen(api_key, model_name, qa_prompt, pil_images, sys_prompt, temperature)
        # Handle potential direct ValueErrors from safety checks etc. in llm_image_gen
        except ValueError as e:
            print(f"Sync Img: Non-retryable error during generation: {e}")
            return None # Return None for non-retryable errors
        except Exception as e: # Catch other potential exceptions
            status_code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
            if status_code == 429:
                 if retries < max_retries:
                     retries += 1
                     print(f"Sync Img: Rate limit inferred ({status_code}). Retrying in {delay} seconds (Retry {retries}/{max_retries})...")
                     time.sleep(delay)
                     delay *= 2
                 else:
                     print(f"Sync Img: Max retries reached after inferred rate limit error. Returning None.")
                     return None
            else:
                print(f"Sync Img: An unexpected error occurred: {e} (Code: {status_code})")
                return None # Return None for other errors

    return None


# --- New Asynchronous Functions ---

async def async_llm_gen(api_key, model_name, qa_prompt, sys_prompt=None, temperature=0.3):
    """Asynchronous version of llm_gen."""
    # Configuration should ideally happen once globally or be managed externally
    client = genai.Client(api_key=api_key)
    # config = types.GenerateContentConfig(
    #     system_instruction=sys_prompt,
    #     temperature=temperature)
    try:
        response = await client.aio.models.generate_content(
            model=model_name, 
            contents=qa_prompt,
            #config=config
            )
        return response.text
    except Exception as e:
        print(f"Error during async async_llm_gen: {e}")
        # Re-raise for retry logic to catch
        raise e

async def async_llm_image_gen(api_key, model_name, qa_prompt, pil_images: List[PIL.Image.Image], sys_prompt=None, temperature=0.3):
    """Asynchronous version of llm_image_gen."""
    client = genai.Client(api_key=api_key)
    # config = types.GenerateContentConfig(
    #     system_instruction=sys_prompt,
    #     temperature=temperature)
    try:
        response = await client.aio.models.generate_content(
            model=model_name,  #　"gemini-2.0-flash-exp",
            contents=[qa_prompt]+pil_images,
            #config=config
            )
        return response.text
    except Exception as e:
        print(f"Error during async async_llm_image_gen: {e}")
        raise e


async def async_llm_gen_w_retry(api_key, model_name, qa_prompt, sys_prompt=None, temperature=0.3, max_retries=3, initial_delay=1):
    """ Asynchronous llm_gen with retry logic """
    retries = 0
    delay = initial_delay

    while retries <= max_retries:
        try:
            # Call the base asynchronous function
            return await async_llm_gen(api_key, model_name, qa_prompt, sys_prompt, temperature)
        except ValueError as e:
            print(f"Async: Non-retryable error during generation: {e}")
            return None # Return None for non-retryable errors
        except Exception as e: # Catch other potential exceptions
            status_code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
            if status_code == 429: # Check if it looks like a rate limit error anyway
                 if retries < max_retries:
                     retries += 1
                     print(f"Async: Rate limit inferred ({status_code}). Retrying in {delay} seconds (Retry {retries}/{max_retries})...")
                     await asyncio.sleep(delay) # Use asyncio.sleep
                     delay *= 2
                 else:
                     print(f"Async: Max retries reached after inferred rate limit error. Returning None.")
                     return None
            else:
                print(f"Async: An unexpected error occurred: {e} (Code: {status_code})")
                return None # Return None for other errors

    return None


async def async_llm_image_gen_w_retry(api_key, model_name, qa_prompt, pil_images: List[PIL.Image.Image], sys_prompt=None, temperature=0.3, max_retries=3, initial_delay=1):
    """ Asynchronous llm_image_gen with retry logic """
    retries = 0
    delay = initial_delay

    while retries <= max_retries:
        try:
            # Call the base asynchronous function
            return await async_llm_image_gen(api_key, model_name, qa_prompt, pil_images, sys_prompt, temperature)
        # Handle potential direct ValueErrors from safety checks etc. in async_llm_image_gen
        except ValueError as e:
            print(f"Async Img: Non-retryable error during generation: {e}")
            return None # Return None for non-retryable errors
        except Exception as e: # Catch other potential exceptions
            status_code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
            if status_code == 429:
                 if retries < max_retries:
                     retries += 1
                     print(f"Async Img: Rate limit inferred ({status_code}). Retrying in {delay} seconds (Retry {retries}/{max_retries})...")
                     await asyncio.sleep(delay) # Use asyncio.sleep
                     delay *= 2
                 else:
                     print(f"Async Img: Max retries reached after inferred rate limit error. Returning None.")
                     return None
            else:
                print(f"Async Img: An unexpected error occurred: {e} (Code: {status_code})")
                return None # Return None for other errors

    return None

# # --- Example Usage (Optional) ---
# async def example_async_call():
#     API_KEY = "YOUR_GOOGLE_API_KEY" # Replace with your key
#     if API_KEY == "YOUR_GOOGLE_API_KEY":
#        print("Please replace YOUR_GOOGLE_API_KEY with your actual key.")
#        return

#     # Configure the API key once (recommended practice)
#     genai.configure(api_key=API_KEY)

#     prompt = "Explain asynchronous programming in Python."
#     model = "gemini-pro" # Or other suitable model like "gemini-1.5-flash"

#     print("--- Testing async_llm_gen_w_retry ---")
#     try:
#         # Use the retry wrapper
#         response_text = await async_llm_gen_w_retry(
#             api_key=API_KEY, # Pass key for consistency, though configure is main driver
#             model_name=model,
#             qa_prompt=prompt,
#             temperature=0.5,
#             max_retries=2,
#             initial_delay=2
#         )

#         if response_text:
#             print("\nLLM Response:")
#             print(response_text[:500] + "...") # Print partial response
#         else:
#             print("\nFailed to get LLM response after retries.")

#     except Exception as e:
#         print(f"\nAn error occurred during the example call: {e}")

# To run the example:
# if __name__ == "__main__":
#    asyncio.run(example_async_call())