import os
from openai import OpenAI
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import ast
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import gc
import torch
import logging
import ollama

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def encode(self, text: str | List[str], prompt_name: str | None = None, prompt: str | None = None) -> List[float] | List[List[float]]:
        """
        Generate embeddings for text using Ollama.
        
        Args:
            text: Single string or list of strings to embed
            prompt_name: Name of the prompt template (will warn if used - not supported by Ollama)
            prompt: Additional prompt context to prepend to text
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        # # Warn if prompt_name is used since Ollama doesn't support it
        # if prompt_name is not None:
        #     import warnings
        #     warnings.warn(
        #         f"prompt_name='{prompt_name}' is not supported by Ollama embeddings API and will be ignored. "
        #         "Consider using the 'prompt' parameter instead.",
        #         UserWarning
        #     )
        
        def _prepare_text(input_text: str) -> str:
            """Prepare text by prepending prompt if provided."""
            if prompt:
                return f"{prompt} {input_text}"
            return input_text
        
        try:
            if isinstance(text, str):
                prepared_text = _prepare_text(text)
                response = ollama.embeddings(model=self.model_name, prompt=prepared_text)
                return response['embedding']
            elif isinstance(text, list):
                embeddings = []
                for t in text:
                    prepared_text = _prepare_text(t)
                    response = ollama.embeddings(model=self.model_name, prompt=prepared_text)
                    embeddings.append(response['embedding'])
                return embeddings
            else:
                raise ValueError("Text must be a string or list of strings")
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings with {self.model_name}: {str(e)}")



def free_model(model: Optional[AutoModelForCausalLM] = None, tokenizer: Optional[AutoTokenizer] = None):
    """
    delete model (not really necessary)
    """
    try:
        model.cpu()
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(e)


def get_embedding_e5mistral(model, tokenizer, sentence, task=None):
    model.eval()
    device = model.device

    if task is not None:
        # It's a query to be embed
        sentence = get_detailed_instruct(task, sentence)

    sentence = [sentence]

    max_length = 4096
    # Tokenize the input texts
    batch_dict = tokenizer(
        sentence, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True
    )
    # append eos_token_id to every input_ids
    batch_dict["input_ids"] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")

    batch_dict.to(device)

    embeddings = model(**batch_dict).detach().cpu()

    assert len(embeddings) == 1

    return embeddings[0]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def get_embedding_sts(model: SentenceTransformer, text: str, prompt_name=None, prompt=None):
    embedding = model.encode(text, prompt_name=prompt_name, prompt=prompt)
    return embedding


def parse_raw_entities(raw_entities: str):
    parsed_entities = []
    left_bracket_idx = raw_entities.index("[")
    right_bracket_idx = raw_entities.index("]")
    try:
        parsed_entities = ast.literal_eval(raw_entities[left_bracket_idx : right_bracket_idx + 1])
    except Exception as e:
        pass
    logging.debug(f"Entities {raw_entities} parsed as {parsed_entities}")
    return parsed_entities


def parse_raw_triplets(raw_triplets: str):
    # Look for enclosing brackets
    unmatched_left_bracket_indices = []
    matched_bracket_pairs = []

    collected_triples = []
    for c_idx, c in enumerate(raw_triplets):
        if c == "[":
            unmatched_left_bracket_indices.append(c_idx)
        if c == "]":
            if len(unmatched_left_bracket_indices) == 0:
                continue
            # Found a right bracket, match to the last found left bracket
            matched_left_bracket_idx = unmatched_left_bracket_indices.pop()
            matched_bracket_pairs.append((matched_left_bracket_idx, c_idx))
    for l, r in matched_bracket_pairs:
        bracketed_str = raw_triplets[l : r + 1]
        try:
            parsed_triple = ast.literal_eval(bracketed_str)
            if len(parsed_triple) == 3 and all([isinstance(t, str) for t in parsed_triple]):
                if all([e != "" and e != "_" for e in parsed_triple]):
                    collected_triples.append(parsed_triple)
            elif not all([type(x) == type(parsed_triple[0]) for x in parsed_triple]):
                for e_idx, e in enumerate(parsed_triple):
                    if isinstance(e, list):
                        parsed_triple[e_idx] = ", ".join(e)
                collected_triples.append(parsed_triple)
        except Exception as e:
            pass
    logger.debug(f"Triplets {raw_triplets} parsed as {collected_triples}")
    return collected_triples


def parse_relation_definition(raw_definitions: str):
    descriptions = raw_definitions.split("\n")
    relation_definition_dict = {}

    for description in descriptions:
        if ":" not in description:
            continue
        index_of_colon = description.index(":")
        relation = description[:index_of_colon].strip()

        relation_description = description[index_of_colon + 1 :].strip()

        if relation == "Answer":
            continue

        relation_definition_dict[relation] = relation_description
    logger.debug(f"Relation Definitions {raw_definitions} parsed as {relation_definition_dict}")
    return relation_definition_dict


def is_model_openai(model_name):
    return "gpt" in model_name or 'deepseek' in model_name


def generate_completion_transformers(
    input: list,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_token=256,
    answer_prepend="",
):
    device = model.device
    tokenizer.pad_token = tokenizer.eos_token

    messages = tokenizer.apply_chat_template(input, add_generation_prompt=True, tokenize=False) + answer_prepend

    model_inputs = tokenizer(messages, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    generation = model.generate(**model_inputs, generation_config=generation_config)
    sequences = generation["sequences"]
    generated_ids = sequences[:, model_inputs["input_ids"].shape[1] :]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    logging.debug(f"Prompt:\n {messages}\n Result: {generated_texts}")
    return generated_texts


def openai_chat_completion(model, system_prompt, history, temperature=0, max_tokens=512):
    base_url = None
    api_key = None

    if model.startswith('gpt'):
        base_url = None
        api_key = os.environ.get('OPENAI_API_KEY', None)
    else:
        base_url = 'https://api.deepseek.com'
        api_key = os.environ.get('DEEPSEEK_API_KEY', None)
        
    if api_key is None:
        api_key = input(f'Enter your {model} api key here')
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60,
    )

    response = None
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history
    while response is None:
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as e:
            logging.debug(f'{e=}')
            time.sleep(5)
    logging.debug(f"Model: {model}\nPrompt:\n {messages}\n Result: {response.choices[0].message.content}")
    return response.choices[0].message.content
