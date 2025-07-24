from typing import List, Optional
import edc.utils.llm_utils as llm_utils
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging

logger = logging.getLogger(__name__)

class Extractor:
    # The class to handle the first stage: Open Information Extraction
    def __init__(self, model: Optional[AutoModelForCausalLM] = None, tokenizer: Optional[AutoTokenizer] = None, openai_model=None) -> None:
        assert openai_model is not None or (model is not None and tokenizer is not None)
        self.model = model
        self.tokenizer = tokenizer
        self.openai_model = openai_model

    def extract(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        prompt_template_str: str,
        entities_hint: Optional[str] = None,
        relations_hint: Optional[str] = None,
    ) -> List[List[str]]:
        assert (entities_hint is None and relations_hint is None) or (relations_hint is not None and relations_hint is not None)

        filled_prompt = prompt_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
                "input_text": input_text_str,
                "entities_hint": entities_hint,
                "relations_hint": relations_hint,
            }
        )

        messages = [{"role": "user", "content": filled_prompt}]

        logger.info(f'{messages=}')

        if self.openai_model is None:
            # llm_utils.generate_completion_transformers([messages], self.model, self.tokenizer, device=self.device)
            completion = llm_utils.generate_completion_transformers(
                messages, self.model, self.tokenizer, answer_prepend="Triplets: "
            )
        else:
            completion = llm_utils.openai_chat_completion(self.openai_model, None, messages)
            logger.info(completion)

        extracted_triplets_list = llm_utils.parse_raw_triplets(str(completion))
        return extracted_triplets_list
