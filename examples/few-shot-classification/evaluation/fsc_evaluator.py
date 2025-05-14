import sys
sys.path.append('..')

from typing import Optional, List
import numpy as np
import torch
from transformers import (AutoTokenizer,
                          AutoModelForMaskedLM)

SUPPORTED_MASK_LMS = ['distilroberta-base', 'roberta-base', 'roberta-large', 'indolem/indobert-base-uncased']

class PromptedClassificationEvaluator:
    def __init__(
        self,
        task_lm: str,
        is_mask_lm: Optional[bool],
        num_classes: int,
        verbalizers: List[str],
        template: Optional[str],
        prompt: str
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.task_lm = task_lm
        print("Task LM:", self.task_lm)

        if is_mask_lm is None: 
            self.is_mask_lm = True if 'bert' in self.task_lm else False
        else:
            self.is_mask_lm = is_mask_lm

        if self.is_mask_lm:
            assert self.task_lm in SUPPORTED_MASK_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm)
            self._generator = (AutoModelForMaskedLM
                               .from_pretrained(self.task_lm)
                               .to(self.device))
        else:
            print("Error: Only BERT-based models are supported for now.")
            raise NotImplementedError(
                "Only BERT-based models are supported for now. "
                "Please use a different model or set is_mask_lm to True."
            )

        self.num_classes = num_classes
        self.verbalizers = verbalizers

        self.verbalizer_ids = [self._tokenizer.convert_tokens_to_ids(v)
                               for v in self.verbalizers]
        if template is None:
            self.template = self.load_default_template()
        else:
            self.template = template

        self.prompt = prompt

    def _get_mask_token_index(self, input_ids: torch.Tensor) -> np.ndarray:
        mask_token_index = torch.where(
            input_ids == self._tokenizer.mask_token_id)[1]
        return mask_token_index

    def load_default_template(self) -> List[str]:
        mask_token = self._tokenizer.mask_token
        template = f"{{sentence_1}} {{prompt}} {mask_token} ."

        return template

    @torch.no_grad()
    def _get_logits(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        batch_size = len(texts)
        encoded_inputs = self._tokenizer(texts, padding='longest', max_length=512,
                                         truncation=True, return_tensors="pt",
                                         add_special_tokens=True)

        token_logits = self._generator(**encoded_inputs.to(self.device)).logits
        mask_token_indices = self._get_mask_token_index(encoded_inputs['input_ids'])
        out_logits = token_logits[range(batch_size), mask_token_indices, :]

        return out_logits

    def _format_prompts(
        self,
        prompts: List[str],
        source_strs: List[str]
    ) -> List[str]:
        return [self.template.format(sentence_1=s_1, prompt=prompt)
                for s_1, prompt in zip(source_strs, prompts)]

    def forward(
        self,
        dataloader
    ) -> float:
        num_of_examples = dataloader.dataset.__len__()
        correct_sum = 0
        for _, batch in enumerate(dataloader):
            inputs = batch['source_texts']
            targets = batch['class_labels']
            batch_size = targets.size(0)
            current_prompts = [self.prompt for _ in range(batch_size)]
            formatted_templates = self._format_prompts(current_prompts, inputs)
            all_logits = self._get_logits(formatted_templates)
            class_probs = torch.softmax(all_logits[:, self.verbalizer_ids], -1)
            
            predicted_labels = torch.argmax(class_probs, dim=-1)
            label_agreement = torch.where(
                targets.cuda() == predicted_labels, 1, 0)
            correct_sum += label_agreement.sum()
        accuracy = correct_sum/num_of_examples
        return accuracy
