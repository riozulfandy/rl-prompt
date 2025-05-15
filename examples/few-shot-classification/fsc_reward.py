import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, GPT2LMHeadModel
from typing import List, Dict, Optional, Tuple, Union, Any
from collections import defaultdict
from rlprompt.rewards import BaseReward

SUPPORTED_MASK_LMS = ['distilroberta-base', 'roberta-base', 'roberta-large', 'indolem/indobert-base-uncased']

class PromptedClassificationReward(BaseReward):
    def __init__(
        self,
        task_lm: str,
        is_mask_lm: Optional[bool],
        compute_zscore: bool,
        incorrect_coeff: float, 
        correct_coeff: float,
        num_classes: int,
        verbalizers: List[str],
        template: Optional[str]
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        
        self.task_lm = task_lm
        if is_mask_lm is None: 
            self.is_mask_lm = True if 'bert' in self.task_lm else False
        else:
            raise NotImplementedError(
                "Only BERT-based models are supported for now. "
                "Please use a different model or set is_mask_lm to True."
            )

        print('Task LM:', self.task_lm)
        if self.is_mask_lm:
            assert self.task_lm in SUPPORTED_MASK_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm, truncate_side='left')
            self._generator = (AutoModelForMaskedLM
                               .from_pretrained(self.task_lm)
                               .to(self.device))

        self.compute_zscore = compute_zscore
        self.incorrect_coeff = incorrect_coeff
        self.correct_coeff = correct_coeff
        self.num_classes = num_classes
        self.verbalizers = verbalizers
        print('Verbalizers:', self.verbalizers)

        self.verbalizer_ids = [self._tokenizer.convert_tokens_to_ids(v)
                               for v in self.verbalizers]
        
        if template is None:
            self.template = self.load_default_template()
        else: 
            self.template = template

        self._counter = 0

    def load_default_template(self) -> str:
        mask_token = self._tokenizer.mask_token
        template = f"{{sentence_1}} {{prompt}} {mask_token} ."

        return template

    def forward(
        self,
        source_texts: List[str],
        class_labels: List[int],
        output_tokens: List[List[str]],
        to_tensor: bool,
        mode: str
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
        assert mode in ["train", "infer"]
        
        if mode == "train":
            self._counter += 1

        # Process prompts and verbalizer indices
        prompt_tokens = output_tokens
        prompt_strings = self._convert_tokens_to_string(prompt_tokens)
        batch_size = len(source_texts)

        rewards: List[torch.Tensor] = []
        input_rewards: Dict[str, List[float]] = defaultdict(list)
        quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for i, prompt in enumerate(prompt_strings):
            current_prompts = [prompt for _ in source_texts]
            formatted_templates = self._format_prompts(source_texts,
                                                       current_prompts)
            all_logits = self._get_logits(formatted_templates)
            
            class_probs = torch.softmax(all_logits[:, self.verbalizer_ids], -1)
            label_probs = class_probs[range(batch_size), class_labels]
            
            not_label_probs = torch.where(
                class_probs == label_probs.unsqueeze(1),
                torch.Tensor([-1]).to(self.device), class_probs)
            
            max_not_label_probs, _ = torch.max(not_label_probs, -1)

            # Compute piecewise gap reward
            gap = (label_probs - max_not_label_probs)
            correct = (gap > 0).long()
            gap_rewards = gap * (self.correct_coeff * correct + self.incorrect_coeff * (1 - correct))
            reward = gap_rewards.mean().detach()

            # Log quantities such as accuracy and class-wise reward
            acc = correct.float().mean()
            quantities_to_log['acc'] = acc

            # Calculate F1 score
            predicted_labels = torch.argmax(class_probs, dim=-1)
            class_labels_tensor = torch.tensor(class_labels, device=predicted_labels.device, dtype=torch.long)
            
            current_prompt_f1_scores = []
            for c_idx in range(self.num_classes):
                tp = ((predicted_labels == c_idx) & (class_labels_tensor == c_idx)).sum().float()
                fp = ((predicted_labels == c_idx) & (class_labels_tensor != c_idx)).sum().float()
                fn = ((predicted_labels != c_idx) & (class_labels_tensor == c_idx)).sum().float()

                precision = tp / (tp + fp + 1e-8)  # Add epsilon to avoid division by zero
                recall = tp / (tp + fn + 1e-8)     # Add epsilon to avoid division by zero
                
                f1_class = 2 * (precision * recall) / (precision + recall + 1e-8)
                quantities_to_log[f"f1_class_{c_idx}"].append(f1_class.item())
                current_prompt_f1_scores.append(f1_class)
            
            if current_prompt_f1_scores: # Ensure list is not empty
                macro_f1 = torch.stack(current_prompt_f1_scores).mean().item()
                quantities_to_log['macro_f1'].append(macro_f1)
            else: # Handle case with no classes or empty batch for F1
                quantities_to_log['macro_f1'].append(0.0)

            for c in range(self.num_classes):
                class_idx = np.array(class_labels) == c
                class_rewards = gap_rewards[class_idx]
                quantities_to_log[f"gap_reward_class_{c}"].append(class_rewards.mean().item())
                
            quantities_to_log['gap_reward'].append(reward.item())
            rewards.append(reward)

            input_rewards['z'] += [reward.item()]

            print_strs = [self._counter, '|', prompt, '\n']
            for c in range(self.num_classes):
                class_example_idx = np.where(np.array(class_labels) == c)[0][0]
                class_example = formatted_templates[class_example_idx]
                class_example_probs = class_probs[class_example_idx, :].tolist()
                class_example_probs = [round(prob, 2) \
                                       for prob in class_example_probs]
                print_strs += ['Class', c, 'Example:', 
                               class_example, '|',
                               'Probs:', class_example_probs, '\n']
            print_strs += ['Accuracy:', acc.item(), '|',
                           'Reward:', round(reward.item(), 2), "|",
                           'Macro F1:', round(macro_f1, 2), '\n']
            print(*print_strs)
        rewards_tensor = torch.stack(rewards)

        if mode == 'train' and self.compute_zscore:
            input_reward_means = {k: np.mean(v)
                                  for k, v in input_rewards.items()}
            input_reward_stds = {k: np.std(v)
                                 for k, v in input_rewards.items()}
            
            idx_means = torch.tensor(input_reward_means['z']).float()
            idx_stds = torch.tensor(input_reward_stds['z']).float()
            rewards_tensor = (rewards_tensor - idx_means)/(idx_stds + 1e-4)
            for i in range(rewards_tensor.size(0)):
                quantities_to_log['resized_reward'].append(
                    rewards_tensor[i].item())
                
        elif mode == 'infer': 
            score = rewards_tensor.mean().item()
            print('Our Prompt:')
            print(prompt_strings, score)

        rewards_log = dict(
            (reward_key, torch.mean(torch.tensor(reward_vals)))
            for reward_key, reward_vals in quantities_to_log.items())

        if to_tensor is True:
            return rewards_tensor, rewards_log
        else:
            return rewards_tensor.tolist(), rewards_log

    def _get_mask_token_index(self, input_ids: torch.Tensor) -> np.ndarray:
        mask_token_index = torch.where(
            input_ids == self._tokenizer.mask_token_id)[1]
        return mask_token_index

    def ensure_exactly_one_mask_token(
        self,
        model_inputs: Dict[str, torch.Tensor]
    ) -> None:
        for input_ids in model_inputs["input_ids"]:
            masked_index = self._get_mask_token_index(input_ids)
            numel = np.prod(masked_index.shape)
            assert numel == 1

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

    def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
        return [self._tokenizer.convert_tokens_to_string(s)
                for s in tokens]

    def _format_prompts(
        self,
        source_strs: List[str],
        prompt_strs: List[str],
    ) -> List[str]:
        return [self.template.format(sentence_1=s_1, prompt=p)
                for s_1, p in zip(source_strs, prompt_strs)]
