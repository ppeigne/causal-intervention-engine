import einops
import torch as t
from torch import Tensor

from jaxtyping import Float, Int, Array
from typing import List, Union, Optional, Callable, Tuple

from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

from dataclasses import dataclass, field

@dataclass
class CausalInterventionEngine():
    model: HookedTransformer
    prompts: List[str]
    answers: List[Tuple[str]]
    corrupted_prompts: Optional[List[str]] = None
    
    # Post init
    tokens: Int[Tensor, "batch seq_len"] = field(init=False)
    corrupted_tokens: Optional[Int[Tensor, "batch seq_len"]] = field(init=False)
    answers_token: Int[Tensor, "batch 2"] = field(init=False)

    answer_residual_directions: Float[Tensor, "batch 2 d_model"] = field(init=False)
    logit_diff_directions: Float[Tensor, "batch d_model"] = field(init=False)

    original_logits: Float[Tensor, "batch seq d_vocab"] = field(init=False)
    cache: ActivationCache = field(init=False)

    def __post_init__(self):
        # Generate tokens from prompts
        self.tokens = self.model.to_tokens(self.prompts, prepend_bos=True)
        self.corrupted_tokens = self.model.to_tokens(self.corrupted_prompts, prepend_bos=True) if self.corrupted_prompts else None
        self.answer_tokens = t.concat([
            self.model.to_tokens(names, prepend_bos=False).T for names in self.answers
        ])
        # Move the tokens to the GPU
        self.tokens = self.tokens.cuda() 
        self.original_logits, self.cache = self.model.run_with_cache(self.tokens)

        if self.corrupted_prompts:
            self.corrupted_tokens = self.corrupted_tokens.cuda() 
            self.corrupted_logits, self.corrupted_cache = self.model.run_with_cache(self.corrupted_tokens)

        answer_residual_directions = self.model.tokens_to_residual_directions(self.answer_tokens)
        
        self.correct_residual_directions, self.incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
        self.logit_diff_directions = self.correct_residual_directions - self.incorrect_residual_directions


    def logits_to_avg_logit_diff(
          self, 
          logits: Optional[Float[Tensor, "batch seq d_vocab"]] = None, 
          corrupted=False, 
          per_prompt=False
          ) -> Union[Float, Float[Tensor, "batch"]]:
        '''
        Returns logit difference between the correct and incorrect answer.

        If per_prompt=True, return the array of differences rather than the average.
        '''
        # Only the final logits are relevant for the answer
        final_logits: Float[t.Tensor, "batch d_vocab"]
        if type(logits) == Tensor:
            final_logits = logits[:, -1, :]
        elif corrupted:
            final_logits = self.corrupted_logits[:, -1, :]
        else:
            final_logits = self.original_logits[:, -1, :]
            
        # Get the logits corresponding to the indirect object / subject tokens respectively
        answer_logits: Float[t.Tensor, "batch 2"] = final_logits.gather(dim=-1, index=self.answer_tokens)
        # Find logit difference
        correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
        answer_logit_diff = correct_logits - incorrect_logits
        if per_prompt:
            return answer_logit_diff
        return answer_logit_diff.mean()


    def residual_stack_to_logit_diff(
        self,
        residual_stack: Float[Tensor, "... batch d_model"]
    ) -> Float[Tensor, "..."]:
        '''
        Gets the avg logit difference between the correct and incorrect
        answer for a given stack of components in the residual stream.
        '''
        scaled_residual_stack = self.cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
        return einops.einsum(
            scaled_residual_stack, self.logit_diff_directions, 
            "... batch d_model, batch d_model -> ..."
        ) / len(self.prompts)

    def accumulated_res_logit_diff(self) -> Tuple[Float[Tensor, "..."], List[str]]:
      acc_res, labels = self.cache.accumulated_resid(layer=-1, 
                                                     incl_mid=True, 
                                                     pos_slice=-1, 
                                                     return_labels=True)
      logit_diff = self.residual_stack_to_logit_diff(acc_res)
      return logit_diff, labels

    def layerwise_res_logit_diff(self) -> Tuple[Float[Tensor, "..."], List[str]]:
      acc_res, labels = self.cache.decompose_resid(layer=-1, 
                                             pos_slice=-1, 
                                             return_labels=True)
      logit_diff = self.residual_stack_to_logit_diff(acc_res)
      return logit_diff, labels

    def headwise_res_logit_diff(self) -> Tuple[Float[Tensor, "..."], List[str]]:
      acc_res, labels = self.cache.stack_head_results(layer=-1, 
                                                pos_slice=-1, 
                                                return_labels=True)
      logit_diff = self.residual_stack_to_logit_diff(acc_res)
      return logit_diff, labels