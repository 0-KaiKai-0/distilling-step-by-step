# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer


"""T5 Multi-Task by Task Prefix
"""
class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        pred_features = features_df.loc[:, ~features_df.columns.isin(['aux_labels', 'expl_input_ids', 'expl_attention_mask'])].to_dict('records')
        expl_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
            columns={'aux_labels': 'labels', 'expl_input_ids': 'input_ids', 'expl_attention_mask': 'attention_mask'}).to_dict('records')

        pred_features = super().__call__(pred_features, return_tensors)
        expl_features = super().__call__(expl_features, return_tensors)

        return {
            'pred': pred_features,
            'expl': expl_features,
        }


class TaskPrefixTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale


    def compute_loss(self, model, inputs, return_outputs=False):
        # import pdb
        # pdb.set_trace()
        pred_outputs = model(**inputs['pred'])
        expl_outputs = model(**inputs['expl'])

        loss = self.alpha * pred_outputs.loss + (1. - self.alpha) * expl_outputs.loss

        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        if self.output_rationale:
            expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)
        else:
            expl_outputs = pred_outputs # placeholder only

        loss = self.alpha * pred_outputs[0]  + (1 - self.alpha) * expl_outputs[0]

        return (
            loss,
            [pred_outputs[1], expl_outputs[1]],
            [pred_outputs[2], expl_outputs[2]],
        )


class RationaleDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        expl_features = features_df.loc[:, ~features_df.columns.isin(['labels'])].rename(
            columns={'rationales': 'labels'}
        ).to_dict('records')
        if 'rationales' not in features[0]:
            features_df['rationales'] = features_df['input_ids']
        pred_features = features_df.loc[:, ~features_df.columns.isin(['input_ids', 'attention_mask'])].rename(
            columns={'rationales': 'input_ids'}).to_dict('records')

        pred_features = super().__call__(pred_features, return_tensors)
        expl_features = super().__call__(expl_features, return_tensors)

        return {
            'expl': expl_features,
            'pred': pred_features,
        }


class RationaleTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale


    def compute_loss(self, model, inputs, return_outputs=False):
        expl_outputs = model(**inputs['expl'])

        with torch.no_grad():
            output_rationales = model.module.generate(inputs=inputs['expl']['input_ids'], attention_mask=inputs['expl']['attention_mask'], max_length=128)
            inputs['pred']['attention_mask'] = None
            inputs['pred']['input_ids'] = output_rationales

        pred_outputs = model(**inputs['pred'])
        
        alpha = 0.6
        loss = alpha * pred_outputs.loss + (1- alpha) * expl_outputs.loss

        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)

        inputs['pred']['attention_mask'] = None
        inputs['pred']['input_ids'] = expl_outputs[1]
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)

        return pred_outputs
