'''
The code is written by the author of the paper: ''Multimodal Processing with Cognition Data'', Yingjia Wan, University of Cambridge, 2023.

The COGMAPL model consists of several components:

    1. LanguageDecoder: 
    This class wraps the encoder model from the transformers library. 
    It prepares inputs for generation, performs the forward pass.

    2. MappingNetwork: 
    This class is a mapping network that takes a tensor of cognition embeddings as input and maps them to the same embedding dimension as the language decoder's embeddings.

    3. COGMAPL: 
    This is the main model class that combines the language decoder and mapping network. 
    It initializes the model configuration, tokenizer, language decoder, and mapping network. 
    It also defines functions for embedding text and cognition, and performs the forward pass of the model.



When you run the code model = COGMAPL().to(device), the following functions in the COGMAPL class will be called:

    __init__: 
    Initializes the COGMAPL model by setting up its components, 
    including the encoder model configuration, tokenizer, language decoder, and mapping network.

    embed_text: 
    This function is not directly called but is used internally in the forward function. 
    It embeds the target text input using the language decoder's embed_tokens method.

    embed_cognition: 
    This function is called in the forward function. 
    It takes the cognition input and passes it through the mapping network to obtain the embedded cognition embeddings.

    forward: 
    This function is called when you pass inputs and labels to the model. 
    It takes the cognition input, target text input, and prefix text input (optional) and performs the forward pass of the model. 
    It calls embed_cognition and embed_text to obtain the embeddings for the inputs and then concatenates them. 
    It applies padding and attention mask and passes the inputs through the language decoder to obtain the outputs.

'''


from pathlib import Path
from typing import List, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import logging
from utils import accumulate_padding, torch_dtype, is_remote_url

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel

# set the device to cuda if available, otherwise cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


logging.set_verbosity_error()
logger = logging.get_logger('transformers')


class LanguageDecoder(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_generation # a customized method that overrides the default behavior of 
                                                                                    # the prepare_inputs_for_generation method in the base language model
                                                                                    # Only activated when specified the function.
                                                                                    # Not activated by default when calling LanguageDecoder.
        self.forward = self.model.forward # call forward in model

        self.config = self.model.config # call config in model
        self.config.pad_token_id = self.config.eos_token_id # set pad_token_id to eos_token_id
        self.encoder = AutoModel.from_config(self.config) # call encoder in model

    @property
    def model_id(self) -> str:
        return type(self.model).__name__.lower()

    @property
    def embed_dim(self) -> int:
        if 'bert' or 'roberta' in self.model_id:
            return self.model.config.hidden_size
        else:
            raise NotImplementedError

    # @property
    def embed_tokens(self, input_ids) -> nn.Module:
        if 'bert' or 'roberta' in self.model_id:
            embeddings = self.encoder(input_ids).last_hidden_state
            return embeddings
        else:
            raise NotImplementedError

    def prepare_inputs_for_generation(self, input_ids, attention_mask, cognition_embeds, past_key_values=None, use_cache=None, **kwargs):
        # align the batch size of cognition embeddings with the corresponding input tokens. 
        expand_size = input_ids.size(0) // cognition_embeds.size(0)
        cognition_embeds = cognition_embeds.repeat_interleave(expand_size, dim=0)
        # Create a mask tensor (cognition_mask) with ones of the same shape as the cognition embeddings to indicate the presence of visual information.
        # (remember the data type should be float tensor for cognition.)
        cognition_mask = torch.ones(cognition_embeds.shape[:2], device=cognition_embeds.device)


        if input_ids[0][0] == self.model.config.bos_token_id:
            input_ids = input_ids[:, 1:]
            attention_mask = attention_mask[:, 1:]

        token_embeds = self.embed_tokens(input_ids)
        
        input_embeds = torch.cat([cognition_embeds, token_embeds], dim=1)
        attention_mask = torch.cat([cognition_mask, attention_mask], dim=1)
        # Note: the binary mask tensor is used to indicate which parts of the input sequence contain visual information and which parts contain tokens from the text sequence. 
            # By concatenating the visual mask and attention mask to create the final attention mask, 
            # where 1s indicate the presence of visual information and tokens to attend to, and 0s indicate padding or tokens to ignore., 
            # the model can differentiate between the visual embeddings and the text embeddings during generation and pay attention to the relevant parts accordingly.

        input_embeds, attention_mask = accumulate_padding(input_embeds, attention_mask)

        if past_key_values:
            input_embeds = input_embeds[:, -1].unsqueeze(1)

        return {
            'inputs_embeds': input_embeds,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'use_cache': use_cache
        }


class MappingNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        output_length: int = 41, #TODO: empirically observed from different batches max_cog_length
        # F.relu refers to the Rectified Linear Unit (ReLU) activation function from the torch.nn.functional module.
            # by default (activation undefined), the activation argument is set to the ReLU activation function (F.relu).
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,

        proj_bias: bool = True
        
    ) -> None:
        super().__init__()
        # self.down = nn.Linear(input_dim, hidden_dim, bias=proj_bias)
        self.up = nn.Linear(input_dim, output_dim, bias=proj_bias)

        # The purpose of including self.const is to introduce a fixed bias or prior knowledge into the mapping process. 
            # By incorporating this constant value, 
            # the model can learn to leverage the fixed information provided by self.const 
            # to improve the quality of the projection.
        self.const = nn.Parameter(torch.randn(output_length, input_dim))
        # self.norm = nn.LayerNorm(output_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenates x with a repeated version of self.const (constant tensor) along the second dimension. 
            # This concatenation allows the model to have access to a fixed additional input that does not depend on the input data but can influence the output.
            ## dim =1: concatenate along the second dimension
            ## unsqueeze(0): add a dimension of size 1 in the first dimension to match the sahpe of x (batch_size, sequence_length, embedding_size)
            ## expand(x.size(0), -1): expand the tensor along the first dimension to match the size of x while -1 means the second dimension remains unchanged
        x = torch.cat((x, self.const.unsqueeze(0).expand(x.size(0), -1, -1)), dim=1)
        x = x[:, -self.const.size(0):]
        x = x.to(torch.float32) # because the const tensor is initialized as a float64 tensor
        x = self.up(x)
        # x = self.norm(x)
        return x


class COGMAPL(nn.Module):
# the COGMAPL model is designed to incorporate language understanding and generation capabilities alongside cognition processing.
    def __init__(
        self,
        gpt_model_id: str = None,
        cognition_type: str = None,
        num_labels: int = None
    ) -> None:
        super().__init__()
        # print('num_labels: ', num_labels)
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path=gpt_model_id, num_labels=num_labels)

        self.text_processor = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=gpt_model_id)
        # padding token:  the BERT tokenizer includes a padding token ('[PAD]') by default. But errors may occur here.
        if self.text_processor._pad_token is None:                 # so we add it manually to make sure.
            if self.text_processor._eos_token is None:
                self.text_processor.eos_token = '<eos>'
                self.text_processor.pad_token = '<eos>'
            else:
                self.text_processor.pad_token = self.text_processor.eos_token
                self.text_processor.pad_token_id = self.text_processor.eos_token_id
        # Padding side (left/right) also doesn't matter, as long as it is consistent.


        if cognition_type == 'ET':
            self.input_dim = 4
        elif cognition_type == 'ET_ZAB':
            self.input_dim = 5
        elif cognition_type == 'EEG':
            self.input_dim = 104
        elif cognition_type == 'ET+EEG':
            self.input_dim = 108 # 104 + 4
        elif cognition_type == 'ET_ZAB+EEG':
            self.input_dim = 109 # 104 + 5
        
        # Call the LanguageDecoder class to perform a series of functions in/outside the COGMAPL class: prepare input for geneation, forward pass, and and generation.
        self.language_decoder = LanguageDecoder(AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=gpt_model_id, num_labels=num_labels))
        
        # FREEZE or unfreeze THE language_decoder #TODO: change?
        for param in self.parameters():
            param.requires_grad = True

        # output_dim = language_model.config.hidden_size  # Use the hidden size of the LM
        # print("self.language_decoder.embed_dim", self.language_decoder.embed_dim)
        self.mapper = MappingNetwork(
            input_dim=self.input_dim,
            output_dim=self.language_decoder.embed_dim
        )

        # print("\n Completed initializing COGMAPL model! Passing inputs to Forward()...\n")
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        dtype: Union[str, torch.dtype] = None,
        **kwargs
    ) -> nn.Module:
        with torch_dtype(dtype):
            model = cls(**kwargs)

        logger.info(f"Loading mapper weights from {checkpoint_path}")
        if is_remote_url(checkpoint_path):
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_path, map_location='cpu')
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict) # load整个model的state_dict, 包括language_decoder和mapper的state_dict
        # 不需要load optimizer的state_dict, 因为load checkpoint是为了evaluation,不是为了继续训练

        return model
    
    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            token_embeds = self.language_decoder.embed_tokens(input_ids)
        return token_embeds

    def embed_cognition(self, cognition: torch.Tensor) -> torch.Tensor:
        patch_embeds = self.mapper(cognition)
        return patch_embeds # output: (batch_size, num_patches, embed_dim)
    
    def forward(
        self,
        cognition: torch.Tensor = None,
        text: str = None,
        labels: torch.Tensor = None, # labels is added for text classification tasks.
        prefix_ids: torch.Tensor = None # prefix_ids is the context input for text completion and generation tasks.
    ) -> torch.Tensor:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. 
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # tokenize the text input
        if text is not None:
            input_ids = self.text_processor(text, return_tensors='pt', padding='longest', truncation=True,
                    # max_length = self.max_seq_len, 
                    # pad_token = self.use_tokenizer.eos_token, 
                    )['input_ids']
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_ids = input_ids.to(device)
        # convert input ids to LM-dimension token embeddings
            target_embeds = self.embed_text(input_ids)

        # concatenate the attention masks for cognition and text
        if cognition is None:
            input_embeds = target_embeds
            attention_mask = (input_ids != self.text_processor.pad_token_id).long()
            if prefix_ids is not None:
                prefix_token_mask = (prefix_ids != self.text_processor.pad_token_id).long()
                attention_mask = torch.cat((prefix_token_mask, target_token_mask), dim=1)
        else:
            # Map cognition to LM-dimension patch embeddings
            cognition_embeds = self.embed_cognition(cognition)
            # concatenate the cognition and text embeddings into input embeddings
            if prefix_ids is None:
                input_embeds = torch.cat((target_embeds, cognition_embeds), dim=1)
            else:
                prefix_embeds = self.embed_text(prefix_ids)
                input_embeds = torch.cat((prefix_embeds, target_embeds, cognition_embeds), dim=1)

            # concatenate the attention masks for cognition and text
            # cognition_mask = torch.ones(cognition_embeds.shape[:2], device=cognition_embeds.device) # corrected below
            cognition_mask = (cognition_embeds != -1).any(dim=-1).long().to(cognition_embeds.device) # because the -1 is the padding value
            target_token_mask = (input_ids != self.text_processor.pad_token_id).long()
            if prefix_ids is None:
                attention_mask = torch.cat((target_token_mask, cognition_mask), dim=1)
            else:
                prefix_token_mask = (prefix_ids != self.text_processor.pad_token_id).long()
                attention_mask = torch.cat((prefix_token_mask, target_token_mask, cognition_mask), dim=1)

        # padding both the input_embeds and attention_mask
        input_embeds, attention_mask = accumulate_padding(
                                                        input_embeds, 
                                                        attention_mask,
                                                        padding_side = 'right')

        '''
        Classification training: calling the language_decoder and thus the model. 
        The backbone model will compute and return: loss, logits, hidden_states, attentions.
        if labels is None, loss = None; 
        otherwise, loss = CrossEntropyLoss(logits, labels)
        '''
        # print('\n Calling language_decoder... \n')
        outputs = self.language_decoder(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
        
        return outputs
    
    
    def text_transform(self, text: Union[str, List[str]], **kwargs) -> torch.Tensor:
        return self.text_processor(text, padding='longest', return_tensors='pt', **kwargs)
