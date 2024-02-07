from ml_things import fix_text
import torch

class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask. This class will be used as an argument for DataLoader.
    See: pytorch DataLoader at https://pytorch.org/docs/stable/data.html
    
    We customize it to use a given tokenizer and label encoder to convert any text and labels to go straight into a GPT2 model.

    In the case of COGMAPL, the input is a cognition embedding tensor and a sentence embedding tensor, which are concatenated together and then fed into the COGMAPL model.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.
    """

    def __init__(self, use_tokenizer, labels_encoder, pad_cognition, max_seq_len): 
        # Label encoder used inside the class (which is basically a dict).
        self.labels_encoder = labels_encoder
        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        self.pad_cognition = pad_cognition
        self.max_seq_len = max_seq_len
        return
    
    # def pad_tensor(self, tensor, max_cog_len): # This is for in case we want to pad the cognition embedding before feeding it into the model for concatenation with the sentence embedding.
    #     cog_len = tensor.shape[0]
    #     if cog_len > max_cog_len:
    #         # Truncate the tensor
    #         tensor = tensor[:, :max_cog_len]
    #     elif cog_len < max_cog_len:
    #         # Pad the tensor with zeros (by changing the shape[0] of the tensor)
    #             #  the first dimension represents the batch dimension (sequence length); 
    #             #  the second dimension represents the sequence dimension (cognition embeds).
    #             # e.g., you would want the padding tensor to have a shape of (max_cog_len, 5)

    #         # preserve the second dimension (sequence dimension): tensor.shape[1]
    #         padding = torch.zeros((max_cog_len - cog_len, tensor.shape[1]), dtype=torch.double)
    #         # concatenate along the first dimension (batch dimension): tensor.shape[0] (on the LEFT side of the tensor)
    #         tensor = torch.cat([padding, tensor], dim=0)
    #     return tensor

    #TODO: corrected
    
    def pad_tensor(self, tensor, max_cog_len):
        cog_len = tensor.shape[0]
        if cog_len > max_cog_len:
            # Truncate the tensor
            tensor = tensor[:, :max_cog_len]
        elif cog_len < max_cog_len:
            '''Pad the tensor with the pad_vector(by changing the shape[0] of the tensor)
                #  the first dimension represents the batch dimension (sequence length); 
                #  the second dimension represents the sequence dimension (cognition embeds).
                # e.g., you would want the padding tensor to have a shape of (max_cog_len, vector dimension)
            '''
            # padded the cog vector with -1s.
            pad_vector = torch.ones((1, tensor.shape[1])) * -1 # or any other unique value
            padding = pad_vector.repeat((max_cog_len - cog_len, 1))
            tensor = torch.cat([tensor, padding], dim=0) # pad on the right for encoder-models
        return tensor
    
    def get_max_cog_length(self, sequences):
        max_cog_len = 0
        for sample in sequences:
            cog_len = len(sample[1].split()) # split the sentence into words and count the number of words.
            # Note: this is not the same as the max length of the sequence (which means the max number of tokens), 
            # but rather the max number of words that are readily split by the cognition corpora.
            # the max length of the sequence would be: cog_len = len(self.tokenizer(sample['sentence']))
            if cog_len > max_cog_len:
                max_cog_len = cog_len
        return max_cog_len

    def __call__(self, sequences, **kwargs):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """
        ################################# Texts ##################################
        # Get all texts from sequences list.
        # print(type(sequences))
        # print("sequences: ", sequences)
        texts = [sequence[1] for sequence in sequences] # TODO: double check printout

        # Fix any unicode issues.
        texts = [fix_text(text) for text in texts]
        # print("texts: ", texts)
        # Encode all texts using tokenizer (we pad here because torchscript need fixed size inputs).
        tokenized_sentence_ids = self.use_tokenizer(text = texts, return_tensors="pt", padding='longest', truncation=True,
                    max_length = self.max_seq_len, 
                    padding_side = 'left', # TODO: conflicting with AutoTokenizer
                    **kwargs
                    # truncation_side = 'left'
                    )
        input_ids = torch.tensor(tokenized_sentence_ids['input_ids'])
        # print("tokenized_input_ids: ", input_ids) # double check
        # (you can avoid padding using the following code, but torch.stack requires fixed length to convert to a tensor. Thus, you will find it struggling to feed them into the model to get text embeddings without padding)
        # input_ids = []
        # for text in texts:
        #     tokenized_sentence_ids = self.use_tokenizer(text = text, return_tensors="pt", padding='longest', truncation=True,
        #                         padding_side = 'left', truncation_side = 'left', max_length = self.max_seq_len)
        #     input_ids.append(tokenized_sentence_ids['input_ids'])
        # print("tokenized_sentence_ids: ", input_ids)
        # input_ids = torch.stack(input_ids) # ERROR HERE

        ################################# Cognition ##################################
        # convert cognition to a float tensor (Do not convert to long (int64) tensor, because that will truncate the decimal values.)
        cognition = [sequence[0] for sequence in sequences]
        cognition = [torch.tensor(feature) for feature in cognition]
        # We cannot do: cognition = [torch.tensor(cognition)]
        # because unlike input_ids or labels, cognition is not a list of single items, 
        # but a list of lists that contain different numbers of tensors (because of the different number of words in each sentence).

        # [OPTIONAL depending on what we want our concatenated embedding to be like]
        # Pad the cognition tensor to the max number of words (on the left).
        max_cog_len = self.get_max_cog_length(sequences)
        # print("max_cog_len: ", max_cog_len)
        padded_cognition = [self.pad_tensor(tensor, max_cog_len) for tensor in cognition]
        padded_cognition = torch.stack(padded_cognition) # now we can stack the tensors into a single tensor.

        ################################# Labels ##################################

        # Get all labels from sequences list.
        labels = [sequence[2] for sequence in sequences]

        # Encode all labels using label encoder. 
        labels = [self.labels_encoder[label] for label in labels] # label_encoder: {'negative': 0, 'neutral':1, 'positive': 2}
        labels = torch.tensor(labels, dtype=torch.long)


        ############################### Return Input Dictionary ##################################
        # Return the inputs that are (almost) ready to feed into the model --
        if self.pad_cognition == True:
          inputs = {'cognition': padded_cognition, 'input_ids': input_ids, 'labels': labels} 
          #padded_cognition is a single tensor.
        else:
          inputs = {'cognition': cognition, 'input_ids': input_ids, 'labels': labels} 
          #cognition is a list of tensors, not a single tensor.
        return inputs