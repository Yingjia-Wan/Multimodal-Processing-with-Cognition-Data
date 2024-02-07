from ml_things import fix_text
import torch

class ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask. This class will be used as an argument for DataLoader.
    See: pytorch DataLoader at https://pytorch.org/docs/stable/data.html
    
    We customize it to use a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

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
        if self.use_tokenizer.pad_token is None:
            if self.use_tokenizer.eos_token is not None:
                self.use_tokenizer.pad_token = self.use_tokenizer.eos_token
            else:
                # You need to make sure that the EOS token is not None
                self.use_tokenizer.eos_token = '<eos>'
                self.use_tokenizer.pad_token = '<eos>'
        self.pad_cognition = pad_cognition
        self.max_seq_len = max_seq_len
        return

    def pad_tensor(self, tensor, max_cog_len):
        cog_len = tensor.shape[0]
        if cog_len > max_cog_len:
            # Truncate the tensor
            tensor = tensor[:, :max_cog_len]
        elif cog_len < max_cog_len:
            # Pad the tensor with the pad_vector(by changing the shape[0] of the tensor)
                #  the first dimension represents the batch dimension (sequence length); 
                #  the second dimension represents the sequence dimension (cognition embeds).
                # e.g., you would want the padding tensor to have a shape of (max_cog_len, vector dimension)
            # create a cognition padding vector (a vector of -1s).
            pad_vector = torch.ones((1, tensor.shape[1])) * -1 # or any other unique value
            padding = pad_vector.repeat((max_cog_len - cog_len, 1))
            tensor = torch.cat([tensor, padding], dim=0) # pad on the right for encoder-models
        return tensor
    def get_max_cog_length(self, sequences):
        max_cog_len = 0
        for sample in sequences:
            cog_len = len(sample['sentence'].split())
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
        texts = [sequence['sentence'] for sequence in sequences]
        # Fix any unicode issues.
        texts = [fix_text(text) for text in texts]
        # print("texts: ", texts)
        # Encode all texts using tokenizer (we pad here because torchscript need fixed size inputs).
        # tokenized_sentence_ids = self.use_tokenizer(text = texts, return_tensors="pt", padding='longest', truncation=True,
        #             max_length = self.max_seq_len, pad_token = self.use_tokenizer.eos_token, **kwargs
        #             )
        # input_ids = torch.tensor(tokenized_sentence_ids['input_ids'])


        ################################# Cognition ##################################
        if self.pad_cognition:
            # convert cognition to a float tensor (Do not convert to long (int64) tensor, because that will truncate the decimal values.)
            cognition = [sequence['cognition'] for sequence in sequences]
            cognition = [torch.tensor(feature) for feature in cognition]
            # We cannot do: cognition = [torch.tensor(cognition)]
            # because unlike input_ids or labels, cognition is not a list of single items, 
            # but a list of lists that contain different numbers of tensors (because of the different number of words in each sentence).

            # Pad the cognition tensor to the max number of words (on the left).
            max_cog_len = self.get_max_cog_length(sequences)
            # print("max_cog_len: ", max_cog_len)
            padded_cognition = [self.pad_tensor(tensor, max_cog_len) for tensor in cognition]
            padded_cognition = torch.stack(padded_cognition) # now we can stack the tensors into a single tensor.

        ################################# Labels ##################################
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder. 
        # We also don't convert them into tensor (yet) because they are needed in training to store as true_labels numpy array..
        labels = [self.labels_encoder[label] for label in labels]

        ############################### Return Input Dictionary ##################################
        # Return the inputs that are (almost) ready to feed into the model --
        if self.pad_cognition == True:
          inputs = {'cognition': padded_cognition, 'text': texts, 'labels':labels} #padded_cognition is a single tensor.
        else:
          inputs = {'text': texts, 'labels':labels}
        return inputs