"""Utility functions for training.
"""

import json
import torch
import torchtext


# ===========================================================
# Vocabulary.
# ===========================================================

class Vocabulary(object):
    """Keeps track of all the words in the vocabulary.
    """

    # Reserved symbols
    SYM_PAD = '<pad>'    # padding.
    SYM_SOQ = '<start>'  # Start of question.
    SYM_SOR = '<resp>'   # Start of response.
    SYM_EOS = '<end>'    # End of sentence.
    SYM_UNK = '<unk>'    # Unknown word.

    def __init__(self):
        """Constructor for Vocabulary.
        """
        # Init mappings between words and ids
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word(self.SYM_SOQ)
        self.add_word(self.SYM_EOS)
        self.add_word(self.SYM_UNK)
        self.add_word(self.SYM_PAD)        
        self.add_word(self.SYM_SOR)

    def __call__(self, word):
        return self.__getitem__(word)
        
    def add_word(self, word):
        """Adds a new word and updates the total number of unique words.

        Args:
            word: String representation of the word.
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def remove_word(self, word):
        """
        Removes a specified word and updates the total number of unique words.

        Args:
            word: String representation of the word.
        """
        if word in self.word2idx:
            self.word2idx.pop(word)
            self.idx2word.pop(self.idx)
            self.idx -= 1

    def __getitem__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.SYM_UNK]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def save(self, location):
        with open(location, 'w') as f:
            f.write(json.dumps({'word2idx': self.word2idx,
                       'idx2word': self.idx2word,
                       'idx': self.idx}))

    def load(self, location):
        with open(location, 'r') as f:
            data = json.loads(f.read())
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.idx = data['idx']

    def tokens_to_words(self, tokens):
        """Converts tokens to vocab words.

        Args:
            tokens: 1D Tensor of Token outputs.

        Returns:
            A list of words.
        """
        words = []
        for token in tokens:
            try:
                word = self.idx2word[str(token)]
            except:
                word = self.idx2word[str(token.item())]
            if word == self.SYM_EOS:
                break
            if word not in [self.SYM_PAD, self.SYM_SOQ,
                            self.SYM_SOR, self.SYM_EOS]:
                words.append(word)
        sentence = str(' '.join(words))
        return sentence


def get_glove_embedding(name, embed_size, vocab):
    """Construct embedding tensor.

    Args:
        name (str): Which GloVe embedding to use.
        embed_size (int): Dimensionality of embeddings.
        vocab: Vocabulary to generate embeddings.
    Returns:
        embedding (vocab_size, embed_size): Tensor of
            GloVe word embeddings.
    """

    glove = torchtext.vocab.GloVe(name=name,
                                  dim=str(embed_size))
    vocab_size = len(vocab)
    embedding = torch.zeros(vocab_size, embed_size)
    for i in range(vocab_size):
        embedding[i] = glove[vocab.idx2word[str(i)]]
    return embedding


# ===========================================================
# Helpers.
# ===========================================================

def process_lengths(inputs, pad=0):
    """Calculates the lenght of all the sequences in inputs.

    Args:
        inputs: A batch of tensors containing the question or response
            sequences.

    Returns: A list of their lengths.
    """
    max_length = inputs.size(1)
    if inputs.size(0) == 1:
        lengths = list(max_length - inputs.data.eq(pad).sum(1))
    else:
        lengths = list(max_length - inputs.data.eq(pad).sum(1).squeeze())
    return lengths


# ===========================================================
# Evaluation metrics.
# ===========================================================

def gaussian_KL_loss(mus, logvars, eps=1e-8):
    """Calculates KL distance of mus and logvars from unit normal.

    Args:
        mus: Tensor of means predicted by the encoder.
        logvars: Tensor of log vars predicted by the encoder.

    Returns:
        KL loss between mus and logvars and the normal unit gaussian.
    """
    KLD = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
    kl_loss = KLD/(mus.size(0) + eps)
    """
    if kl_loss > 100:
        print kl_loss
        print KLD
        print mus.min(), mus.max()
        print logvars.min(), logvars.max()
        1/0
    """
    return kl_loss


def vae_loss(outputs, targets, mus, logvars, criterion):
    """VAE loss that combines cross entropy with KL divergence.

    Args:
        outputs: The predictions made by the model.
        targets: The ground truth indices in the vocabulary.
        mus: Tensor of means predicted by the encoder.
        logvars: Tensor of log vars predicted by the encoder.
        criterion: The cross entropy criterion.
    """
    CE = criterion(outputs, targets)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = gaussian_KL_loss(mus, logvars)
    return CE + KLD



class AverageMeter(object):
    """Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(preds, targets, k):
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 1, True, True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)


def calculate_caption_lengths(word_dict, captions):
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (word_dict['<start>'], word_dict['<eos>'], word_dict['<pad>']):
                continue
            else:
                lengths += 1
    return lengths