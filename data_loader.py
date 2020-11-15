"""Loads question answering data and feeds it to the models.
"""

import h5py
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import os
import copy
import pickle

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class DatasetLoader(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, opt, dataset, transform=None, max_examples=None,
                 indices=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions and images.
            images: hdf5 file with questions and imags.
            transform: image transformer.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
            indices: List of indices to use.
        """
        self.dataset = dataset
        self.transform = transform
        self.max_examples = max_examples
        self.indices = indices
        # self.annos_allowed = annos_allowed
        self.opt = copy.copy(opt)



        self.dir_raw = os.path.join(self.opt['dir'], 'raw')
        if not os.path.exists(self.dir_raw):
            self._raw()
  
        if 'select_questions' in opt.keys() and opt['select_questions']:
            self.dir_interim = os.path.join(self.opt['dir'], 'selected_interim')
            if not os.path.exists(self.dir_interim):
                self._interim(select_questions=True)
        else:
            self.dir_interim = os.path.join(self.opt['dir'], 'interim')
            if not os.path.exists(self.dir_interim):
                self._interim(select_questions=False)
  
        self.dir_processed = os.path.join(self.opt['dir'], 'processed')
        self.subdir_processed = self.subdir_processed()
        if not os.path.exists(self.subdir_processed):
            self._processed()

        path_aid_to_ans  = os.path.join(self.subdir_processed, 'aid_to_ans.pickle')
        path_ans_to_aid  = os.path.join(self.subdir_processed, 'ans_to_aid.pickle')
        # path_dataset     = os.path.join(self.subdir_processed, self.data_split+'set.pickle')
        # path_cid_to_concept = os.path.join(self.subdir_processed, 'cid_to_concept.pickle')
        # path_concept_to_cid = os.path.join(self.subdir_processed, 'concept_to_cid.pickle')
        
        # with open(path_wid_to_word, 'rb') as handle:
        #     self.wid_to_word = pickle.load(handle)
  
        # with open(path_word_to_wid, 'rb') as handle:
        #     self.word_to_wid = pickle.load(handle)
  
        with open(path_aid_to_ans, 'rb') as handle:
            self.aid_to_ans = pickle.load(handle)
  
        with open(path_ans_to_aid, 'rb') as handle:
            self.ans_to_aid = pickle.load(handle)
 
        # with open(path_dataset, 'rb') as handle:
        #     self.dataset = pickle.load(handle)

        if not hasattr(self, 'images'):
            annos = h5py.File(self.dataset, 'r')
            self.questions = annos['questions']
            self.answers = annos['answers']
            self.answer_types = annos['answer_types']
            self.image_indices = annos['image_indices']
            self.images = annos['images']
        
            self.labeln = np.array(self.answer_types)
            self.unique_labels = np.unique(self.labeln)
        
    def subdir_processed(self):
        subdir = 'nans,' + str(self.opt['nans']) \
              + '_maxlength,' + str(self.opt['maxlength']) \
              + '_minwcount,' + str(self.opt['minwcount']) \
              + '_nlp,' + self.opt['nlp'] \
              + '_pad,' + self.opt['pad'] \
              + '_trainsplit,' + self.opt['trainsplit']
        if 'select_questions' in self.opt.keys() and self.opt['select_questions']:
            subdir += '_filter_questions'
        subdir = os.path.join(self.dir_processed, subdir)
        return subdir
    def vocab_answers(self):
        return self.aid_to_ans
    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
        if self.indices is not None:
            index = self.indices[index]
        question = self.questions[index]
        answer = self.answers[index]
        answer_type = self.answer_types[index]
        image_index = self.image_indices[index]
        image = self.images[image_index]

        question = torch.from_numpy(question)
        answer = torch.from_numpy(answer)
        alength = answer.size(0) - answer.eq(0).sum(0).squeeze()
        qlength = question.size(0) - question.eq(0).sum(0).squeeze()
        if self.transform is not None:
            image = self.transform(image)
        return (image, question, answer, answer_type,
                qlength, alength.item())

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        annos = h5py.File(self.dataset, 'r')
#         print(annos['questions'].shape)
        return annos['questions'].shape[0]


# def collate_fn(data):
#     """Creates mini-batch tensors from the list of tuples.

#     We should build custom collate_fn rather than using default collate_fn,
#     because merging caption (including padding) is not supported in default.

#     Args:
#         data: list of tuple (image, question, answer, answer_type, length).
#             - image: torch tensor of shape (3, 256, 256).
#             - question: torch tensor of shape (?); variable length.
#             - answer: torch tensor of shape (?); variable length.
#             - answer_type: Int for category label
#             - qlength: Int for question length.
#             - alength: Int for answer length.

#     Returns:
#         images: torch tensor of shape (batch_size, 3, 256, 256).
#         questions: torch tensor of shape (batch_size, padded_length).
#         answers: torch tensor of shape (batch_size, padded_length).
#         answer_types: torch tensor of shape (batch_size,).
#         qindices: torch tensor of shape(batch_size,).
#     """
#     # Sort a data list by caption length (descending order).
#     data.sort(key=lambda x: x[5], reverse=True)
#     images, questions, answers, answer_types, qlengths, _ = zip(*data)
#     images = torch.stack(images, 0)
#     questions = torch.stack(questions, 0).long()
#     answers = torch.stack(answers, 0).long()
#     answer_types = torch.Tensor(answer_types).long()
#     qindices = np.flip(np.argsort(qlengths), axis=0).copy()
#     qindices = torch.Tensor(qindices).long()
#     return images, questions, qlengths, answers, answer_types, qindices





class UnSupDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, data_path, clean_transform=None, noise_transform=None, max_examples=None,
                 indices=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions and images.
            images: hdf5 file with questions and imags.
            transform: image transformer.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
            indices: List of indices to use.
        """
        self.data_path = data_path
        annos = h5py.File(data_path, 'r')
        self.images = annos['images']

        # self.images = images
        # self.image_dir = img_dir
        self.clean_transform = clean_transform
        self.noise_transform = noise_transform
        self.max_examples = max_examples
        self.indices = indices

    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
        image = self.images[index]
        # image = Image.fromarray(image)
        # print(type(image))
        # image = torch.from_numpy(image)
        # print(image.size())
        # exit()
        target = self.clean_transform(image)
        input = self.noise_transform(image)
        return input, target

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        else:
            annos = h5py.File(self.data_path, 'r')
            return annos['images'].shape[0]


def new_collate(images):
    images = torch.stack(images, 0)
    return images

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples.

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, question, answer, answer_type, length).
            - image: torch tensor of shape (3, 256, 256).
            - question: torch tensor of shape (?); variable length.
            - answer: torch tensor of shape (?); variable length.
            - answer_type: Int for category label
            - qlength: Int for question length.
            - alength: Int for answer length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        questions: torch tensor of shape (batch_size, padded_length).
        answers: torch tensor of shape (batch_size, padded_length).
        answer_types: torch tensor of shape (batch_size,).
        qindices: torch tensor of shape(batch_size,).
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: x[5], reverse=True)
    images, questions, answers, answer_types, qlengths, _ = zip(*data)
    images = torch.stack(images, 0)
    questions = torch.stack(questions, 0).long()
    answers = torch.stack(answers, 0).long()
    answer_types = torch.Tensor(answer_types).long()
    qindices = np.flip(np.argsort(qlengths), axis=0).copy()
    qindices = torch.Tensor(qindices).long()
    # print(qlengths)
    qlengths = torch.stack(qlengths).long()
    return images, questions, qlengths, answers, answer_types, qindices

