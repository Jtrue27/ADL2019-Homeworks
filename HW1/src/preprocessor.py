import json
import logging
from multiprocessing import Pool
from dataset import DialogDataset
from tqdm import tqdm
import nltk


class Preprocessor:
    """

    Args:
        embedding_path (str): Path to the embedding to use.
    """
    def __init__(self, embedding):
        self.embedding = embedding
        self.logging = logging.getLogger(name=__name__)

    def tokenize(self, sentence):
        """ Tokenize a sentence.
        Args:
            sentence (str): One string.
        Return:
            indices (list of str): List of tokens in a sentence.
        """
        tokenizes=nltk.word_tokenize(sentence)
        return tokenizes
        

    def sentence_to_indices(self, sentence):
        """ Convert sentence to its word indices.
        Args:
            sentence (str): One string.
        Return:
            indices (list of int): List of word indices.
        """
        # TODO
        # Hint: You can use `self.embedding`
        tokenizes=self.tokenize(sentence)
        word_indices=[]
        for token in tokenizes:
            word_indices.append(self.embedding.to_index(token))
        return word_indices

    def collect_words(self, data_path, n_workers=12):
        with open(data_path) as f:
            data = json.load(f)

        utterances = [] # express
        for sample in data: # sample is  a dialog
            utterances += (
                [message['utterance']
                 for message in sample['messages-so-far']] # message talk and response
                + [option['utterance']
                   for option in sample['options-for-next']] 

            )
        utterances = list(set(utterances)) # remove repeat value to lot of word
       
        chunks = [
            ' '.join(utterances[i:i + len(utterances) // n_workers])
            for i in range(0, len(utterances), len(utterances) // n_workers)
        ]
        with Pool(n_workers) as pool:
            chunks = pool.map_async(self.tokenize, chunks)
            words = set(sum(chunks.get(), []))

        return words

    def get_dataset(self, data_path, n_workers=4, dataset_args={}):
        """ Load data and return Dataset objects for training and validating.

        Args:
            data_path (str): Path to the data.
            valid_ratio (float): Ratio of the data to used as valid data.
        """
        self.logging.info('loading dataset...')
        with open(data_path) as f:
            dataset = json.load(f) 

        self.logging.info('preprocessing data...')

        results = [None] * n_workers # [None, None, None, None,..]
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):# parallel
                batch_start = (len(dataset) // n_workers) * i
                if i == n_workers - 1:
                    batch_end = len(dataset)
                else:
                    batch_end = (len(dataset) // n_workers) * (i + 1)

                batch = dataset[batch_start: batch_end]
                results[i] = pool.apply_async(self.preprocess_samples, [batch]) # pool assign worker

                # When debugging, you'd better not use multi-thread.
                # results[i] = self.preprocess_dataset(batch, preprocess_args)

            pool.close()
            pool.join()

        processed = []
        for result in results:
            processed += result.get()

        padding = self.embedding.to_index('</s>')
        return DialogDataset(processed, padding=padding, **dataset_args) #proceessed 處理過後的 word vector

    def preprocess_samples(self, dataset):
        """ Worker function.

        Args:
            dataset (list of dict)
        Returns:
            list of processed dict.
        """
        processed = []
        for sample in tqdm(dataset):# dateset = entire json
            processed.append(self.preprocess_sample(sample))

        return processed

    def preprocess_sample(self, data): #data is [{}]
        """
        Args:
            data (dict)
        Returns:
            dict
        """
        processed = {}
        processed['id'] = data['example-id'] #outer [{}]

        # process messages-so-far
        processed['context'] = []
        processed['speaker'] = []
        for message in data['messages-so-far']: #messqge= one person talk
            processed['context'].append(
                self.sentence_to_indices(message['utterance'].lower()+' '+message['speaker']) # add 
            )

        # process options
        processed['options'] = []
        processed['option_ids'] = []

        # process correct options
        if 'options-for-correct-answers' in data:
            processed['n_corrects'] = len(data['options-for-correct-answers'])
            for option in data['options-for-correct-answers']:
                processed['options'].append(
                    self.sentence_to_indices(option['utterance'].lower()+' participant_2') # 把correct answer to index 放到processed['option']
                )
                processed['option_ids'].append(option['candidate-id']) # candidate-id 放到 processed['option_ids']
        else:
            processed['n_corrects'] = 0

        # process the other options
        for option in data['options-for-next']:
            if option['candidate-id'] in processed['option_ids']:# avoid repeat append
                continue

            processed['options'].append(
                self.sentence_to_indices(option['utterance'].lower()+' participant_2')
            )
            processed['option_ids'].append(option['candidate-id'])

        return processed
        # processed={
        #   'id':,
        #   'context':對話的句子,
        #   'speaker':,
        #   'options':utterance,
        #   'option_ids':candidate-id,
        #   'n_corrects':答案數,
        # }