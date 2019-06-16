import torch
import pdb

class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class Recall(Metrics):
    """
    Args:
         ats (int): @ to eval.
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self, at=10):
        self.at = at
        self.n = 0
        self.n_correct = 0
        self.name = 'Recall@{}'.format(at)

    def reset(self):
        self.n = 0
        self.n_corrects = 0

    def update(self, predicts, batch):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        predicts = predicts.cpu() # use cpu
        # TODO
        # This method will be called for each batch.
        # You need to
        # - increase self.n, which implies the total number of samples.
        # - increase self.n_corrects based on the prediction and labels
        #   of the batch.
        # n_batch=10 n_sample=5
        
        n_batch,n_samples=predicts.size()
        predict,indexs=predicts.sort(descending=True)
        label=batch['labels'] 
        self.n+=len(batch['labels'])
        for i in range(n_batch):
            for index in indexs[i][:self.at]:
                if index<self.at:
                    ind=index.item()
                    if label[i][ind]==1:
                        self.n_corrects+=1
                        
                        

        
        # for predict in predicts:
        #     if torch.sort(predict, descending=True)[1].tolist().index(0) < self.at:
        #         self.n_corrects += 1
            
        # pdb.set_trace()
    def get_score(self):
        return self.n_corrects / self.n

    def print_score(self):
        score = self.get_score()
        return '{:.2f}'.format(score)
