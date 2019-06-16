import torch
import torch.nn.functional as F 



class RnnNet(torch.nn.Module):

    def __init__(self, dim_embeddings,similarity='inner_product'):
        super(RnnNet, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=300,hidden_size=128,num_layers=2,batch_first=True)
        self.relu= torch.nn.ReLU()
        self.nn=torch.nn.Linear(128,128)
        self.softmax=torch.nn.Softmax()
    

    def forward(self, context, context_lens, options, option_lens):
        c_out, (h_n, h_c) =self.rnn(context,None)
        u=self.relu(c_out)
        uW=self.nn(u)
        out_context= uW[:,-1,:] # 10 128

        logits=[]
        for i, option in enumerate(options.transpose(1, 0)):
            o_out, (h_n, h_c)=self.rnn(option,None)
            #v=self.nn(o_out)
            v=self.relu(o_out)
            out_option= v[:,-1,:] # 10 128
            u_context=out_context.unsqueeze(1) # 10 1 128
            u_option=out_option.unsqueeze(2) # 10 128 1
            score=torch.bmm(u_context,u_option)
            logit=score.squeeze(1).squeeze(1)
            logits.append(logit) 
        logits = torch.stack(logits, 1)
        return logits



