import torch
import torch.nn.functional as F 




 
class BestNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product'):

        # Best
        super(BestNet, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=dim_embeddings,hidden_size=128,num_layers=1,batch_first=True,bidirectional=True)
        self.rnn2 = torch.nn.LSTM(input_size=256*4,hidden_size=128,num_layers=1,batch_first=True,bidirectional=True)
        self.nn=torch.nn.Linear(256,256)
        self.nn2=torch.nn.Linear(256,256)
        self.softmax=torch.nn.Softmax()
        self.relu= torch.nn.ReLU()
        self.dropout=torch.nn.Dropout(0.4)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256)
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256)
        )
        
        
        
    # run batch training
    def forward(self, context, context_lens, options, option_lens):
        c_out, (h_n, h_c) =self.rnn(context,None) # 10 30 256    
        #last=c_out.max(1)[0]# 10 1 128
        c_out=self.dropout(c_out)
        # match=self.nn(c_out) # 10 33 256
        match=self.mlp(c_out) # 10 33 256
        logits=[]
        for i, option in enumerate(options.transpose(1, 0)):
            o_out, (h_n, h_c)=self.rnn(option,None) # 10 50 128
            # o_out=self.bn(o_out, momentum=0.5)
            # 1. c_out and o_out bmm
            #o_out_1=o_out[:,-1,:] # 10 1 128
            # o_out=self.relu(o_out)
            o_out=self.dropout(o_out)
            option_t=o_out.transpose(1,2) # 10 256 50
            
            matchs=torch.bmm(match,option_t) # 10 33 50
            #softmax become a hat
            a_hat=F.softmax(matchs,1) # 10 33 50
            # compute c
            a_hat_t=a_hat.transpose(1,2) # 10 50 33
            z=torch.bmm(a_hat,o_out) # 33 128  two
            c=torch.bmm(a_hat_t,c_out) # 50 128
            r_Min_c=c_out-z
            r_Mul_c=c_out*z
            r_minus_c=o_out-c
            r_mul_c=o_out*c
            # chosing c 
            in_rnn=torch.cat((o_out,c,r_mul_c,r_minus_c), 2)
            in_rnn2=torch.cat((c_out,z,r_Mul_c,r_Min_c),2)
            out_rnn,(h_n, h_c)=self.rnn2(in_rnn,None)# 10 50 128*4
            out_rnn2,(h_n, h_c)=self.rnn2(in_rnn2,None) # 10 33 128*4
            out_rnn2=self.dropout(out_rnn2)

           
            out_rnn=out_rnn.max(1)[0]# u 1 128
            out_rnn2=out_rnn2.max(1)[0]# v 1 128
            out_rnn=self.mlp2(out_rnn)
            out_rnn=self.dropout(out_rnn)
            out_rnn=out_rnn.unsqueeze(1) # 10 1 256
            out_rnn2=out_rnn2.unsqueeze(2) # 10 256 1 
            #pdb.set_trace()       
            
            score=torch.bmm(out_rnn,out_rnn2)
            logit=score.squeeze(1).squeeze(1)
            logits.append(logit) 
            
        logits = torch.stack(logits, 1)
        return logits



        


