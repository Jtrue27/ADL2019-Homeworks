import torch
import torch.nn.functional as F 



class AttentionNet(torch.nn.Module):

    def __init__(self, dim_embeddings,similarity='inner_product'):
        super(AttentionNet, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=dim_embeddings,hidden_size=128,num_layers=1,batch_first=True)
        self.rnn_atten = torch.nn.LSTM(input_size=128*4,hidden_size=128,num_layers=1,batch_first=True)
        self.nn=torch.nn.Linear(128,128)

        self.dropout=torch.nn.Dropout(0.4)

    def forward(self, context, context_lens, options, option_lens):
         # With Attention
        c_out, (h_n, h_c) =self.rnn(context,None) # 10 30 128 
        match=self.nn(c_out)
        logits=[]
        for i, option in enumerate(options.transpose(1, 0)):
            o_out, (h_n, h_c)=self.rnn(option,None) # 10 50 128
            o_out=self.dropout(o_out)
            o_out_t=o_out.transpose(1,2) # 32 128 60
            matchs=torch.bmm(match,o_out_t) # 10 33 50
            #softmax become a hat
            a_hat=F.softmax(matchs,1) # 10 33 50
            # compute c
            a_hat_t=a_hat.transpose(1,2) # 10 50 30
            c=torch.bmm(a_hat_t,c_out) # 50 128
            r_minus_c=o_out-c
            r_mul_c=o_out*c
            # chosing c 
            in_rnn=torch.cat((o_out,c,r_mul_c,r_minus_c), 2)
            out_rnn,(h_n, h_c)=self.rnn_atten(in_rnn,None)# 10 50 128*4
            
            atten=out_rnn.max(1)[0]# u 1 128
            contx=c_out.max(1)[0]
          
            
            contx=contx.unsqueeze(1) # 10 1 128
            atten=atten.unsqueeze(2) # 10 128 1 
            score=torch.bmm(contx,atten)
            logit=score.squeeze(1).squeeze(1)
            logits.append(logit) 
        logits = torch.stack(logits, 1)
        return logits


