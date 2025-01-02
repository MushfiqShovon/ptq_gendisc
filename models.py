import torch.nn as nn
import torch.nn.utils.rnn
#import brevitas.nn as qnn

#from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, SignedBinaryWeightPerTensorConst, SignedBinaryActPerTensorConst, SignedTernaryWeightPerTensorConst, SignedTernaryActPerTensorConst, Uint8ActPerTensorFixedPoint

#Class for full precision Model
class LSTMNet(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(LSTMNet,self).__init__()
        
        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim, bias=False)
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)
        
        packed_output,(hidden_state,cell_state) = self.lstm(embedded)

        hidden = hidden_state[-1,:,:]
        
        dense_outputs=self.fc(hidden)

        return dense_outputs


class Gen(nn.Module):
    def __init__(self, ntoken, ninp, nlabelembed, nhid, nlayers, nclass, dropout, use_cuda, 
        tied, use_bias, concat_label, avg_loss, one_hot):
        
        super(Gen, self).__init__()
        
        self.drop = nn.Dropout(dropout)
        
        self.encoder = nn.Embedding(ntoken, ninp)
          
        self.label_encoder = nn.Embedding(nclass, nlabelembed)

        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=False)

        self.decoder = nn.Linear((nhid + nlabelembed), ntoken, False)

        self.nlayers = nlayers
        self.nhid = nhid
        self.loss = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda
        self.nclass = nclass
        self.avg_loss = avg_loss
        self.use_bias = use_bias
        self.concat_label = concat_label
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, x_pred, y_ext, hidden, criterion, model_state = 'fp'):
        
        #print("x.data.shape:", x.data.shape)
        embedded_sents = self.encoder(x.data)
        #print("embedded_sents.shape:", embedded_sents.shape)

        embedded_label = self.label_encoder(y_ext.data)
        #print("embedded_label.shape:", embedded_label.shape)

        output, (_, _) = self.rnn(embedded_sents, hidden)
        #print("output.shape:", output.shape)

        if model_state == 'quant': #calibration of quantized model
            output=output.mean(dim=1)
            hidden_data = torch.cat([output, embedded_label], dim=1)  # float cat
        else: #inference with fp model
            hidden_data = torch.cat((output, embedded_label), 1)
        #print("hidden_data.shape:", hidden_data.shape)

        hidden_data = self.drop(hidden_data)
        #print("hidden_data.shape:", hidden_data.shape)

        out = self.decoder(hidden_data)
        #print("out.shape:", out.shape)
        return out


#Class for Brevitas Quantization model (QAT)

class QLSTM(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(QLSTM,self).__init__()
        
        self.embedding = qnn.QuantEmbedding(vocab_size,embedding_dim, weight_quant = Int8WeightPerTensorFixedPoint, weight_bit_width=bit_witdh)

        self.quantLSTM = qnn.QuantLSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            weight_bit_width=bit_witdh,
            io_bit_width=bit_witdh,
            gate_acc_bit_width=bit_witdh,
            sigmoid_bit_width=bit_witdh,
            tanh_bit_width=bit_witdh,
            weight_quant = Int8WeightPerTensorFixedPoint, 
            io_quant = Int8ActPerTensorFixedPoint,
            gate_acc_quant=Int8ActPerTensorFixedPoint,
            sigmoid_quant=Uint8ActPerTensorFixedPoint,
            tanh_quant=Int8ActPerTensorFixedPoint,
            return_quant_tensor=True,
            batch_first=True)

        
        self.fc = qnn.QuantLinear(hidden_dim, output_dim, bias=False, weight_quant = Int8WeightPerTensorFixedPoint, weight_bit_width=bit_witdh)
        
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)

        
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        padded_embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        
        padded_output,(hidden_state,cell_state) = self.quantLSTM(padded_embedded)

        dense_outputs=self.fc(hidden_state)

        return dense_outputs


#Class for Brevitas Quantization model with 1bit (QAT)
class QLSTM_1bit(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(QLSTM_1bit,self).__init__()
        
        self.embedding = qnn.QuantEmbedding(vocab_size,embedding_dim, weight_quant = SignedBinaryWeightPerTensorConst, weight_bit_width=bit_witdh)

        self.quantLSTM = qnn.QuantLSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            weight_bit_width=bit_witdh,
            io_bit_width=bit_witdh,
            gate_acc_bit_width=bit_witdh,
            sigmoid_bit_width=bit_witdh,
            tanh_bit_width=bit_witdh,
            weight_quant = SignedBinaryWeightPerTensorConst, 
            io_quant = SignedBinaryActPerTensorConst,
            gate_acc_quant=SignedBinaryActPerTensorConst,
            sigmoid_quant=SignedBinaryActPerTensorConst,
            tanh_quant=SignedBinaryActPerTensorConst,
            return_quant_tensor=True,
            batch_first=True)

        
        self.fc = qnn.QuantLinear(hidden_dim, output_dim, bias=False, weight_quant = SignedBinaryWeightPerTensorConst, weight_bit_width=bit_witdh)
        
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)

        
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        padded_embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        
        padded_output,(hidden_state,cell_state) = self.quantLSTM(padded_embedded)

        dense_outputs=self.fc(hidden_state)

        return dense_outputs



#Class for Brevitas Quantization model with 2bit (QAT)
class QLSTM_2bit(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(QLSTM_2bit,self).__init__()
        
        self.embedding = qnn.QuantEmbedding(vocab_size,embedding_dim, weight_quant = SignedTernaryWeightPerTensorConst, weight_bit_width=bit_witdh)

        self.quantLSTM = qnn.QuantLSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            weight_bit_width=bit_witdh,
            io_bit_width=bit_witdh,
            gate_acc_bit_width=bit_witdh,
            sigmoid_bit_width=bit_witdh,
            tanh_bit_width=bit_witdh,
            weight_quant = SignedTernaryWeightPerTensorConst, 
            io_quant = SignedTernaryActPerTensorConst,
            gate_acc_quant=SignedTernaryActPerTensorConst,
            sigmoid_quant=SignedTernaryActPerTensorConst,
            tanh_quant=SignedTernaryActPerTensorConst,
            return_quant_tensor=True,
            batch_first=True)

        
        self.fc = qnn.QuantLinear(hidden_dim, output_dim, bias=False, weight_quant = SignedTernaryWeightPerTensorConst, weight_bit_width=bit_witdh)
        
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)

        
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        padded_embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        
        padded_output,(hidden_state,cell_state) = self.quantLSTM(padded_embedded)

        dense_outputs=self.fc(hidden_state)

        return dense_outputs


class QLSTM_OnlyLSTM(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(QLSTM_OnlyLSTM,self).__init__()
        
        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        self.quantLSTM = qnn.QuantLSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            weight_bit_width=bit_witdh,
            io_bit_width=bit_witdh,
            gate_acc_bit_width=bit_witdh,
            sigmoid_bit_width=bit_witdh,
            tanh_bit_width=bit_witdh,
            weight_quant = Int8WeightPerTensorFixedPoint, 
            io_quant = Int8ActPerTensorFixedPoint,
            gate_acc_quant=Int8ActPerTensorFixedPoint,
            sigmoid_quant=Uint8ActPerTensorFixedPoint,
            tanh_quant=Int8ActPerTensorFixedPoint,
            return_quant_tensor=True,
            batch_first=True)

        
        self.fc = nn.Linear(hidden_dim, output_dim, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)

        
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        padded_embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        
        padded_output,(hidden_state,cell_state) = self.quantLSTM(padded_embedded)

        dense_outputs=self.fc(hidden_state)

        return dense_outputs



#Class for Brevitas Quantization model with 1bit (QAT)
class QLSTM_OnlyLSTM_1bit(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(QLSTM_OnlyLSTM_1bit,self).__init__()
        
        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        self.quantLSTM = qnn.QuantLSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            weight_bit_width=bit_witdh,
            io_bit_width=bit_witdh,
            gate_acc_bit_width=bit_witdh,
            sigmoid_bit_width=bit_witdh,
            tanh_bit_width=bit_witdh,
            weight_quant = SignedBinaryWeightPerTensorConst, 
            io_quant = SignedBinaryActPerTensorConst,
            gate_acc_quant=SignedBinaryActPerTensorConst,
            sigmoid_quant=SignedBinaryActPerTensorConst,
            tanh_quant=SignedBinaryActPerTensorConst,
            return_quant_tensor=True,
            batch_first=True)

        
        self.fc = nn.Linear(hidden_dim, output_dim, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)

        
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        padded_embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        
        padded_output,(hidden_state,cell_state) = self.quantLSTM(padded_embedded)

        dense_outputs=self.fc(hidden_state)

        return dense_outputs



#Class for Brevitas Quantization model with 2bit (QAT)
class QLSTM_OnlyLSTM_2bit(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(QLSTM_OnlyLSTM_2bit,self).__init__()
        
        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        self.quantLSTM = qnn.QuantLSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            weight_bit_width=bit_witdh,
            io_bit_width=bit_witdh,
            gate_acc_bit_width=bit_witdh,
            sigmoid_bit_width=bit_witdh,
            tanh_bit_width=bit_witdh,
            weight_quant = SignedTernaryWeightPerTensorConst, 
            io_quant = SignedTernaryActPerTensorConst,
            gate_acc_quant=SignedTernaryActPerTensorConst,
            sigmoid_quant=SignedTernaryActPerTensorConst,
            tanh_quant=SignedTernaryActPerTensorConst,
            return_quant_tensor=True,
            batch_first=True)

        
        self.fc = nn.Linear(hidden_dim, output_dim, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)

        
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        padded_embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        
        padded_output,(hidden_state,cell_state) = self.quantLSTM(padded_embedded)

        dense_outputs=self.fc(hidden_state)

        return dense_outputs



class QLSTM_Others(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(QLSTM_Others,self).__init__()
        
        self.embedding = qnn.QuantEmbedding(vocab_size,embedding_dim, weight_quant = Int8WeightPerTensorFixedPoint, weight_bit_width=bit_witdh)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        
        self.fc = qnn.QuantLinear(hidden_dim, output_dim, bias=False, weight_quant = Int8WeightPerTensorFixedPoint, weight_bit_width=bit_witdh)
        
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)

        
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        padded_embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        
        padded_output,(hidden_state,cell_state) = self.lstm(padded_embedded)

        dense_outputs=self.fc(hidden_state)

        return dense_outputs


class QLSTM_Others_1bit(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(QLSTM_Others_1bit,self).__init__()
        
        self.embedding = qnn.QuantEmbedding(vocab_size,embedding_dim, weight_quant = SignedBinaryWeightPerTensorConst, weight_bit_width=bit_witdh)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        
        self.fc = qnn.QuantLinear(hidden_dim, output_dim, bias=False, weight_quant = SignedBinaryWeightPerTensorConst, weight_bit_width=bit_witdh)
        
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)

        
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        padded_embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        
        padded_output,(hidden_state,cell_state) = self.lstm(padded_embedded)

        dense_outputs=self.fc(hidden_state)

        return dense_outputs



#Class for Brevitas Quantization model with 2bit (QAT)
class QLSTM_Others_2bit(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(QLSTM_Others_2bit,self).__init__()
        
        self.embedding = qnn.QuantEmbedding(vocab_size,embedding_dim, weight_quant = SignedTernaryWeightPerTensorConst, weight_bit_width=bit_witdh)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
      
        self.fc = qnn.QuantLinear(hidden_dim, output_dim, bias=False, weight_quant = SignedTernaryWeightPerTensorConst, weight_bit_width=bit_witdh)
        
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)

        
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        padded_embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        
        padded_output,(hidden_state,cell_state) = self.lstm(padded_embedded)

        dense_outputs=self.fc(hidden_state)

        return dense_outputs


class RNNNet(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(RNNNet,self).__init__()
        
        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)

        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True)
        #padded_embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        
        packed_output,hidden_state = self.rnn(packed_embedded)

        hidden = hidden_state[-1,:,:]
        
        dense_outputs=self.fc(hidden)

        return dense_outputs


class QRNN(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(QRNN,self).__init__()
        
        self.embedding = qnn.QuantEmbedding(vocab_size, embedding_dim, weight_quant = Int8WeightPerTensorFixedPoint, weight_bit_width=bit_witdh)

        self.qrnn = qnn.QuantRNN(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            weight_bit_width=bit_witdh,
            io_bit_width=bit_witdh,
            gate_acc_bit_width=bit_witdh,
            sigmoid_bit_width=bit_witdh,
            tanh_bit_width=bit_witdh,
            weight_quant = Int8WeightPerTensorFixedPoint, 
            io_quant = Int8ActPerTensorFixedPoint,
            gate_acc_quant=Int8ActPerTensorFixedPoint,
            sigmoid_quant=Uint8ActPerTensorFixedPoint,
            tanh_quant=Int8ActPerTensorFixedPoint,
            return_quant_tensor=True,
            batch_first=True)
        
        self.fc = qnn.QuantLinear(hidden_dim, output_dim, bias=False, weight_quant = Int8WeightPerTensorFixedPoint, weight_bit_width=bit_witdh)
        
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)

        padded_embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        
        packed_output,hidden_state = self.qrnn(padded_embedded)
        
        dense_outputs=self.fc(hidden_state)

        return dense_outputs


class QRNN_2bit(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(QRNN_2bit,self).__init__()
        
        self.embedding = qnn.QuantEmbedding(vocab_size, embedding_dim, weight_quant = Int8WeightPerTensorFixedPoint, weight_bit_width=bit_witdh)

        self.qrnn = qnn.QuantRNN(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            weight_bit_width=bit_witdh,
            io_bit_width=bit_witdh,
            gate_acc_bit_width=bit_witdh,
            sigmoid_bit_width=bit_witdh,
            tanh_bit_width=bit_witdh,
            weight_quant = SignedTernaryWeightPerTensorConst, 
            io_quant = SignedTernaryActPerTensorConst,
            gate_acc_quant=SignedTernaryActPerTensorConst,
            sigmoid_quant=SignedTernaryActPerTensorConst,
            tanh_quant=SignedTernaryActPerTensorConst,
            return_quant_tensor=True,
            batch_first=True)
        
        self.fc = qnn.QuantLinear(hidden_dim, output_dim, bias=False, weight_quant = SignedTernaryWeightPerTensorConst, weight_bit_width=bit_witdh)
        
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)

        padded_embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        
        packed_output,hidden_state = self.qrnn(padded_embedded)
        
        dense_outputs=self.fc(hidden_state)

        return dense_outputs


class QRNN_1bit(nn.Module):
    
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout, bit_witdh):
        
        super(QRNN_1bit,self).__init__()
        
        self.embedding = qnn.QuantEmbedding(vocab_size, embedding_dim, weight_quant = Int8WeightPerTensorFixedPoint, weight_bit_width=bit_witdh)

        self.qrnn = qnn.QuantRNN(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            weight_bit_width=bit_witdh,
            io_bit_width=bit_witdh,
            gate_acc_bit_width=bit_witdh,
            sigmoid_bit_width=bit_witdh,
            tanh_bit_width=bit_witdh,
            weight_quant = SignedBinaryWeightPerTensorConst, 
            io_quant = SignedBinaryActPerTensorConst,
            gate_acc_quant=SignedBinaryActPerTensorConst,
            sigmoid_quant=SignedBinaryActPerTensorConst,
            tanh_quant=SignedBinaryActPerTensorConst,
            return_quant_tensor=True,
            batch_first=True)
        
        self.fc = qnn.QuantLinear(hidden_dim, output_dim, bias=False, weight_quant = SignedBinaryWeightPerTensorConst, weight_bit_width=bit_witdh)
        
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self,text,text_lengths):
        embedded = self.embedding(text)

        padded_embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        
        packed_output,hidden_state = self.qrnn(padded_embedded)
        
        dense_outputs=self.fc(hidden_state)

        return dense_outputs