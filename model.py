import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # initializing Decoder fields
        self.embedding_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout=0
        
        # defining the embedding, LSTM, linear and activation
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, self.num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights."""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)

    def forward(self, features, captions):
        # batch_size is the first dim, since batch_first=true
        batch_size = features.size(0)
        # stripping out the <end> tag 
        captions = captions[:, :-1]
        embedding = self.embedding(captions)
       
        features = features.unsqueeze(1)
        # concatenate the image features and word embeddings.
        lstm_input = torch.cat((features, embedding), 1)
        lstm_output, _ = self.lstm(lstm_input)
        
        # Convert LSTM outputs to word predictions
        outputs = self.fc(lstm_output)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_sentence = []
        lstm_prev_state = None

        for i in range(max_len):
            lstm_output, lstm_prev_state = self.lstm(inputs, lstm_prev_state)
            fc_output = self.fc(lstm_output)

            # Get the prediction
            predicted_output = torch.argmax(fc_output, dim=2)
            predicted_index = predicted_output.item()
            predicted_sentence.append(predicted_index)

            # Getting the embedding for the next execution
            inputs = self.embedding(predicted_output)

        return predicted_sentence