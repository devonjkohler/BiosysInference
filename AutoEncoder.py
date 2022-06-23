
import numpy as np

import torch
import torch.nn as nn

import copy
import pickle

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=6,
            batch_first=True
        )
        self.rnn3 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        # x = x.reshape((1, self.seq_len, self.n_features))
        # print(x.shape)
        x, (_, _) = self.rnn1(x)
        # print(x.shape)
        x, (_, _) = self.rnn2(x)
        x, (hidden_n, _) = self.rnn3(x)
        # print(x.shape)
        # print(hidden_n.shape)
        return hidden_n.squeeze()


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.hidden_dim, self.n_features, self.input_dim = 2 * input_dim, n_features, input_dim
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=7,
            batch_first=True
        )
        self.output_layer = nn.Linear(input_dim, n_features)

    def forward(self, x):
        # print(x.shape)
        # print(self.seq_len)
        x = x.repeat(1, self.seq_len)
        # print(x.shape)
        x = x.reshape((x.shape[0], self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        # x = x.reshape((self.seq_len, self.n_features))
        # print(x.shape)
        return self.output_layer(x)


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_model(model, train_dataset, val_dataset, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.HuberLoss()

    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        for i, data in enumerate(train_dataset):
            optimizer.zero_grad()
            seq_pred = model(data.cuda())
            loss = criterion(seq_pred, data.cuda())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dataset):
            seq_pred = model(data.cuda())
            loss = criterion(seq_pred, data.cuda())
            val_losses.append(loss.item())
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    history['train'].append(train_loss)
    history['val'].append(val_loss)
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
    # print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    model.load_state_dict(best_model_wts)
    return model.eval(), history

def main():

    ## Load data
    with open('/home/kohler.d/biosim_project/BiosysInference/data/LotkaVolterraObs.pickle', 'rb') as handle:
        x = pickle.load(handle)

    ## Normalize and prep input
    v0_min = x[:, 0].min()
    v0_max = x[:, 0].max()
    x[:, 0] = (x[:, 0] - v0_min) / (v0_max - v0_min)

    v1_min = x[:, 1].min()
    v1_max = x[:, 1].max()
    x[:, 1] = (x[:, 1] - v1_min) / (v1_max - v1_min)

    v2_min = x[:, 2].min()
    v2_max = x[:, 2].max()
    x[:, 2] = (x[:, 2] - v2_min) / (v2_max - v2_min)

    x = x.reshape((10000, 1000, 3))
    train_x = x[:9000]
    val_x = x[9001:]
    n_seq, seq_len, n_features = train_x.shape

    ## Create Loaders
    training_loader = torch.utils.data.DataLoader(train_x, batch_size=32, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(val_x, batch_size=32, shuffle=False, num_workers=2)

    model = RecurrentAutoencoder(seq_len, n_features, 128).cuda()

    model, history = train_model(
        model,
        training_loader,
        validation_loader,
        n_epochs=20,
    )

    torch.save(model.state_dict(), "/scratch/kohler.d/encoder_model.pt")
    with open(r'/scratch/kohler.d/model_history.pickle', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
