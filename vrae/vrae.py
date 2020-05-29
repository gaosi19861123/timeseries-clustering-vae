import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from .base import BaseEstimator
from torch.utils.data import DataLoader
from torch.autograd import Variable
from .utils import *
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from torch.nn import functional as F


class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """
    def __init__(self, batch_size, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout, bidirectional=True)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout, bidirectional=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder

        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """

        enc_out, (h_end, c_end) = self.model(x)
        #print("encoderデータサイズ",enc_out.size())
        return torch.sum(h_end, dim=0), enc_out


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        
        else:
            return self.latent_mean

class selfAttention(torch.nn.Module):

  def __init__(self, method="dot"):
    super(selfAttention, self).__init__()
    self.method = method
    #self.hidden_size = hidden_size

  def forward(self, encoder_outputs):
    #encoder_outputs [batch_size, seq_len, hidden_dim]
    #decoder_hidden [batch_size, hidden_dim, seq_len]
    #return context [batch_size, seq_len, hidden_dim]

    if self.method == "dot":
      # For the dot scoring method, no weights or linear layers are involved
      scale_size = encoder_outputs.size(2)
      sim_matrix = encoder_outputs.permute(1, 0, 2).bmm(encoder_outputs.permute(1, 2, 0))
      sim_matrix = sim_matrix / np.sqrt(scale_size)
      sim_matrix = torch.nn.functional.softmax(sim_matrix, dim=2)
      
      def context_fun(encoder_outputs, sim_matrix, index):
          context_vec = encoder_outputs.permute(1, 0, 2) * sim_matrix[:, index, :].unsqueeze(2)
          return context_vec.sum(dim=1).unsqueeze(1)
      
      return torch.cat([context_fun(encoder_outputs, sim_matrix, i) \
                    for i in range(sim_matrix.size(2))], dim=1)

class Decoder(nn.Module):
    """Converts latent vector into output

    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, block='LSTM', lam = Lambda):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype
        self.attention = selfAttention() #Attention(100)
        self.lam = lam(2 * hidden_size, latent_length)

        if block == 'LSTM':
            self.model = nn.LSTM(latent_length * 2, self.hidden_size, self.hidden_layer_depth, bidirectional=True)
        elif block == 'GRU':
            self.model = nn.GRU(latent_length * 2, self.hidden_size, self.hidden_layer_depth, bidirectional=True)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length,  self.hidden_size)
        self.hidden_to_myu = nn.Linear(2 * self.hidden_size, self.output_size)
        self.hidden_to_scale = nn.Linear(2 * self.hidden_size, self.output_size)

        #[sequence_length, batch_size, feature_size] 
        #self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.dtype)
        self.c_0 = torch.zeros(2 * self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(self.dtype)
        self.h_0 = torch.zeros(2 * self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(self.dtype)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_myu.weight)
        nn.init.xavier_uniform_(self.hidden_to_scale.weight)

    def forward(self, latent, encoder_output, need_attention=True):
        """Converts latent to hidden to output

        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        #print(latent.size())
        #h_state = self.latent_to_hidden(latent)
        #h_state.unsqueeze_(dim=1)
        latent_repeated = torch.repeat_interleave(latent.unsqueeze_(dim=1), self.sequence_length, dim=1)
        #print("latent_repeat:", latent_repeated.size())

        if need_attention:
            #context_vec = self.attention(encoder_output.permute(1, 0, 2), \
            #                        decoder_output.permute(1, 2, 0))
            #out = context_vec.permute(1, 0, 2) + decoder_output
            context_vec = self.attention(encoder_output)
            decoder_Input = self.lam(context_vec)
            #print("decoder_size:", decoder_Input.size())
            
            decoder_conbinded = torch.cat((decoder_Input, latent_repeated), dim=2) 

        if isinstance(self.model, nn.LSTM):
            #h_0 = torch.stack([h_state for _ in range(2*self.hidden_layer_depth)])
            decoder_output, _ = self.model(decoder_conbinded.permute(1, 0, 2), (self.h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            #h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(decoder_conbinded.permute(1, 0, 2), self.h_0)
        else:
            raise NotImplementedError

        #print("decoder_output:", decoder_output.size())

        myu = self.hidden_to_myu(decoder_output)
        scale = self.hidden_to_scale(decoder_output)
        scale = F.softplus(scale)
        #print("output.size:", myu.size())
        return myu, self.lam.latent_mean, self.lam.latent_logvar, scale

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

class LaplicanLoss(torch.nn.Module):
    def __init__(self, size_average=True):
        super(LaplicanLoss, self).__init__()
        self.size_average = size_average
    
    def forward(self, x, myu, scale):

        prob = -torch.abs(x - myu) / scale + torch.log(1 / (2. * scale))
        
        if self.size_average: 
            log_prob = torch.mean(torch.sum(prob, dim=0))
        else: 
            log_prob = torch.sum(torch.sum(prob, dim=0))

        return log_prob

class VRAE(BaseEstimator, nn.Module):
    """Variational recurrent auto-encoder. This module is used for dimensionality reduction of timeseries

    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size:  hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param learning_rate: the learning rate of the module
    :param block: GRU/LSTM to be used as a basic building block
    :param n_epochs: Number of iterations/epochs
    :param dropout_rate: The probability of a node being dropped-out
    :param optimizer: ADAM/ SGD optimizer to reduce the loss function
    :param loss: SmoothL1Loss / MSELoss / ReconLoss / any custom loss which inherits from `_Loss` class
    :param boolean cuda: to be run on GPU or not
    :param print_every: The number of iterations after which loss should be printed
    :param boolean clip: Gradient clipping to overcome explosion
    :param max_grad_norm: The grad-norm to be clipped
    :param dload: Download directory where models are to be dumped
    """
    def __init__(self, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, block='LSTM',
                 n_epochs=5, dropout_rate=0., optimizer='Adam', loss='MSELoss',
                 cuda=False, print_every=100, clip=True, max_grad_norm=5, dload='.',
                 k2=0.01):

        super(VRAE, self).__init__()


        self.dtype = torch.FloatTensor
        self.use_cuda = cuda
        self.total_epoch = 0

        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False

        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor


        self.encoder = Encoder(batch_size=batch_size,
                               number_of_features = number_of_features,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               dropout=dropout_rate,
                               block=block)

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=latent_length)

        self.decoder = Decoder(sequence_length=sequence_length,
                               batch_size = batch_size,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               output_size=number_of_features,
                               block=block,
                               dtype=self.dtype)
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.print_every = print_every
        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.is_fitted = False
        self.dload = dload
        self.loss_type = loss
        self.k2 = k2

        if self.use_cuda:
            self.cuda()

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError('Not a recognized optimizer')

        if loss == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss(size_average=False)
        elif loss == 'MSELoss':
            self.loss_fn = nn.MSELoss(size_average=False)
        elif loss == 'LaplacianLoss':
            self.loss_fn = LaplicanLoss(size_average=False) 

    def __repr__(self):
        return """VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                cuda=self.use_cuda)

    def forward(self, x):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder

        :param x:input tensor
        :return: the decoded output, latent vector
        """
        cell_output, enc_output = self.encoder(x)
        latent = self.lmbd(cell_output)
        x_decoded, self.T_mean, self.T_var, scale = self.decoder(latent, enc_output)
        #print("T_mean_size:{0}, T_var_size:{1}".format(self.T_mean.size(), self.T_var.size()))
        return x_decoded, latent, scale

    def _rec(self, x_decoded, x, loss_fn, scale, k1, niter):
        """
        Compute the loss given output x decoded, input x and the specified loss function

        :param x_decoded: output of the decoder
        :param x: input to the encoder
        :param loss_fn: loss function specified
        :return: joint loss, reconstruction loss and kl-divergence loss
        """
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar
        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        TK1_loss = -0.5 * torch.mean(torch.sum(1 + self.T_var - self.T_mean.pow(2) - self.T_var.exp(), dim=1))
        
        if (self.loss_type == "MSELoss") or (self.loss_type == "SmoothL1Loss"):
            recon_loss = loss_fn(x_decoded, x)

        elif self.loss_type == "LaplacianLoss":
            recon_loss = -loss_fn(x, x_decoded, scale)

        return recon_loss + k1[niter] * kl_loss + self.k2 * TK1_loss, recon_loss, k1[niter] * kl_loss, self.k2 * TK1_loss

    def compute_loss(self, X, k1, niter):
        """
        Given input tensor, forward propagate, compute the loss, and backward propagate.
        Represents the lifecycle of a single iteration

        :param X: Input tensor
        :return: total loss, reconstruction loss, kl-divergence loss and original input
        """
        x = Variable(X[:,:,:].type(self.dtype), requires_grad = True)

        x_decoded, _, scale = self(x)
        loss, recon_loss, kl_loss, TK1_loss = self._rec(x_decoded, x.detach(), self.loss_fn, scale, k1=k1, niter=niter)

        return loss, recon_loss, kl_loss, TK1_loss, x


    def _train(self, train_loader, vis):
        """
        For each epoch, given the batch_size, run this function batch_size * num_of_batches number of times

        :param train_loader:input train loader with shuffle
        :return:
        """
        self.train()
        k1 = frange_cycle_linear(train_loader.__len__() * self.n_epochs)

        epoch_loss = 0
        t = 0

        for t, X in enumerate(train_loader):

            # Index first element of array to return tensor
            X = X[0]

            # required to swap axes, since dataloader gives output in (batch_size x seq_len x num_of_features)
            X = X.permute(1,0,2)

            self.optimizer.zero_grad()
            loss, recon_loss, kl_loss, TK1_loss, _ = self.compute_loss(X, k1, self.total_epoch)
            loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)
            
            self.total_epoch += 1
            # accumulator
            epoch_loss += loss.item()
            self.optimizer.step()

            #visulization_train_process
            visulization(vis, "line", X=self.total_epoch, Y=loss.item(), win_name="all")
            visulization(vis, "line", X=self.total_epoch, Y=recon_loss.item(), win_name="recon")  
            visulization(vis, "line", X=self.total_epoch, Y=kl_loss.item(), win_name="k1") 
            visulization(vis, "line", X=self.total_epoch, Y=TK1_loss.item(), win_name="Tk1") 

            if (t + 1) % self.print_every == 0:
                print('Batch %d, loss = %.4f, recon_loss = %.4f, kl_loss = %.4f, TK1_loss = %.4f' % (t + 1, loss.item(),
                                                                                    recon_loss.item(), kl_loss.item(), TK1_loss.item()))

        print('Average loss: {:.4f}'.format(epoch_loss / t))


    def fit(self, train_dataset, val_dataset, val_label, env = "vrae", save = False):
        """
        Calls `_train` function over a fixed number of epochs, specified by `n_epochs`

        :param dataset: `Dataset` object
        :param bool save: If true, dumps the trained model parameters as pickle file at `dload` directory
        :return:
        """

        vis = visdom.Visdom(env=env, port=8097)
        train_loader = DataLoader(dataset = train_dataset,
                                  batch_size = self.batch_size,
                                  shuffle = True,
                                  drop_last=True)

        for i in range(self.n_epochs):
            print('Epoch: %s' % i)

            self._train(train_loader, vis)

        self.is_fitted = True
        if save:
            self.save('model.pth')


    def _batch_transform(self, x):
        """
        Passes the given input tensor into encoder and lambda function

        :param x: input batch tensor
        :return: intermediate latent vector
        """
        return self.lmbd(
                    self.encoder(
                        Variable(x.type(self.dtype), requires_grad = False)
                    )[0]
        ).cpu().data.numpy()

    def _batch_reconstruct(self, x):
        """
        Passes the given input tensor into encoder, lambda and decoder function

        :param x: input batch tensor
        :return: reconstructed output tensor
        """

        x = Variable(x.type(self.dtype), requires_grad = False)
        x_decoded, _ = self(x)

        return x_decoded.cpu().data.numpy()

    def reconstruct(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_reconstruct`
        Prerequisite is that model has to be fit

        :param dataset: input dataset who's output vectors are to be obtained
        :param bool save: If true, dumps the output vector dataframe as a pickle file
        :return:
        """

        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader

        if self.is_fitted:
            with torch.no_grad():
                x_decoded = []

                for t, x in enumerate(test_loader):
                    x = x[0]
                    x = x.permute(1, 0, 2)

                    x_decoded_each = self._batch_reconstruct(x)
                    x_decoded.append(x_decoded_each)

                x_decoded = np.concatenate(x_decoded, axis=1)

                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    x_decoded.dump(self.dload + '/z_run.pkl')
                return x_decoded

        raise RuntimeError('Model needs to be fit')


    def transform(self, dataset, for_visual=False, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_transform`
        Prerequisite is that model has to be fit

        :param dataset: input dataset who's latent vectors are to be obtained
        :param bool save: If true, dumps the latent vector dataframe as a pickle file
        :return:
        """
        self.eval()
        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last = True) # Don't shuffle for test_loader

        if (self.is_fitted) or (for_visual):
            with torch.no_grad():
                z_run = []
                T_mean = []

                for t, x in enumerate(test_loader):
                    x = x[0]
                    x = x.permute(1, 0, 2)

                    z_run_each = self._batch_transform(x)
                    z_run.append(z_run_each)
                    T_mean.append(self.T_mean.cpu().detach().numpy())

                z_run = np.concatenate(z_run, axis=0)
                T_mean = np.concatenate(T_mean, axis=0)
                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    z_run.dump(self.dload + '/z_run.pkl')
                return z_run, T_mean

        raise RuntimeError('Model needs to be fit')

    def fit_transform(self, dataset, save = False):
        """
        Combines the `fit` and `transform` functions above

        :param dataset: Dataset on which fit and transform have to be performed
        :param bool save: If true, dumps the model and latent vectors as pickle file
        :return: latent vectors for input dataset
        """
        self.fit(dataset, save = save)
        return self.transform(dataset, save = save)

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later

        :param file_name: the filename to be saved as,`dload` serves as the download directory
        :return: None
        """
        PATH = self.dload + '/' + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned

        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))