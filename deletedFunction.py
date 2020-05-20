# class Attention(torch.nn.Module):
#   def __init__(self, hidden_size, method="dot"):
#     super(Attention, self).__init__()
#     self.method = method
#     self.hidden_size = hidden_size
    
#     #https://blog.floydhub.com/attention-mechanism/
#     # Defining the layers/weights required depending on alignment scoring method
#     if method == "general":
#       self.fc = torch.nn.Linear(hidden_size, hidden_size, bias=False)
      
#     elif method == "concat":
#       self.fc = torch.nn.Linear(hidden_size, hidden_size, bias=False)
#       self.weight = torch.nn.Parameter(torch.FloatTensor(1, hidden_size))
  
#   def forward(self, encoder_outputs, decoder_hidden):
#     #encoder_outputs [batch_size, seq_len, hidden_dim]
#     #decoder_hidden [batch_size, hidden_dim, seq_len]
#     #return context [batch_size, seq_len, hidden_dim]

#     if self.method == "dot":
#       # For the dot scoring method, no weights or linear layers are involved
#       sim_matrix = encoder_outputs.bmm(decoder_hidden)
#       sim_matrix = torch.nn.functional.softmax(sim_matrix, dim=2)
      
#       def context_fun(encoder_outputs, sim_matrix, index):
#           context_vec = encoder_outputs * sim_matrix[:, :, index].unsqueeze(2)
#           return context_vec.sum(dim=1).unsqueeze(1)
      
#       return torch.cat([context_fun(encoder_outputs, sim_matrix, i) \
#                     for i in range(sim_matrix.size(2))], dim=1)

    #elif self.method == "general":
      # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
    #  out = self.fc(decoder_hidden)
    #  return encoder_outputs.bmm(out.view(1,-1,1)).squeeze(-1)
    
    #elif self.method == "concat":
      # For concat scoring, decoder hidden state and encoder outputs are concatenated first
    #  out = torch.tanh(self.fc(decoder_hidden+encoder_outputs))
    #  return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)