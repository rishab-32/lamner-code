import torch
from rouge import FilesRouge
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
import math
import time
from torchtext import data
import torchtext.vocab as vocab
import nltk
from rouge import Rouge
from six.moves import map


def print_log(text):
    
  with open("shell-log-seq2seq-mom.txt", "a") as f:
    f.write(text+"\n")
  return




class Encoder(nn.Module):
  def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
    super().__init__()
    
    self.embedding = nn.Embedding(input_dim, emb_dim)
    
    self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
    
    self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
    
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, src, src_len):
      
    embedded = self.dropout(self.embedding(src))
    
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)
            
    packed_outputs, hidden = self.rnn(packed_embedded)
                              
    outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
        
    hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
    
    return outputs, hidden


class Attention(nn.Module):
  def __init__(self, enc_hid_dim, dec_hid_dim):
    super().__init__()
    
    self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
    self.v = nn.Linear(dec_hid_dim, 1, bias = False)
      
  def forward(self, hidden, encoder_outputs, mask):
      
    batch_size = encoder_outputs.shape[1]
    src_len = encoder_outputs.shape[0]
    
    hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

    encoder_outputs = encoder_outputs.permute(1, 0, 2)
    
    energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
    
    attention = self.v(energy).squeeze(2)
    
    attention = attention.masked_fill(mask == 0, -1e10)
    
    return F.softmax(attention, dim = 1)


class Decoder(nn.Module):
  def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
    super().__init__()

    self.output_dim = output_dim
    self.attention = attention
    
    self.embedding = nn.Embedding(output_dim, emb_dim)
    
    self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
    
    self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
    
    self.dropout = nn.Dropout(dropout)
      
  def forward(self, input, hidden, encoder_outputs, mask):
            
    input = input.unsqueeze(0)
    
    embedded = self.dropout(self.embedding(input))
    
    a = self.attention(hidden, encoder_outputs, mask)
            
    a = a.unsqueeze(1)
    
    encoder_outputs = encoder_outputs.permute(1, 0, 2)
    
    weighted = torch.bmm(a, encoder_outputs)
    
    weighted = weighted.permute(1, 0, 2)
    
    rnn_input = torch.cat((embedded, weighted), dim = 2)
    
    output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
    
    assert (output == hidden).all()
    
    embedded = embedded.squeeze(0)
    output = output.squeeze(0)
    weighted = weighted.squeeze(0)
    
    prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
    
    return prediction, hidden.squeeze(0), a.squeeze(1)
	
class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, src_pad_idx, device):
    super().__init__()
    
    self.encoder = encoder
    self.decoder = decoder
    self.src_pad_idx = src_pad_idx
    self.device = device
      
  def create_mask(self, src):
    mask = (src != self.src_pad_idx).permute(1, 0)
    return mask
      
  def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
                
    batch_size = src.shape[1]
    trg_len = trg.shape[0]
    trg_vocab_size = self.decoder.output_dim
  
    outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
    
    encoder_outputs, hidden = self.encoder(src, src_len)
            
    input = trg[0,:]
    
    mask = self.create_mask(src)

    
    for t in range(1, trg_len):
        
      
      output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
      
      
      outputs[t] = output
      
      
      teacher_force = random.random() < teacher_forcing_ratio
      
      
      top1 = output.argmax(1) 
      
    
      input = trg[t] if teacher_force else top1
        
    return outputs

def init_weights(m):
  for name, param in m.named_parameters():
    if 'weight' in name:
      nn.init.normal_(param.data, mean=0, std=0.01)
    else:
      nn.init.constant_(param.data, 0)

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
def train(model, iterator, optimizer, criterion, clip):
    
  model.train()
  
  epoch_loss = 0
  
  for i, batch in enumerate(iterator):
      
    src, src_len = batch.code
    trg = batch.summary
    
    optimizer.zero_grad()
    
    output = model(src, src_len, trg)
    
    #trg = [trg len, batch size]
    #output = [trg len, batch size, output dim]
    
    output_dim = output.shape[-1]
    
    output = output[1:].view(-1, output_dim)
    trg = trg[1:].view(-1)
    
    #trg = [(trg len - 1) * batch size]
    #output = [(trg len - 1) * batch size, output dim]
    
    loss = criterion(output, trg)
    
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    
    optimizer.step()
    
    epoch_loss += loss.item()
      
  return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    
  model.eval()
  
  epoch_loss = 0
  
  with torch.no_grad():
  
    for i, batch in enumerate(iterator):

      src, src_len = batch.code
      trg = batch.summary

      output = model(src, src_len, trg, 0) #turn off teacher forcing
      
      #trg = [trg len, batch size]
      #output = [trg len, batch size, output dim]

      output_dim = output.shape[-1]
      
      output = output[1:].view(-1, output_dim)
      trg = trg[1:].view(-1)

      #trg = [(trg len - 1) * batch size]
      #output = [(trg len - 1) * batch size, output dim]

      loss = criterion(output, trg)

      epoch_loss += loss.item()
      
  return epoch_loss / len(iterator)
  
def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 64):

  model.eval()
  tokens = [token.lower() for token in sentence]
    

  tokens = [src_field.init_token] + tokens + [src_field.eos_token]
      
  src_indexes = [src_field.vocab.stoi[token] for token in tokens]
  
  src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

  src_len = torch.LongTensor([len(src_indexes)]).to(device)
  
  with torch.no_grad():
    encoder_outputs, hidden = model.encoder(src_tensor, src_len)

  mask = model.create_mask(src_tensor)
      
  trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

  attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
  
  for i in range(max_len):

    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            
    with torch.no_grad():
      output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

    attentions[i] = attention
        
    pred_token = output.argmax(1).item()
    
    trg_indexes.append(pred_token)

    if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
      break
  
  trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
  
  return trg_tokens[1:], attentions[:len(trg_tokens)-1]
  
def get_preds(data, src_field, trg_field, model, device, max_len = 64):
    
  trgs = []
  pred_trgs = []
  
  for datum in data:
    p = ""
    t= ""
    src = vars(datum)['code']
    trg = vars(datum)['summary']
    
    pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
    
    #cut off <eos> token
    pred_trg = pred_trg[:-1]
    p = " ".join(pred_trg)
    p = p.strip()

    t = " ".join(trg)
    t = t.strip()

    pred_trgs.append(p)
    trgs.append(t)
      
  return pred_trgs,trgs
  
def write_files(p,t,epoch, test=False, Warmup=False):
  predicted_file_name = "predictions.out-"+str(epoch)+".txt"
  ref_file_name = "trgs.given-"+str(epoch)+".txt"
  
  if test:

    predicted_file_name = "test-predictions.out.txt"
    ref_file_name = "test-trgs.given.txt"
    with open(predicted_file_name, "w", encoding="utf-8") as f:
      for i in p:
        f.write(i+"\n")
  
    with open(ref_file_name, "w", encoding="utf-8") as f:
      for i in t:
        f.write(i+"\n")

  elif Warmup:
    predicted_file_name = "warm-predictions.out-"+str(epoch)+".txt"
    ref_file_name = "warm-trgs.given-"+str(epoch)+".txt"
    with open(predicted_file_name, "w", encoding="utf-8") as f:
      for i in p:
        f.write(i+"\n")
  
    with open(ref_file_name, "w", encoding="utf-8") as f:
      for i in t:
        f.write(i+"\n")


  else:
    with open(predicted_file_name, "w", encoding="utf-8") as f:
      for i in p:
        f.write(i+"\n")
    
    with open(ref_file_name, "w", encoding="utf-8") as f:
      for i in t:
        f.write(i+"\n")
  
  return
  
def calculate_rouge(epoch,test=False, Warmup=False):

  if test:

    predicted_file_name = "test-predictions.out.txt"
    ref_file_name = "test-trgs.given.txt"
    
    
  elif Warmup:
    predicted_file_name = "warm-predictions.out-"+str(epoch)+".txt"
    ref_file_name = "warm-trgs.given-"+str(epoch)+".txt"
  
  else:
    predicted_file_name = "predictions.out-"+str(epoch)+".txt"
    ref_file_name = "trgs.given-"+str(epoch)+".txt"

  
   
  files_rouge = FilesRouge()
  rouge = files_rouge.get_scores(
          hyp_path=predicted_file_name, ref_path=ref_file_name, avg=True, ignore_empty=True)
  return round(rouge['rouge-l']["f"]*100, 2)
  

def get_max_lens(train_data, test_data, valid_data, code=True):
  
  encoder_max = -1

  if code:
    for i in range(len(train_data)):
      if encoder_max< len(vars(train_data.examples[i])["code"]):
        encoder_max = len(vars(train_data.examples[i])["code"])

    for i in range(len(test_data)):
      if encoder_max< len(vars(test_data.examples[i])["code"]):
        encoder_max = len(vars(test_data.examples[i])["code"])

    for i in range(len(valid_data)):
      if encoder_max< len(vars(valid_data.examples[i])["code"]):
        encoder_max = len(vars(valid_data.examples[i])["code"])

  else:
    for i in range(len(train_data)):
      if encoder_max< len(vars(train_data.examples[i])["summary"]):
        encoder_max = len(vars(train_data.examples[i])["summary"])

    for i in range(len(test_data)):
      if encoder_max< len(vars(test_data.examples[i])["summary"]):
        encoder_max = len(vars(test_data.examples[i])["summary"])

    for i in range(len(valid_data)):
      if encoder_max< len(vars(valid_data.examples[i])["summary"]):
        encoder_max = len(vars(valid_data.examples[i])["summary"])
  return encoder_max
  
def main():

  SEED = 1234

  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True


  BATCH_SIZE = 16
  ENC_EMB_DIM = 512
  DEC_EMB_DIM = 512
  ENC_HID_DIM = 512
  DEC_HID_DIM = 512
  ENC_DROPOUT = 0.5
  DEC_DROPOUT = 0.5
  N_EPOCHS = 200
  CLIP = 1
  best_valid_loss = float('inf')
  cur_rouge = -float('inf')
  best_rouge = -float('inf')
  best_epoch = -1
  make_weights_static = False
  MIN_LR = 0.0000001
  MAX_VOCAB_SIZE = 50_000
  LEARNING_RATE = 0.1
  early_stop = False
  cur_lr = LEARNING_RATE
  num_of_epochs_not_improved = 0


  SRC = Field(init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            include_lengths = True)

  TRG = Field(init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)
  train_data, valid_data, test_data = data.TabularDataset.splits(
          path='./data_seq2seq/', train='train_seq.csv',
          skip_header=True,
          validation='valid_seq.csv', test='test_seq.csv', format='CSV',
          fields=[('code', SRC), ('summary', TRG)])


  custom_embeddings_semantic_encoder = vocab.Vectors(name = 'custom_embeddings/semantic_embeds.txt',
                                      cache = 'custom_embeddings_semantic_encoder',
                                      unk_init = torch.Tensor.normal_)
    
  custom_embeddings_syntax_encoder = vocab.Vectors(name = 'custom_embeddings/syntax_embeds.txt',
                                      cache = 'custom_embeddings_syntax_encoder',
                                      unk_init = torch.Tensor.normal_)
    
  custom_embeddings_decoder = vocab.Vectors(name = 'custom_embeddings/decoder_embeddings.txt',
                                      cache = 'custom_embeddings_decoder',
                                      unk_init = torch.Tensor.normal_)
  SRC.build_vocab(train_data, 
                     max_size = MAX_VOCAB_SIZE, 
                     vectors = custom_embeddings_semantic_encoder
                   )
    
    
  TRG.build_vocab(train_data, 
                     max_size = MAX_VOCAB_SIZE, 
                     vectors = custom_embeddings_decoder
                   )
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
          (train_data, valid_data, test_data), 
          batch_size = BATCH_SIZE,
          sort_within_batch = True,
          shuffle=True,
          sort_key = lambda x : len(x.code),
          device = device)

  INPUT_DIM = len(SRC.vocab)
  OUTPUT_DIM = len(TRG.vocab)
  SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
  attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
  enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
  dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
  model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)
  model.apply(init_weights)
  SRC.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = custom_embeddings_semantic_encoder
               )

  embeddings_enc1 = SRC.vocab.vectors

  SRC.build_vocab(train_data, 
				 max_size = MAX_VOCAB_SIZE, 
				 vectors = custom_embeddings_syntax_encoder
			   )

  embeddings_enc2 = SRC.vocab.vectors

  embeddings_enc3 = torch.cat([embeddings_enc1, embeddings_enc2], dim=1)

  model.encoder.embedding.weight.data.copy_(embeddings_enc3)

  embeddings_trg = TRG.vocab.vectors
  model.decoder.embedding.weight.data.copy_(embeddings_trg)

  del embeddings_trg, embeddings_enc1, embeddings_enc3, embeddings_enc2
  optimizer = optim.SGD(model.parameters(),lr=LEARNING_RATE, momentum=0.9)
  TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
  criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

  cd_len = get_max_lens(train_data, test_data, valid_data, code=True)
  sm_len = get_max_lens(train_data, test_data, valid_data, code=False)

  print_log("Maximum Input length is " + str(cd_len) + "... Maximum Output Length is " + str(sm_len))
  print_log("Encoder Vocab Size " + str(INPUT_DIM) + "... Decoder Vocab Size " + str(OUTPUT_DIM))
  #print_log(model)
  print_log('The model has ' + str(count_parameters(model))+  ' trainable parameters')



  print_log("\nTraining Started.....")
  cur_lr = LEARNING_RATE
  optimizer.param_groups[0]['lr'] = cur_lr


  for epoch in range(N_EPOCHS):
    if MIN_LR>optimizer.param_groups[0]['lr']:
      early_stop = True
      break

    if num_of_epochs_not_improved==7:
      #reduce LR
      model.load_state_dict(torch.load('best-seq2seq.pt'))
      optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
      num_of_epochs_not_improved = 0
      
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    p, t = get_preds(valid_data, SRC, TRG, model, device)
    write_files(p,t,epoch+1)
    cur_rouge = calculate_rouge(epoch+1)
    torch.save(model.state_dict(), 'seq2seq-'+str(epoch+1)+'.pt')

    if best_valid_loss>valid_loss:
      best_valid_loss = valid_loss
      best_epoch = epoch + 1
      num_of_epochs_not_improved = 0
    else:
      num_of_epochs_not_improved = num_of_epochs_not_improved + 1 
    
    if cur_rouge > best_rouge:
      best_rouge = cur_rouge
      torch.save(model.state_dict(), 'best-seq2seq.pt')
    
    if make_weights_static==True:
      model.encoder.embedding.weight.requires_grad=False
      make_weights_static=False
      print_log("Embeddings are static now")

	
	  
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
	
	
    print_log('Epoch: ' + str(epoch+1) + ' | Time: '+ str(epoch_mins) + 'm' +  str(epoch_secs) + 's')
    print_log('\t Learning Rate: ' + str(optimizer.param_groups[0]['lr']))
    print_log('\t Train Loss: ' + str(round(train_loss, 2)) + ' | Train PPL: ' + str(round(math.exp(train_loss), 2)))
    print_log('\t Val. Loss: ' + str(round(valid_loss, 2 )) + ' |  Val. PPL: '+ str(round(math.exp(valid_loss), 2)))
    print_log('\t Current Val. Rouge: ' + str(cur_rouge) + ' |  Best Rouge '+ str(best_rouge) + ' |  Best Epoch '+ str(best_epoch))
    print_log('\t Number of Epochs of no Improvement '+ str(num_of_epochs_not_improved))

  model.load_state_dict(torch.load('best-seq2seq.pt'))
  test_loss = evaluate(model, test_iterator, criterion)
  print_log('Test Loss: ' + str(round(test_loss, 2)) + ' | Test PPL: ' + str(round(math.exp(test_loss), 2)))
  p, t = get_preds(test_data, SRC, TRG, model, device)
  write_files(p,t,epoch=0, test=True)
  
if __name__ == '__main__':
  main()
