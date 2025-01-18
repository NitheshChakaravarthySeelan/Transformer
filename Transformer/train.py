import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import sentencepiece as spm
import numpy as np
from typing import List, Tuple
import os
from datasets import load_dataset
from model import build_transformer,Transformer

class SentencePieceTokenizer:
    def __init__(self, vocab_size= 32000):
        self.vocab_size = vocab_size
        self.sp_model = None
        
    def train(self, texts, model_prefix = "tokenizer"):
        with open('train_text.txt', 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        spm.SentencePieceTrainer.train(
            input='train_text.txt',
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type='bpe',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='[PAD]',
            unk_piece='[UNK]',
            bos_piece='[BOS]',
            eos_piece='[EOS]'
        )
        
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f"{model_prefix}.model")
    
    def encode(self, text):
        return [2] + self.sp_model.encode_as_ids(text) + [3]  
    
    def decode(self, ids) :
        ids = [id for id in ids if id not in [0, 2, 3]]
        return self.sp_model.decode_ids(ids)

class TextGenerationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            
            tokens = self.tokenizer.encode(text)
            
            for i in range(0, len(tokens) - max_length + 1, max_length // 2):
                sequence = tokens[i:i + max_length]
                if len(sequence) == max_length:
                    self.examples.append(sequence)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

def create_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).float()
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

class TransformerTrainer:
    def __init__(
        self,
        model,
        tokenizer: SentencePieceTokenizer,
        device= "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer
    ):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            src_mask = torch.ones((src.shape[0], 1, src.shape[1])).to(self.device)
            tgt_mask = create_mask(tgt.shape[1]).to(self.device)
            
            encoder_output = self.model.encode(src, src_mask)
            decoder_output = self.model.decode(encoder_output, src_mask, src, tgt_mask)
            proj_output = self.model.project(decoder_output)
            
            loss = self.criterion(
                proj_output.view(-1, proj_output.size(-1)),
                tgt.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return total_loss / len(dataloader)
    
    def generate(
        self,
        prompt,
        max_length= 100,
        temperature= 1.0
    ):
        self.model.eval()
        tokens = self.tokenizer.encode(prompt)
        src = torch.tensor(tokens).unsqueeze(0).to(self.device)
        
        for _ in range(max_length):
            src_mask = torch.ones((1, 1, src.shape[1])).to(self.device)
            tgt_mask = create_mask(src.shape[1]).to(self.device)
            
            encoder_output = self.model.encode(src, src_mask)
            decoder_output = self.model.decode(encoder_output, src_mask, src, tgt_mask)
            proj_output = self.model.project(decoder_output)
            
            next_token_logits = proj_output[0, -1, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, 1)
            
            if next_token.item() == 3:
                break
                
            src = torch.cat([src, next_token.unsqueeze(0)], dim=1)
        
        return self.tokenizer.decode(src[0].tolist())

def train_transformer(model_config, training_config):
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    print("Preparing texts for tokenizer...")
    train_texts = dataset['train']['text']
    
    print("Training tokenizer...")
    tokenizer = SentencePieceTokenizer(vocab_size=model_config['vocab_size'])
    tokenizer.train(train_texts)
    
    print("Creating dataset...")
    train_dataset = TextGenerationDataset(
        train_texts,
        tokenizer,
        max_length=model_config['seq_len']
    )
    
    print("Creating dataloader...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True
    )
    
    print("Initializing model...")
    model = Transformer(
        src_vocab_size=model_config['vocab_size'],
        tgt_vocab_size=model_config['vocab_size'],
        src_seq_len=model_config['seq_len'],
        tgt_seq_len=model_config['seq_len'],
        d_model=model_config['d_model'],
        N=model_config['n_layers'],
        h=model_config['n_heads'],
        dropout=model_config['dropout'],
        d_ff=model_config['d_ff']
    )
    
    print("Setting up trainer...")
    trainer = TransformerTrainer(model, tokenizer)
    optimizer = Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    print("Starting training...")
    for epoch in range(training_config['epochs']):
        print(f"Epoch {epoch + 1}/{training_config['epochs']}")
        loss = trainer.train_epoch(train_dataloader, optimizer)
        print(f"Epoch {epoch + 1} Loss: {loss:.4f}")
        
        if (epoch + 1) % training_config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f"checkpoint_epoch_{epoch + 1}.pt")
    
    return trainer

if __name__ == "__main__":
    model_config = {
        'vocab_size': 32000,
        'seq_len': 512,
        'd_model': 512,
        'n_layers': 6,
        'n_heads': 8,
        'dropout': 0.1,
        'd_ff': 2048
    }

    training_config = {
        'batch_size': 32,
        'learning_rate': 0.0001,
        'epochs': 10,
        'save_every': 1
    }

    trainer = train_transformer(model_config, training_config)

    generated_text = trainer.generate(
        prompt="In the early days of artificial intelligence",
        max_length=100,
        temperature=0.7
    )
    print("\nGenerated text:")
    print(generated_text)