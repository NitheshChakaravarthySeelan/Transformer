import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from datasets import load_dataset
from model import build_transformer
import warnings
warnings.filterwarnings("ignore")

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(ds, lang):
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    special_tokens = ["<unk>", "<pad>", "<s>", "</s>"]
    trainer = WordLevelTrainer(
        special_tokens=special_tokens,
        min_frequency=2
    )
    
    sentences = get_all_sentences(ds, lang)
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    
    return tokenizer

def translate(model, tokenizer_src, tokenizer_tgt, src_text, max_len=128, device='cpu'):
    model.eval()
    encoder_input = tokenizer_src.encode(src_text)
    encoder_input_tokens = torch.tensor([tokenizer_src.token_to_id("<s>")] + 
                                      encoder_input.ids + 
                                      [tokenizer_src.token_to_id("</s>")],
                                      dtype=torch.int64)
    if len(encoder_input_tokens) < max_len:
        encoder_input_tokens = torch.cat([
            encoder_input_tokens,
            torch.tensor([tokenizer_src.token_to_id("<pad>")] * (max_len - len(encoder_input_tokens)),
                        dtype=torch.int64)
        ])
    encoder_mask = (encoder_input_tokens != tokenizer_src.token_to_id("<pad>")).unsqueeze(0).unsqueeze(0).int()
    
    encoder_input_tokens = encoder_input_tokens.unsqueeze(0).to(device) 
    encoder_mask = encoder_mask.to(device)
    
    with torch.no_grad():
        encoder_output = model.encode(encoder_input_tokens, encoder_mask)
        decoder_input = torch.tensor([[tokenizer_tgt.token_to_id("<s>")]], dtype=torch.int64).to(device)

        for _ in range(max_len - 1):
            decoder_mask = (decoder_input != tokenizer_tgt.token_to_id("<pad>")).unsqueeze(0).unsqueeze(0).int()
            decoder_mask = decoder_mask & torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).to(device) == 0
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            prob = proj_output[:, -1]
            next_token = torch.argmax(prob, dim=-1, keepdim=True)
            decoder_input = torch.cat([decoder_input, next_token], dim=-1)
            if next_token.item() == tokenizer_tgt.token_to_id("</s>"):
                break

    output_tokens = decoder_input[0].cpu().numpy()
    output_text = []
    for token in output_tokens:
        if token == tokenizer_tgt.token_to_id("</s>"):
            break
        if token == tokenizer_tgt.token_to_id("<s>"):
            continue
        output_text.append(tokenizer_tgt.id_to_token(token))
    
    return " ".join(output_text)

def main():
    config = {
        'seq_len': 128,
        'd_model': 256,
        'src_lang': 'en',
        'tgt_lang': 'it'
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        print("Loading dataset...")
        ds_raw = load_dataset('opus_books', f"{config['src_lang']}-{config['tgt_lang']}", split='train')
        
        print("Building tokenizers...")
        tokenizer_src = get_or_build_tokenizer(ds_raw, config['src_lang'])
        tokenizer_tgt = get_or_build_tokenizer(ds_raw, config['tgt_lang'])
        
        print("Creating model...")
        model = build_transformer(
            src_vocab_size=tokenizer_src.get_vocab_size(),
            tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
            src_seq_len=config['seq_len'],
            tgt_seq_len=config['seq_len'],
            d_model=config['d_model']
        )
        
        print("Loading model weights...")
        model.load_state_dict(torch.load('transformer_model.pth', map_location=device))
        model.to(device)
        
        print(f"\nModel loaded successfully!")
        print(f"Source vocabulary size: {tokenizer_src.get_vocab_size()}")
        print(f"Target vocabulary size: {tokenizer_tgt.get_vocab_size()}")
        
        while True:
            src_text = input("\nEnter English text to translate (or 'q' to quit): ")
            
            if src_text.lower() == 'q':
                break
            print("\nTranslating...")
            translated_text = translate(model, tokenizer_src, tokenizer_tgt, src_text, device=device)
            print(f"\nEnglish: {src_text}")
            print(f"Italian: {translated_text}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()