import os
import urllib.request
import torch
from .model import CIMNanoGPT
from profiles.device_profile import DeviceProfile
import json

def get_tiny_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = "tiny_shakespeare.txt"
    if not os.path.exists(filepath):
        print("Downloading Tiny Shakespeare dataset...")
        urllib.request.urlretrieve(url, filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    # reduce size to 10% for ultra fast demo
    return text[:len(text)//10]

def main(epochs=5, device_path=None):
    print("🚀 [CIM Nano-GPT] Starting Generative AI Large Language Model on Organic Memory Devices...")
    
    # 1. Load Device Profile
    profile = None
    if device_path and os.path.exists(device_path):
        with open(device_path, 'r') as f:
            profile = DeviceProfile(**json.load(f))
        print(f"✅ Loaded CIM Device Profile: {profile.name}")
    else:
        print("⚠️ No valid device profile provided. Running standard idealized evaluation.")
        
    # 2. Prepare Data
    text = get_tiny_shakespeare()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    block_size = 64
    batch_size = 32
    
    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x, y

    # 3. Model
    model = CIMNanoGPT(vocab_size, n_embd=64, n_head=4, n_layer=3, block_size=block_size, device_profile=profile)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 4. Training Loop
    print(f"🧠 Training Nano-GPT for {epochs} epochs...")
    model.train()
    steps_per_epoch = 100
    for epoch in range(epochs):
        total_loss = 0
        for step in range(steps_per_epoch):
            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/steps_per_epoch:.4f}")
        
    # 5. Generative AIGC Evaluation
    print("\n✨ Generating text based on physical device characteristics...")
    model.eval()
    if profile:
        # Simulate severe drift/retention loss after 1 year to show Hardware impact
        for m in model.modules():
            if hasattr(m, 'drift_hours'):
                m.drift_hours = 8760.0
                
    context = torch.zeros((1, 1), dtype=torch.long) # start with \n
    generated = decode(model.generate(context, max_new_tokens=200)[0].tolist())
    print("\n" + "="*50)
    print("📝 GENERATED TEXT:")
    print("="*50)
    print(generated)
    print("="*50)

if __name__ == '__main__':
    main()
