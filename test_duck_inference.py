"""
Test and interact with the trained Duck 1.58-bit quantized model.

This script allows you to:
1. Load the trained model
2. Test inference with various prompts
3. Measure performance metrics (latency, memory)
4. Compare personality responses
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
import time
import sys

# Add src to path
sys.path.insert(0, 'src/training')

from quantize_1_58bit import QuantizedLinear, QuantizedTransformer


class DuckQuantized(nn.Module):
    """Duck Chat model with 1.58-bit quantization"""
    
    def __init__(self, vocab_size: int, hidden_size: int = 768,
                 num_layers: int = 12, num_attention_heads: int = 12):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Token embeddings (not quantized)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Quantized transformer layers
        self.layers = nn.ModuleList([
            QuantizedTransformer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=hidden_size * 4,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])
        
        # Output layer (quantized)
        self.lm_head = QuantizedLinear(hidden_size, vocab_size, channel_wise=False)
        
        # Layer norm (not quantized)
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantized Duck.
        
        Args:
            input_ids: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Embed tokens
        x = self.embedding(input_ids)
        
        # Apply quantized transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm
        x = self.ln(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits


class SimpleTokenizer:
    """Simple character-based tokenizer for testing"""
    
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        
        # Create simple mapping
        for i in range(256):
            self.char_to_id[chr(i)] = i
            self.id_to_char[i] = chr(i)
    
    def encode(self, text: str) -> list:
        """Convert text to token ids"""
        tokens = []
        for char in text[:500]:  # Limit to 500 chars
            if ord(char) < 256:
                tokens.append(ord(char))
            else:
                tokens.append(ord('?'))
        return tokens
    
    def decode(self, tokens: list) -> str:
        """Convert token ids back to text"""
        try:
            return ''.join(chr(t) for t in tokens if 0 <= t < 256)
        except:
            return '<invalid_tokens>'


def load_trained_model(model_path: str, device: str = 'cpu'):
    """Load the trained Duck model"""
    print(f"\n[Loading] Checking for trained model at {model_path}...")
    
    if not Path(model_path).exists():
        print(f"⚠ Model not found at {model_path}")
        print("  Creating fresh quantized Duck model for testing...")
        model = DuckQuantized(
            vocab_size=50257,
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12
        )
        model.to(device)
        model.eval()
        return model
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = DuckQuantized(
            vocab_size=checkpoint['config']['vocab_size'],
            hidden_size=checkpoint['config']['hidden_size'],
            num_layers=checkpoint['config']['num_layers'],
            num_attention_heads=checkpoint['config']['num_attention_heads']
        )
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)
        model.eval()
        
        print(f"✓ Loaded trained model from {model_path}")
        print(f"  Compression: {checkpoint['stats']['total_compression']:.1f}x")
        print(f"  Sparsity: {checkpoint['stats']['avg_sparsity']:.1%}")
        
        return model
    except Exception as e:
        print(f"⚠ Error loading checkpoint: {e}")
        print("  Creating fresh quantized Duck model for testing...")
        model = DuckQuantized(
            vocab_size=50257,
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12
        )
        model.to(device)
        model.eval()
        return model


def generate_response(model, prompt: str, tokenizer, max_len: int = 50, 
                     temperature: float = 0.7, top_k: int = 40):
    """Generate a response from the model"""
    
    model.eval()
    with torch.no_grad():
        # Encode prompt
        tokens = tokenizer.encode(prompt)
        input_ids = torch.LongTensor([tokens]).to(next(model.parameters()).device)
        
        # Start timing
        start_time = time.time()
        
        # Generate tokens
        generated = tokens.copy()
        
        for _ in range(max_len):
            # Get logits for next token
            logits = model(input_ids[:, -512:])  # Use last 512 tokens
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            
            # Update input for next iteration
            input_ids = torch.LongTensor([generated[-512:]]).to(next(model.parameters()).device)
        
        latency = time.time() - start_time
        
        # Decode response
        response = tokenizer.decode(generated)
        
        return response, latency


def test_duck():
    """Interactive test of Duck personality"""
    
    print("\n" + "="*70)
    print("DUCK PERSONALITY TEST - 1.58-BIT QUANTIZATION")
    print("="*70)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = SimpleTokenizer()
    
    # Load model
    model = load_trained_model('models/duck_1_58bit.pt', device=device)
    
    # Get model stats
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] Parameters: {total_params:,}")
    print(f"[Device] {device.upper()}")
    
    # Test prompts to measure Duck's personality
    test_prompts = [
        "Hi Duck, what's your name?",
        "Tell me about yourself.",
        "What do you like to do?",
        "How do you feel about coding?",
        "What's your favorite thing?",
        "Can you help me?",
        "What makes you special?",
        "How are you today?",
    ]
    
    print("\n" + "-"*70)
    print("PERSONALITY TEST - Testing Duck's responses")
    print("-"*70)
    
    results = []
    total_latency = 0
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}/{len(test_prompts)}] Prompt: '{prompt}'")
        
        try:
            response, latency = generate_response(
                model, prompt, tokenizer,
                max_len=30,
                temperature=0.8,
                top_k=40
            )
            
            # Clean response
            response_clean = response.replace('\x00', '').strip()[:100]
            
            print(f"  Response: '{response_clean}'")
            print(f"  Latency: {latency*1000:.1f}ms")
            
            results.append({
                'prompt': prompt,
                'response': response_clean,
                'latency': latency
            })
            total_latency += latency
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'prompt': prompt,
                'response': f'<error: {str(e)[:50]}>',
                'latency': 0
            })
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    avg_latency = total_latency / len(test_prompts) if test_prompts else 0
    print(f"\n✓ Tests completed: {len(results)}")
    print(f"✓ Average latency: {avg_latency*1000:.1f}ms")
    print(f"✓ Total time: {total_latency:.2f}s")
    print(f"✓ Device: {device.upper()}")
    print(f"✓ Model parameters: {total_params:,}")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"✓ GPU memory used: {memory_used:.2f} GB")
    else:
        import psutil
        process = psutil.Process()
        memory_used = process.memory_info().rss / 1024**3
        print(f"✓ CPU memory used: {memory_used:.2f} GB")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Retrain with: python src/training/train_duck_personality.py --epochs 10")
    print("2. Integrate into Duck Chat API: See LARGE_MODEL_TRAINING_GUIDE.md")
    print("3. Deploy to production: See README.md")
    print("\n")
    
    return results


def interactive_chat():
    """Interactive chat with Duck"""
    
    print("\n" + "="*70)
    print("DUCK INTERACTIVE CHAT - 1.58-BIT QUANTIZATION")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = SimpleTokenizer()
    model = load_trained_model('models/duck_1_58bit.pt', device=device)
    
    model.eval()
    
    print(f"\n[Ready] Duck is loaded and ready to chat!")
    print(f"[Device] {device.upper()}")
    print("\nType 'quit' to exit, 'clear' to reset conversation\n")
    
    conversation_history = ""
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("\nDuck: Goodbye! Thanks for chatting with me!")
                break
            
            if user_input.lower() == 'clear':
                conversation_history = ""
                print("Duck: Conversation cleared!\n")
                continue
            
            if not user_input:
                continue
            
            # Add to history
            conversation_history += f"You: {user_input}\nDuck: "
            
            # Generate response
            response, latency = generate_response(
                model, conversation_history, tokenizer,
                max_len=40, temperature=0.8, top_k=50
            )
            
            response_clean = response.split('\n')[0].replace('\x00', '').strip()[:150]
            
            print(f"Duck: {response_clean}")
            print(f"      [latency: {latency*1000:.0f}ms]\n")
            
            # Add response to history
            conversation_history += response_clean + "\n"
            
        except KeyboardInterrupt:
            print("\n\nDuck: Goodbye!")
            break
        except Exception as e:
            print(f"Duck: Sorry, I encountered an error: {e}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Duck personality")
    parser.add_argument('--interactive', action='store_true', help='Start interactive chat')
    parser.add_argument('--test', action='store_true', help='Run automated tests (default)')
    
    args = parser.parse_args()
    
    if args.interactive:
        try:
            interactive_chat()
        except ImportError:
            print("✓ Interactive chat requires psutil. Running test mode instead...\n")
            test_duck()
    else:
        test_duck()
