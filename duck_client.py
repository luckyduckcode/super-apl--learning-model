"""
Duck AI Chat Client - Connects to Duck Server via REST API

This is the client-side application that:
- Connects to Duck Server on localhost:5000
- Provides interactive chat interface
- Shows real-time statistics
"""

import requests
import json
import sys
import time
from typing import Optional, Dict
import socket


class DuckChatClient:
    """Client for connecting to Duck Server"""
    
    def __init__(self, server_url: str = 'http://127.0.0.1:5000', timeout: int = 30):
        self.server_url = server_url
        self.timeout = timeout
        self.session = requests.Session()
        self.conversation_id = int(time.time() * 1000)
        self.message_count = 0
        self.total_latency = 0
        self.connected = False
        self.server_info = {}
    
    def connect(self) -> bool:
        """Connect to Duck Server and verify it's running"""
        print("\n" + "="*70)
        print("DUCK AI CHAT CLIENT - v1.0.0")
        print("="*70)
        
        print(f"\n[Connecting] to Duck Server at {self.server_url}...")
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Test connection with health check
                response = self.session.get(
                    f'{self.server_url}/api/health',
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    # Get server info
                    status_resp = self.session.get(
                        f'{self.server_url}/api/status',
                        timeout=self.timeout
                    )
                    
                    if status_resp.status_code == 200:
                        self.server_info = status_resp.json()
                        self.connected = True
                        
                        print(f"\n✓ Connected to Duck Server!")
                        print(f"  Model: {self.server_info.get('model', 'Unknown')}")
                        print(f"  Parameters: {self.server_info.get('parameters', 'Unknown'):,}")
                        print(f"  Compression: {self.server_info.get('compression_ratio', 0):.1f}x")
                        print(f"  Memory: {self.server_info.get('memory_usage_mb', 0)} MB")
                        print(f"  Sparsity: {self.server_info.get('sparsity', 0):.1%}")
                        
                        return True
            
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries}... Server not ready yet")
                    import time
                    time.sleep(2)
                else:
                    print(f"✗ Connection failed: Could not reach {self.server_url}")
                    print(f"\n  Make sure Duck Server is running:")
                    print(f"    python duck_server.py")
                    print(f"    or")
                    print(f"    DuckServer.exe")
                    return False
            
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries}... Timeout, retrying...")
                    import time
                    time.sleep(2)
                else:
                    print(f"✗ Connection timeout: Server not responding after {max_retries} attempts")
                    return False
            
            except Exception as e:
                print(f"✗ Connection error: {e}")
                return False
        
        return False
    
    def send_message(self, message: str) -> Optional[Dict]:
        """Send a message to Duck Server and get response"""
        
        if not self.connected:
            print("✗ Not connected to server. Please connect first.")
            return None
        
        try:
            start_time = time.time()
            
            response = self.session.post(
                f'{self.server_url}/api/chat',
                json={'message': message},
                timeout=self.timeout
            )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Update stats
                self.message_count += 1
                self.total_latency += latency
                
                return result
            else:
                print(f"✗ Server error: {response.status_code}")
                return None
        
        except requests.exceptions.Timeout:
            print("✗ Request timeout")
            return None
        
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """Get current conversation statistics"""
        avg_latency = self.total_latency / max(self.message_count, 1)
        
        return {
            'conversation_id': self.conversation_id,
            'messages_sent': self.message_count,
            'total_time': self.total_latency,
            'avg_latency': avg_latency,
            'server_info': self.server_info
        }
    
    def print_statistics(self):
        """Print conversation statistics"""
        stats = self.get_statistics()
        
        print("\n" + "-"*70)
        print("CONVERSATION STATISTICS")
        print("-"*70)
        print(f"Messages sent: {stats['messages_sent']}")
        print(f"Average latency: {stats['avg_latency']*1000:.1f}ms")
        print(f"Total conversation time: {stats['total_time']:.1f}s")
        print(f"Server uptime: {self.server_info.get('uptime_seconds', 0):.1f}s")
        print(f"Total server inferences: {self.server_info.get('inference_count', 0)}")


class DuckChatUI:
    """Interactive Chat UI for Duck"""
    
    def __init__(self, client: DuckChatClient):
        self.client = client
    
    def run(self):
        """Run the interactive chat interface"""
        
        if not self.client.connect():
            print("\n✗ Failed to connect to Duck Server")
            print("\nMake sure the Duck Server is running:")
            print("  python duck_server.py")
            sys.exit(1)
        
        print("\n" + "-"*70)
        print("CHAT")
        print("-"*70)
        print("\nTalk to Duck! (Type 'stats' for statistics, 'quit' to exit)\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("\nDuck Server: Thanks for chatting! Goodbye!")
                    self._show_final_stats()
                    break
                
                if user_input.lower() == 'stats':
                    self.client.print_statistics()
                    continue
                
                if user_input.lower() == 'clear':
                    self.client.message_count = 0
                    self.client.total_latency = 0
                    print("Duck Server: Statistics cleared!\n")
                    continue
                
                # Send message to server
                print("Duck Server: ", end="", flush=True)
                
                result = self.client.send_message(user_input)
                
                if result and result.get('status') == 'success':
                    response = result.get('response', 'No response')
                    latency = result.get('latency_ms', 0)
                    
                    print(response)
                    print(f"              [latency: {latency:.0f}ms]")
                else:
                    print("Sorry, I encountered an error. Please try again.")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\nDuck Server: Goodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}\n")
    
    def _show_final_stats(self):
        """Show final statistics before exit"""
        stats = self.client.get_statistics()
        
        print("\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)
        print(f"\nConversation ID: {stats['conversation_id']}")
        print(f"Messages exchanged: {stats['messages_sent']}")
        if stats['messages_sent'] > 0:
            print(f"Average latency: {stats['avg_latency']*1000:.1f}ms")
            print(f"Total time: {stats['total_time']:.1f}s")
        print(f"\nServer Statistics:")
        print(f"  Total inferences: {self.client.server_info.get('inference_count', 0)}")
        print(f"  Server uptime: {self.client.server_info.get('uptime_seconds', 0):.1f}s")
        print(f"  Model: {self.client.server_info.get('model', 'Unknown')}")
        print(f"  Compression: {self.client.server_info.get('compression_ratio', 0):.1f}x")
        print("\n")


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Duck AI Chat Client")
    parser.add_argument('--server', type=str, default='http://127.0.0.1:5000',
                       help='Duck Server URL (default: http://127.0.0.1:5000)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Try to clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=" * 60)
    print("Duck AI Chat Client")
    print("=" * 60)
    
    try:
        # Create client
        client = DuckChatClient(server_url=args.server, timeout=args.timeout)
        
        # Run UI
        ui = DuckChatUI(client)
        ui.run()
    except KeyboardInterrupt:
        print("\n\nChat session ended")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
