#!/usr/bin/env python3
"""
Duck Chat GUI Launcher
Standalone GUI wrapper for Duck Chat with no dependencies on console
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
import json
import os
import sys
from datetime import datetime
import threading
import subprocess
import urllib.request
import urllib.error
import json

class DuckChatGUI:
    """GUI wrapper for Duck Chat"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🦆 Duck Chat")
        self.root.geometry("900x700")
        self.root.configure(bg="#2b2b2b")
        
        # API Configuration
        self.api_url = "http://localhost:5000/api/v1/chat"
        self.session_id = None
        self.api_connected = False
        
        # Load personality
        self.personality = self._load_personality()
        
        # Chat history
        self.chat_history = []
        
        # Build GUI
        self._build_ui()
        self._init_conversation()
        
        # Check API connection in background
        threading.Thread(target=self._check_api_connection, daemon=True).start()

    def _check_api_connection(self):
        """Check if API server is running"""
        try:
            with urllib.request.urlopen("http://localhost:5000/api/v1/status", timeout=1) as response:
                if response.status == 200:
                    self.api_connected = True
                    self.root.after(0, lambda: self._update_status("Online (API)"))
        except:
            self.api_connected = False
            self.root.after(0, lambda: self._update_status("Offline (Local Mode)"))

    def _update_status(self, status_text):
        """Update status label"""
        if hasattr(self, 'status_label'):
            color = "#00ff00" if "Online" in status_text else "#ffff00"
            self.status_label.config(text=f"Status: {status_text}", fg=color)

    def _call_api(self, message):
        """Call the local API"""
        try:
            data = json.dumps({"message": message, "session_id": self.session_id}).encode('utf-8')
            req = urllib.request.Request(self.api_url, data=data, headers={'Content-Type': 'application/json'})
            
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode('utf-8'))
                    self.session_id = result.get("session_id")
                    return result.get("duck_response")
        except Exception as e:
            print(f"API Call failed: {e}")
            return None
    
    def _load_personality(self):
        """Load Duck personality"""
        try:
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))
            
            # Try different paths
            paths = [
                os.path.join(base_path, "duck_personality.json"),
                os.path.join(base_path, "..", "training", "duck_personality.json"),
                os.path.join(base_path, "src", "training", "duck_personality.json"),
            ]
            
            for path in paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        return json.load(f)
        except Exception as e:
            print(f"Warning: {e}")
        
        return {
            "model_name": "Duck",
            "personality_profile": {
                "humor_style": "R2-D2",
                "versatility_style": "C-3PO",
                "system_prompt": "You are Duck"
            }
        }
    
    def _build_ui(self):
        """Build the GUI"""
        # Header
        header = tk.Frame(self.root, bg="#1a1a1a", height=80)
        header.pack(fill=tk.X, padx=0, pady=0)
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🦆 Duck Chat", font=("Arial", 24, "bold"), bg="#1a1a1a", fg="#00ff00")
        title.pack(pady=5)
        
        subtitle = tk.Label(header, text="R2-D2 Humor + C-3PO Versatility", font=("Arial", 10), bg="#1a1a1a", fg="#00cc00")
        subtitle.pack()
        
        self.status_label = tk.Label(header, text="Status: Checking...", font=("Arial", 8), bg="#1a1a1a", fg="#888888")
        self.status_label.pack(anchor=tk.E, padx=10)
        
        # Chat display
        chat_frame = tk.Frame(self.root, bg="#2b2b2b")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Courier New", 10),
            bg="#1a1a1a",
            fg="#00ff00",
            insertbackground="#00ff00",
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        self.chat_display.tag_config("user", foreground="#00ff00", font=("Courier New", 10, "bold"))
        self.chat_display.tag_config("duck", foreground="#ffff00", font=("Courier New", 10))
        
        # Input frame
        input_frame = tk.Frame(self.root, bg="#2b2b2b")
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        label = tk.Label(input_frame, text="You:", font=("Arial", 10, "bold"), bg="#2b2b2b", fg="#00ff00")
        label.pack(anchor=tk.W)
        
        self.input_box = tk.Entry(
            input_frame,
            font=("Courier New", 11),
            bg="#1a1a1a",
            fg="#00ff00",
            insertbackground="#00ff00",
            relief=tk.FLAT
        )
        self.input_box.pack(fill=tk.X, pady=(5, 0))
        self.input_box.bind("<Return>", lambda e: self._send_message())
        
        # Buttons
        button_frame = tk.Frame(input_frame, bg="#2b2b2b")
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        send_btn = tk.Button(button_frame, text="Send (Enter)", command=self._send_message, 
                             bg="#00aa00", fg="#000000", font=("Arial", 10, "bold"), relief=tk.FLAT, padx=10, pady=5)
        send_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        clear_btn = tk.Button(button_frame, text="Clear", command=self._clear_chat,
                              bg="#444444", fg="#ffffff", font=("Arial", 10), relief=tk.FLAT, padx=10, pady=5)
        clear_btn.pack(side=tk.LEFT)
    
    def _init_conversation(self):
        """Start with Duck's greeting"""
        greeting = """*beep boop* 🦆 Hello! I'm Duck!

I'm trained on Llama APL with R2-D2 humor and C-3PO versatility.
Ask me anything! Type 'help' for commands.
"""
        self._display_message("Duck", greeting, "duck")
    
    def _display_message(self, sender, message, tag="user"):
        """Display message in chat"""
        self.chat_display.config(state=tk.NORMAL)
        
        if sender == "Duck":
            self.chat_display.insert(tk.END, f"Duck: ", tag)
        else:
            self.chat_display.insert(tk.END, f"You: ", "user")
        
        self.chat_display.insert(tk.END, f"{message}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def _send_message(self):
        """Send message and get response"""
        user_input = self.input_box.get().strip()
        
        if not user_input:
            return
        
        self._display_message("You", user_input, "user")
        self.input_box.delete(0, tk.END)
        
        # Handle special commands
        if user_input.lower() == "help":
            self._show_help()
        elif user_input.lower() == "clear":
            self._clear_chat()
        elif user_input.lower() in ["exit", "quit", "bye"]:
            self.root.quit()
        else:
            # Get response ONLY from API
            if self.api_connected:
                response = self._call_api(user_input)
                if response:
                    self._display_message("Duck", response, "duck")
                else:
                    self._display_message("Duck", "Error: Connection to Neural Engine lost.", "error")
            else:
                self._display_message("Duck", "Error: Duck Brain (API Server) is OFFLINE.\nPlease run DuckServer.exe first.", "error")
    
    def _show_help(self):
        """Show help"""
        help_text = """Commands:
• 'help' - Show this help
• 'clear' - Clear chat
• 'exit'/'quit'/'bye' - Exit
• Or just chat naturally!"""
        self._display_message("Duck", help_text, "duck")
    
    def _clear_chat(self):
        """Clear chat"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self._init_conversation()


def main():
    """Launch Duck Chat GUI"""
    root = tk.Tk()
    app = DuckChatGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
