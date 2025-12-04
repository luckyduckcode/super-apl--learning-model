import tkinter as tk
from app import SuperAPLApp
import json
import os
import sys

class DuckApp(SuperAPLApp):
    def __init__(self, root):
        super().__init__(root)
        self.root.title("Duck (Super APL) - R2-D2/C-3PO Edition")
        self._apply_duck_personality()

    def _apply_duck_personality(self):
        self.log("\n[System] Initializing Duck Personality Matrix...")
        
        # Try to load the personality file
        try:
            # Determine base path (works for PyInstaller and Source)
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))
            
            json_path = os.path.join(base_path, "duck_personality.json")
            
            # Fallback for dev environment
            if not os.path.exists(json_path):
                json_path = os.path.join(base_path, "..", "training", "duck_personality.json")

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                profile = data.get("personality_profile", {})
                self.log(f"[Duck] Humor Setting: {profile.get('humor_style')}")
                self.log(f"[Duck] Versatility: {profile.get('versatility_style')}")
                self.log(f"[Duck] System Prompt: {profile.get('system_prompt')}")
                
                # Pre-fill some text
                self.input_text.delete("1.0", "end")
                self.input_text.insert("1.0", "Duck, please translate this protocol and make a joke about it.")
            else:
                self.log("[Duck] Personality file not found. Running in default mode.")
                
        except Exception as e:
            self.log(f"[Error] Could not load personality: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DuckApp(root)
    root.mainloop()
