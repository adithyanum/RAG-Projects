import ollama
import re
from config import LLM_MODEL, SYSTEM_PROMPT
from tools import AVAILABLE_TOOLS

class ResearchAgent:
    def __init__(self):
        # We start with the System Prompt to set the rules
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def handle_turn(self, user_input):
        """Processes a single user question until a FINAL ANSWER is reached."""
        self.messages.append({"role": "user", "content": user_input})
        
        retries = 0
        max_retries = 3

        while retries < max_retries:

            response = ollama.chat(model=LLM_MODEL, messages=self.messages)
            bot_text = response['message']['content']

            print(f"DEBUG: Agent thinking...")

            # 1. Check for ACTION (Tool Call)
            action_match = re.search(r'ACTION:\s*(\w+)\(["\'](.*?)["\']\)', bot_text, re.IGNORECASE)
            
            if action_match:
                # Keep the chat history clean by ending at the action
                self.messages.append({"role": "assistant", "content": bot_text})
                
                tool_name = action_match.group(1).strip()
                tool_query = action_match.group(2).strip()
                
                if tool_name in AVAILABLE_TOOLS:
                    print(f"⚙️ Executing {tool_name} for: {tool_query}")
                    observation = AVAILABLE_TOOLS[tool_name](tool_query)
                    self.messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
                    retries = 0 
                    continue 
                else:
                    self.messages.append({"role": "user", "content": f"OBSERVATION: Tool {tool_name} not found."})
                    continue

            # 2. Check for FINAL ANSWER
            if "FINAL ANSWER" in bot_text.upper():
                # Store the final response and return it to main.py
                self.messages.append({"role": "assistant", "content": bot_text})
                return bot_text
            
            # 3. Handle formatting errors (Retries)
            retries += 1
            print(f"⚠️ Formatting error. Retry {retries}/{max_retries}")
            self.messages.append({"role": "user", "content": "Please use the THOUGHT: and ACTION: or FINAL ANSWER: tags strictly."})

        return "❌ I'm sorry, I encountered a loop error. Let's try rephrasing."

    def reset_memory(self):
        """Keep the system prompt but clear the chat history."""
        self.messages = [self.messages[0]]