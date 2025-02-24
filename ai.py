import requests
import json
import fitz  # PyMuPDF
import os

# Define Ollama API endpoint
OLLAMA_URL = "http://localhost:11500/api/chat"
CHAT_HISTORY_FILE = "chat_history.json"

# System message to guide the AI's behavior
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful and precise assistant. Answer user questions directly without unnecessary apologies. and go through the chat history carefully"
}

# Load previous chat history if available
if os.path.exists(CHAT_HISTORY_FILE):
    with open(CHAT_HISTORY_FILE, "r") as file:
        chat_history = json.load(file)
else:
    chat_history = [SYSTEM_PROMPT]  # Ensure system prompt is always the first message

# Ensure system prompt is up to date
if chat_history and chat_history[0]["role"] == "system":
    chat_history[0] = SYSTEM_PROMPT  # Overwrite system message if it exists
else:
    chat_history.insert(0, SYSTEM_PROMPT)  # Insert if missing

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

print("\U0001F916 Chat with Mistral! Type 'exit' to quit. Type 'upload [filename.pdf]' to analyze a PDF.")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye! \U0001F44B")
        with open(CHAT_HISTORY_FILE, "w") as file:
            json.dump(chat_history, file, indent=4)
        break

    # Handle PDF uploads
    if user_input.lower().startswith("upload "):
        pdf_path = user_input.split(" ", 1)[1]
        if os.path.exists(pdf_path):
            pdf_text = extract_text_from_pdf(pdf_path)
            print("\U0001F4C4 Extracted Text:\n", pdf_text[:1000])  # Preview first 1000 chars
            chat_history.append({"role": "user", "content": f"Uploaded PDF text: {pdf_text}"})
        else:
            print("❌ File not found. Please provide a valid file path.")
        continue

    # Ensure system prompt remains updated
    if chat_history[0]["role"] == "system":
        chat_history[0] = SYSTEM_PROMPT

    # Append user message to chat history
    chat_history.append({"role": "user", "content": user_input})

    # Send request to Ollama
    response = requests.post(OLLAMA_URL, json={"model": "mistral", "messages": chat_history}, stream=True)

    # Read streamed response
    ai_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_data = json.loads(line.decode("utf-8"))
                ai_response += json_data["message"]["content"]
            except (json.JSONDecodeError, KeyError):
                print("❌ Error: Invalid JSON chunk received:", line)

    print("\U0001F916 Royal:", ai_response)

    # Append AI response to chat history
    chat_history.append({"role": "assistant", "content": ai_response})

    # Save chat history
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(chat_history, file, indent=4)
