import requests
import json
import pdfplumber  # For better table extraction
import os
import re

# Define Ollama API endpoint
OLLAMA_URL = "http://localhost:11500/api/chat"
CHAT_HISTORY_FILE = "chat_history.json"
MAX_DOC_LENGTH = 5000  # Limit document text size to avoid excessive context

# System message to guide AI
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a highly precise AI assistant. You always reference uploaded documents for relevant questions. "
               "If a document is uploaded, assume it contains important user information. "
               "Always give priority to the answers in chat history rather then searching it online. "
               "If the answer is not in the document, say 'I couldn't find that information.'"
}

# Load or initialize chat history
if os.path.exists(CHAT_HISTORY_FILE):
    try:
        with open(CHAT_HISTORY_FILE, "r") as file:
            data = json.load(file)
            chat_history = data.get("chat_history", [SYSTEM_PROMPT]) if isinstance(data, dict) else [SYSTEM_PROMPT]
            pdf_context = data.get("pdf_context", "") if isinstance(data, dict) else ""
    except json.JSONDecodeError:
        chat_history = [SYSTEM_PROMPT]
        pdf_context = ""
else:
    chat_history = [SYSTEM_PROMPT]
    pdf_context = ""

# Ensure system prompt is always first
chat_history[0] = SYSTEM_PROMPT


def save_json():
    """Saves chat history to JSON file."""
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump({"chat_history": chat_history, "pdf_context": pdf_context}, file, indent=4)
        file.flush()
        os.fsync(file.fileno())  # Ensures immediate write to disk


def extract_text_from_pdf(pdf_path):
    """Extracts both text and tables from a PDF file."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"

        return text if text.strip() else "No readable text found. The document may be image-based."

    except Exception as e:
        return f"Error reading PDF: {e}"


print("\U0001F916 Chat with Mistral! Type 'exit' to quit. Type 'upload [filename.pdf]' to analyze a PDF.")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye! \U0001F44B")
        save_json()
        break

    # ✅ Detect file upload in the same message
    match = re.search(r"\bupload\s+([^\s]+\.pdf)\b", user_input, re.IGNORECASE)
    if match:
        pdf_path = match.group(1)
        user_message = re.sub(r"\bupload\s+[^\s]+\.pdf\b", "", user_input, flags=re.IGNORECASE).strip()

        if os.path.exists(pdf_path):
            pdf_text = extract_text_from_pdf(pdf_path)
            print("\U0001F4C4 Extracted Text (Preview):\n", pdf_text[:1000])

            # ✅ Truncate document if too long
            if len(pdf_text) > MAX_DOC_LENGTH:
                pdf_text = pdf_text[:MAX_DOC_LENGTH] + "...\n[Document truncated]"

            pdf_context = pdf_text  # Save extracted text

            # ✅ Ensure AI answers only from the document
            chat_history.append(
                {"role": "system", "content": "A document has been uploaded. Answer ONLY based on this document. "
                                              "If the answer is not in the document, say 'I couldn't find that information.'"})
            chat_history.append({"role": "user", "content": f"Document Content:\n{pdf_text}"})

            save_json()  # Save after upload

            if user_message:
                chat_history.append({"role": "user", "content": user_message})
        else:
            print("❌ File not found. Provide a valid path.")
            continue
    else:
        user_message = user_input

    # Append user message to chat history
    if user_message:
        chat_history.append({"role": "user", "content": user_message})
        save_json()

    # ✅ Ensure AI is aware of the uploaded document
    if pdf_context:
        chat_history.append({"role": "system", "content": "Remember, answer ONLY from the uploaded document. "
                                                          "If unsure, say 'I couldn't find that information.'"})
    save_json()

    # Send request to Ollama API
    response = requests.post(OLLAMA_URL, json={"model": "mistral", "messages": chat_history}, stream=True)

    # Read streamed response
    ai_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_data = json.loads(line.decode("utf-8"))
                ai_response += json_data.get("message", {}).get("content", "")
            except (json.JSONDecodeError, KeyError):
                print("❌ Error: Invalid JSON chunk received:", line)

    # Append AI response to chat history
    chat_history.append({"role": "assistant", "content": ai_response})
    save_json()

    print("\U0001F916 Buddy:", ai_response.strip())
