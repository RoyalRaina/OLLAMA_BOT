import requests
import json
import pdfplumber
import os
import re
import speech_recognition as sr
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# -------------------------- CONSTANTS --------------------------

# API Endpoint for AI Chat
OLLAMA_URL = "http://localhost:11501/api/chat"

# File to store chat history
CHAT_HISTORY_FILE = "chat_history.json"

# Limit document size to avoid overloading AI with too much context
MAX_DOC_LENGTH = 5000

# System message to guide the AI's behavior
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a highly precise AI assistant. Always reference uploaded documents for relevant questions. "
               "If a document is uploaded, assume it contains important user information. "
               "Give priority to chat history answers rather than searching online. "
               "If the answer is not in the document, say 'I couldn't find that information.'"
}

# -------------------------- GLOBAL VARIABLES --------------------------
chat_history = []  # Stores chat messages
pdf_context = ""   # Stores extracted text from the uploaded document


# -------------------------- FUNCTION DEFINITIONS --------------------------

def load_chat_history():
    """Loads previous chat history and document context from a file."""
    global chat_history, pdf_context
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

    # Ensure the system prompt is always at the beginning
    chat_history[0] = SYSTEM_PROMPT


def save_chat_history():
    """Saves chat history and document context to a file."""
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump({"chat_history": chat_history, "pdf_context": pdf_context}, file, indent=4)
        file.flush()
        os.fsync(file.fileno())  # Ensures immediate write to disk


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF using pdfplumber and OCR (Tesseract) if needed."""
    try:
        text = ""

        # Try extracting text using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if text.strip():
            return text.strip()  # Return text if extracted successfully

        # If no text found, use OCR (Tesseract) on images
        print("📄 No readable text found. Switching to OCR...")
        images = convert_from_path(pdf_path)  # Convert PDF pages to images
        ocr_text = ""

        for img in images:
            ocr_text += pytesseract.image_to_string(img) + "\n"  # Extract text using OCR

        return ocr_text.strip() if ocr_text.strip() else "❌ OCR couldn't extract readable text."

    except Exception as e:
        return f"Error processing PDF: {e}"


def get_user_input():
    """
    Asks the user for input. They can either type or speak.
    If they press Enter, it will use speech recognition.
    """
    recognizer = sr.Recognizer()

    print("\n🗣️ Press [Enter] to speak or type your message below.")
    user_input = input("Type your message (or press Enter to speak): ").strip()

    if not user_input:
        # If no text input, use speech recognition
        with sr.Microphone() as source:
            print("🎤 Speak now...")
            try:
                audio = recognizer.listen(source, timeout=5)  # Capture speech
                user_input = recognizer.recognize_google(audio)  # Convert speech to text
                print("🗣️ You said:", user_input)
            except sr.UnknownValueError:
                print("❌ Could not understand audio. Please type your message.")
                return input("Type here: ").strip()
            except sr.RequestError:
                print("❌ Error with the speech recognition API. Please type your message.")
                return input("Type here: ").strip()

    return user_input


def handle_pdf_upload(user_input):
    """
    Checks if the user wants to upload a PDF.
    If so, extracts its content and stores it in the global context.
    """
    global pdf_context
    match = re.search(r"\bupload\s+([^\s]+\.pdf)\b", user_input, re.IGNORECASE)
    if match:
        pdf_path = match.group(1)
        user_message = re.sub(r"\bupload\s+[^\s]+\.pdf\b", "", user_input, flags=re.IGNORECASE).strip()

        if os.path.exists(pdf_path):
            pdf_text = extract_text_from_pdf(pdf_path)
            print("\U0001F4C4 Extracted Text (Preview):\n", pdf_text[:1000])

            # Truncate the document text if it's too long
            if len(pdf_text) > MAX_DOC_LENGTH:
                pdf_text = pdf_text[:MAX_DOC_LENGTH] + "...\n[Document truncated]"

            pdf_context = pdf_text  # Store extracted text globally

            # Inform AI that document context should be used
            chat_history.append({"role": "system", "content": "A document has been uploaded. Answer ONLY based on this document. "
                                                              "If the answer is not in the document, say 'I couldn't find that information.'"})
            chat_history.append({"role": "user", "content": f"Document Content:\n{pdf_text}"})

            save_chat_history()

            if user_message:
                chat_history.append({"role": "user", "content": user_message})
            return True
        else:
            print("❌ File not found. Provide a valid path.")
            return False
    return False


def send_to_ai():
    """Sends the chat history to the AI model and retrieves the response."""
    response = requests.post(OLLAMA_URL, json={"model": "mistral", "messages": chat_history}, stream=True)

    ai_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_data = json.loads(line.decode("utf-8"))
                ai_response += json_data.get("message", {}).get("content", "")
            except (json.JSONDecodeError, KeyError):
                print("❌ Error: Invalid JSON chunk received:", line)

    chat_history.append({"role": "assistant", "content": ai_response})
    save_chat_history()
    return ai_response.strip()


def chat_loop():
    """Main interactive chat loop where the user interacts with the AI."""
    print("\U0001F916 Chat with Mistral! Type 'exit' to quit. Type 'upload [filename.pdf]' to analyze a PDF.")

    while True:
        user_input = get_user_input()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            save_chat_history()
            break

        # Check if the user is uploading a PDF
        if not handle_pdf_upload(user_input):
            chat_history.append({"role": "user", "content": user_input})
            save_chat_history()

        # Ensure AI is aware of uploaded document
        if pdf_context:
            chat_history.append({"role": "system", "content": "Remember, answer ONLY from the uploaded document. "
                                                              "If unsure, say 'I couldn't find that information.'"})
            save_chat_history()

        # Get response from AI and display it
        ai_response = send_to_ai()
        print("\U0001F916 Buddy:", ai_response)


def main():
    """Entry point of the script. Loads history and starts chat."""
    load_chat_history()
    chat_loop()


# Run the script
if __name__ == "__main__":
    main()
