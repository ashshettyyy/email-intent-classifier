import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import numpy as np

# Load model and tokenizer from your model repository
model_name = r"C:\Users\ashwi\email-intent-classifier\email_intent_classifier\email_intent_classifier"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load label mappings
try:
    with open(f"{model_name}/label_mapping.json") as f:
        mapping = json.load(f)
        label_names = mapping["label_names"]
except:
    # Fallback if label_mapping.json isn't available
    label_names = ["feedback", "information", "problem", "question", "request", "scheduling"]

# Define test inputs
subject = "Test subject"
body = "This is a test email body"

# Define prediction function
def predict_intent(subject, body):
    if not subject.strip() or not body.strip():
        return "Please enter both subject and body text."
    
    input_text = f"SUBJECT: {subject} BODY: {body}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().numpy()
    predicted_class = np.argmax(probs)
    
    # Emoji mapping for intents
    emoji_map = {
        "question": "❓",
        "request": "🙏",
        "scheduling": "📅",
        "information": "ℹ️",
        "problem": "⚠️",
        "feedback": "💬"
    }
    
    intent = label_names[predicted_class]
    confidence = float(probs[predicted_class])
    emoji = emoji_map.get(intent, "")
    
    # Format result
    result = f"## Intent: {emoji} {intent.upper()} ({confidence*100:.1f}%)\n\n"
    
    # Add recommendation
    recommendations = {
        "question": "This email contains a question requiring information. Prepare a response with relevant details.",
        "request": "This email contains a request for action. Determine if you can fulfill this request and provide a timeline.",
        "scheduling": "This email is about scheduling a meeting or event. Check availability and respond with confirmation or alternatives.",
        "information": "This email is sharing information. No immediate action may be required, but you might want to acknowledge receipt.",
        "problem": "This email reports an issue requiring troubleshooting. Escalate to the appropriate technical team.",
        "feedback": "This email contains feedback. Thank the sender and consider if any follow-up actions are needed."
    }
    
    result += f"### Recommended Action\n{recommendations.get(intent, '')}\n\n"
    
    # Add probability table
    result += "### Probability Distribution\n"
    result += "|Intent|Probability|\n|-|-|\n"
    
    # Sort probabilities
    sorted_probs = [(label, float(probs[i])) for i, label in enumerate(label_names)]
    sorted_probs.sort(key=lambda x: x[1], reverse=True)
    
    for label, prob in sorted_probs:
        result += f"|{label}|{prob*100:.1f}%|\n"
        
    return result

# Create Gradio interface
demo = gr.Interface(
    fn=predict_intent,
    inputs=[
        gr.Textbox(label="Email Subject", placeholder="Enter the email subject"),
        gr.Textbox(label="Email Body", placeholder="Enter the email body", lines=5)
    ],
    outputs=gr.Markdown(),
    title="Email Intent Classifier",
    description="Analyze emails to determine their primary intent or purpose.",
    examples=[
        ["Question about API documentation", "Hello team, I've been looking at your API docs and I can't find information about rate limits. Could you point me to the right section? Thanks, Developer"],
        ["Urgent issue with login page", "Support team, Our users are reporting that they can't log in to the system. The page just refreshes without any error message. This is affecting our business operations. Please help ASAP. Regards, Admin"],
        ["Team meeting next Tuesday", "Hi everyone, Let's have our weekly team meeting next Tuesday at 10am. We'll discuss the Q3 roadmap and project assignments. Let me know if this time works for you. Best, Manager"]
    ],
    allow_flagging="never"
)

# Launch app
demo.launch(share=True)