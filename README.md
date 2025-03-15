---
##Title: Email Intent Classifier
short_description: NLP model automatically classifying emails based on intent
---

##Problem: 
Email overload is a significant problem in businesses. Manually sorting and prioritizing emails takes valuable time from employees. Automatically categorizing emails by intent helps with -

    Prioritization of urgent matters
    Routing to appropriate departments
    Providing quick responses
    Managing workflow more efficiently

##Solution: 
This app uses a fine-tuned DeBERTa transformer model to classify emails into different intent categories:
- Question: Emails asking for information Request: Emails asking for action or approval 
- Information: Emails sharing knowledge or updates Scheduling: Emails about setting up meetings or events Other: Any other intent

##How to Use:

    Enter or paste an email text into the input box with subject
    Click "Submit"
    View the predicted intent categories and their probability/confidence scores

##Model Details: 
This model was fine-tuned on an email dataset with the following specifications:
Base Model: DistilBERT (base-uncased) 
Training Dataset: Synthetic dataset of categorized emails across 6 intent categories

##Limitations: 
Very short or ambiguous emails may be classified with lower confidence Emails with multiple intents may be classified based on the dominant intent

##Deployment: 
This application is deployed using Hugging Face Spaces with Gradio. The model weights are stored in the Hugging Face Model Hub.

## Technologies Used
- Python
- Hugging Face Transformers
- PyTorch
- Pandas
- Numpy
