from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import nltk
import random
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

class HuggingFaceChatbot:
    def __init__(self):
        # Initialize Hugging Face GPT-2 model pipeline (for similarity check)
        self.model = pipeline("text-generation", model="gpt2", framework="pt")
        
        # Download required NLTK data for tokenization
        nltk.download('punkt', quiet=True)
        
        # Initialize intent classifier components
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()
        
        # Load intents and prepare training data
        self.patterns, self.responses, self.labels = self._prepare_training_data()
        
        # Train the intent classifier
        self._train_intent_classifier()
        
        # Initialize conversation history for context tracking
        self.conversation_history = []

    def _prepare_training_data(self):
        """
        Prepare training data for intent classification.
        """
        from intents import intent

        intents = intent  # Load intents data from 'intents.py'
        patterns = []
        responses = {}
        labels = []

        for intent in intents["intents"]:
            tag = intent["tag"]
            patterns.extend([(pattern, tag) for pattern in intent["patterns"]])
            responses[tag] = intent["responses"]
            labels.append(tag)

        return patterns, responses, labels

    def _train_intent_classifier(self):
        """
        Train the intent classifier using the provided patterns and labels.
        """
        X, y = zip(*self.patterns)
        X_tfidf = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_tfidf, y)

    def _classify_intent(self, user_input):
        """
        Classify the intent of the user's input.
        """
        X_input = self.vectorizer.transform([user_input.lower()])
        intent = self.classifier.predict(X_input)[0]
        prob = np.max(self.classifier.predict_proba(X_input))
        
        return intent, prob

    def respond(self, user_input):
        """
        Respond to user input based on intent classification.
        """
        intent, confidence = self._classify_intent(user_input)
        
        # If an intent is found (even with low confidence), respond based on intent
        if intent in self.responses:
            response = random.choice(self.responses[intent])
        else:
            response = "Sorry, I didn't understand that."

        # Update conversation history
        self.conversation_history.append(("user", user_input))
        self.conversation_history.append(("assistant", response))
        return response


# Initialize chatbot
chatbot = HuggingFaceChatbot()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_response():
    # Expect JSON payload from the front-end
    data = request.get_json()
    user_input = data.get('user_input')
    
    # Get chatbot response
    response = chatbot.respond(user_input)
    
    # Send the response back as JSON
    return jsonify({'response': response})


if __name__ == "__main__":
    app.run(debug=True)
