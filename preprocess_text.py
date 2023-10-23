import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # Initialize Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Preprocess each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        # Tokenize sentence into words
        words = word_tokenize(sentence)
        
        # Remove stopwords and punctuation
        filtered_words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word.isalnum()]
        
        # Lemmatize words
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        
        # Join words back into sentence
        preprocessed_sentence = ' '.join(lemmatized_words)
        preprocessed_sentences.append(preprocessed_sentence)
    
    # Join preprocessed sentences back into text
    preprocessed_text = ' '.join(preprocessed_sentences)
    
    return preprocessed_text

# Function to extract information from text using NLP
def extract_information(text, query):
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    
    # Tokenize query
    query_tokens = word_tokenize(query.lower())
    
    # Initialize results list
    results = []
    
    # Search for query tokens in preprocessed text
    for token in query_tokens:
        if token in preprocessed_text:
            results.append(token)
    
    return results

# Example usage
text = "Scientific paper or article text goes here."
query = "What are the properties of celestial objects mentioned in the text?"

# Extract information from text using NLP
information = extract_information(text, query)

# Print extracted information
print(information)
