#
import nltk
import streamlit as st
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


# Initialize stopwords and Lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# use a function for preprocessing each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence.lower())
    # Remove stopwords and punctuation
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    return [lemmatizer.lemmatize(word) for word in words]

# Load the text file
@st.cache_data
def load_text():
    try:
        file_path = "alice_in_wonderland.txt"
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().replace('\n', ' ')
    except FileNotFoundError:
        st.error(
            "Text file not found. please ensure file is in the same directory"
        )
        return ""


# Prepare corpus
@st.cache_resource
def prepare_corpus(text):
    sentences =sent_tokenize(text)
    return [preprocess(sentence) for sentence in sentences]


# define Function to calculate Jaccard similarity between two sentences
def jaccard_similarity(query, sentence):
    # Preprocess the sentences
    query_set = set(query)
    sentence_Set = set(sentence)

    # Calculate the intersection and union of the sets and return Jaccard similarity score
    if len(query_set.union(sentence_Set)) == 0:
        return 0
    return len(query_set.intersection(sentence_Set)) / len(query_set.union(sentence_Set))

# Find the most relevant sentence
def get_most_revelant_sentence(query, corpus,original_sentences):
    query = preprocess(query)
    max_similarity = 0
    best_sentence = "I couldn't find a relevant answer."
    for i, sentence in enumerate(corpus):
        similarity = jaccard_similarity(query, sentence)
        if similarity > max_similarity:
            max_similarity = similarity
            best_sentence = original_sentences[1]
    return best_sentence



# Main Function to streamlit APP
def main():
    st.title("Wonderland's Chatbot")
    st.write("Hello! Ask me anything related to Alice in Wonderland")

    with st.expander("Click me for suggestions"):
        st.write("""
        1. Who does Alice meet first in Wonderland?
        2. What is the signature of the bottle labeled 'Drink Me'
        3. How does Alice enter Wonderland?
        4. What is the Queen of Hearts known for?
        5. What game does the Queen of Hearts play with Alice?
        6. What advice does the Caterpillar give Alice?\
        7. Why did Alice follow the White Rabbit?
        """)

    text = load_text()
    if text:
        corpus = prepare_corpus(text)
        original_sentences = sent_tokenize(text)


        user_input = st.text_input("Enter your question:")

        if st.button("Submit"):
            if user_input.strip():
                response = get_most_revelant_sentence(user_input, corpus, original_sentences)
                st.write(f"Chatbot: {response}")
            else:
                st.write("Please enter a Question")




# Run the app
if __name__ =="__main__":
    main()