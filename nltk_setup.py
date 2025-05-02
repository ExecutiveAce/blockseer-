import nltk
from textblob import download_corpora

# Download NLTK dependencies
nltk.download('punkt')
nltk.download('stopwords')

# Download TextBlob corpora
download_corpora.download_all()
