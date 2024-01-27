import nltk
from nltk.tokenize import word_tokenize
import string

# Download the necessary NLTK models (only needs to be done once)
nltk.download('punkt')

class Tokenizer:
    def __init__(self):
        self.stoi = None
        self.itos = None

    def build_vocab(self, text):
        """
        Build vocabulary (stoi and itos) from the given text.
        """
        vocab = set(text)
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def preprocess(self, text):
        """
        Preprocess the text: lowercasing, removing punctuation, etc.
        """
        # # Convert text to lowercase
        # text = text.lower()

        # # Remove punctuation
        # text = text.translate(str.maketrans('', '', string.punctuation))

        return text

    def word_tokenize(self, text):
        """
        Tokenizes the input text into words.
        """
        # Preprocess the text
        preprocessed_text = self.preprocess(text)

        # Tokenize the text
        tokens = word_tokenize(preprocessed_text)

        return tokens

    def char_encode(self, s):
        """
        Encode the string s into a list of integers using stoi dictionary.
        """
        return [self.stoi[ch] for ch in s]

    def char_decode(self, l):
        """
        Decode the list of integers l into a string using itos dictionary.
        """
        return ''.join([self.itos[i] for i in l])

def read_file(file_path):
    """
    Reads text from a file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Example usage
if __name__ == "__main__":
    tokenizer = Tokenizer()

    # Read text from file
    file_path = 'input-03.txt'
    sample_text = read_file(file_path)

    # Build vocabulary from the sample text
    tokenizer.build_vocab(sample_text)

    # Word tokenization
    word_tokens = tokenizer.word_tokenize(sample_text)
    print("Word Tokens:", word_tokens)

    # # Character encoding and decoding
    # char_encoded = tokenizer.char_encode(sample_text)
    # print("Character Encoded:", char_encoded)
    # char_decoded = tokenizer.char_decode(char_encoded)
    # print("Character Decoded:", char_decoded)
