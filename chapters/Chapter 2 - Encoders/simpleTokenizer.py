import re

class SimpleTokenizer:
    def __init__(self, vocabulary):
        self.str_to_int = vocabulary
        self.int_to_str = {v: k for k, v in vocabulary.items()}

    def encode(self, text):
        parts = text.split('</s>')
        main_text = parts[0].lower()
    
        tokens = re.findall(r'\w+|[^\w\s]', main_text, re.UNICODE)
        if len(parts) > 1:  
            tokens.append('</s>')

    def decode(self, tokens):
        return ' '.join([self.int_to_str.get(token, '<unk>') for token in tokens])
    
vocabulary = {token: i for i, token in enumerate(['<pad>', '<unk>', '<s>', '</s>', 'hello', ',', 'how', 'are', 'you', '?', 'i', 'am', 'fine', 'thank', 'you', '.', 'goodbye', '!'])}
# print("Vocabulary:", vocabulary)  

tokenizer = SimpleTokenizer(vocabulary)
text = "yo, how are you? I am fine, thank you. goodbye!</s>"
encoded_text = tokenizer.encode(text)
decoded_text = tokenizer.decode(encoded_text)

# print("Final encoded:", encoded_text)
# print("Final decoded:", decoded_text)


# encoded_text returns the encoded version of the input text using the vocabulary
# decoded_text returns the decoded version of the encoded text using the vocabulary
# vocabulary contains the mapping of tokens to their corresponding integer values
# encode is written in such a way that the special tokens '<s>' and '</s>' are added to the tokens list if they are present in the input text
# decode is written in such a way that the special token '<unk>' is used if the token is not found in the vocabulary