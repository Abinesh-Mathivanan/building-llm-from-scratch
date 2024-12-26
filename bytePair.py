from tiktoken import get_encoding

tokenizer = get_encoding("gpt2")
text = "beens_codes is so cool <|endoftext|>"

encoded_text = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("Final encoded:", encoded_text)

decoded_text = tokenizer.decode(encoded_text)
print("Final decoded:", decoded_text)


# bytepair encoding is a type of subword tokenization that is used to encode text data into smaller subword units
# it's is based on the frequency of subword units in the text data
# for example, the text 'aaabdaaabac' can be encoded as 'aa', 'ab', 'da', 'aa', 'ba', 'c'
# then the frequency of each subword unit is calculated and the most frequent pair is merged together
# this process is repeated until the desired vocabulary size is reached


with open("the-verdict.txt", "r", encoding="utf-8") as file:
    text = file.read()
tokenized_verdict = tokenizer.encode(text) 
print("Length of tokenized verdict:", len(tokenized_verdict))


# you could save the tokenized verdict for future use. Just remove comments below.
# with open("the-verdict-tokenized.txt", "w") as file:
#     file.write(" ".join([str(token) for token in tokenized_verdict]))
# print("Tokenized verdict saved to the-verdict-tokenized.txt")

#* I'll implement the Q/A Parsing of GPT
#? Might be technically less advanced. For educational purposes.
#? Feel free to extend by opening a pull request.


