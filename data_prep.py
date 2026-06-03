def load_and_prepare_data(filename, block_size):
    with open(filename, "r", encoding="utf-8") as f:#Read the text file
        text = f.read()

    chars = sorted(list(set(text)))#Build vocabulary
    vocab_size = len(chars)

    char2idx = {ch: i for i, ch in enumerate(chars)}#Create mappings from char to index and back
    idx2char = {i: ch for i, ch in enumerate(chars)}

    encoded = [char2idx[ch] for ch in text]#Encode entire text 

    #Create input-target pairs
    inputs = []
    targets = []
    for i in range(len(encoded) - block_size):
        inputs.append(encoded[i:i+block_size])
        targets.append(encoded[i+1:i+block_size+1])

    print(f"Loaded {len(text)} characters")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of sequences: {len(inputs)}")

    return inputs, targets, char2idx, idx2char, vocab_size