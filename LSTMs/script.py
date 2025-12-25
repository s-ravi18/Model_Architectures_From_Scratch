## Coding LSTMs from scratch;


import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # weight matrices
        self.Wf = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wc = np.random.randn(hidden_size, hidden_size + input_size) * 0.1

        # biases
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))

    def forward(self, x_t, h_prev, C_prev):
        # concatenate h(t-1) and x(t)
        concat = np.vstack((h_prev, x_t))

        # gates
        f_t = sigmoid(self.Wf @ concat + self.bf)
        i_t = sigmoid(self.Wi @ concat + self.bi)
        o_t = sigmoid(self.Wo @ concat + self.bo)
        C_hat = np.tanh(self.Wc @ concat + self.bc)

        # new cell and hidden state
        C_t = f_t * C_prev + i_t * C_hat
        h_t = o_t * np.tanh(C_t)

        return h_t, C_t

# ------------ Encoder --------------------------

class Encoder:
    def __init__(self, input_size, hidden_size):
        self.lstm = LSTMCell(input_size, hidden_size)

    def encode(self, X):
        h = np.zeros((self.lstm.hidden_size, 1))
        C = np.zeros((self.lstm.hidden_size, 1))

        for x_t in X:   ### Sequential scanning of the input and updating cell and hidden state.
            x_t = x_t.reshape(-1, 1)
            h, C = self.lstm.forward(x_t, h, C)

        return h, C  # context vector

# ------------ Decoder --------------------------

class Decoder:
    def __init__(self, input_size, hidden_size, output_size):
        self.lstm = LSTMCell(input_size, hidden_size)
        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))

    def decode(self, context_h, context_C, max_len, start_vector, vocab):
        h, C = context_h, context_C
        x_t = start_vector.reshape(-1, 1)

        outputs = []

        for _ in range(max_len):
            h, C = self.lstm.forward(x_t, h, C)
            y_pred = softmax(self.Wy @ h + self.by)
            token = np.argmax(y_pred)
            outputs.append(token)

            # next input = embedding of predicted token
            x_t = vocab[token].reshape(-1, 1)

        return outputs



# Fake embeddings for demonstration
embed = {
    "i": np.array([1,0,0], dtype=float),
    "love": np.array([0,1,0], dtype=float),
    "apples": np.array([0,0,1], dtype=float),

    # French tokens
    "<SOS>": np.array([1,1,1], dtype=float),
    "j'": np.array([1,0,1], dtype=float),
    "aime": np.array([0,1,1], dtype=float),
    "les": np.array([1,1,0], dtype=float),
    "pommes": np.array([0,1,0], dtype=float)
}

# vocab index for decoding
vocab_list = ["j'", "aime", "les", "pommes"]
vocab_vectors = [embed[v] for v in vocab_list]

# Encoder input (English sentence)
X = [
    embed["i"],
    embed["love"],
    embed["apples"]
]


encoder = Encoder(input_size=3, hidden_size=4)
decoder = Decoder(input_size=3, hidden_size=4, output_size=len(vocab_list))

context_h, context_C = encoder.encode(X)

# Decode to max length 4
output_ids = decoder.decode(
    context_h=context_h,
    context_C=context_C,
    max_len=4,
    start_vector=embed["<SOS>"],
    vocab=vocab_vectors
)

translated_tokens = [vocab_list[i] for i in output_ids]
print("Predicted translation:", translated_tokens)
