import numpy as np
import tokenization

class DataHolder:
    def __init__(self):
        vocab_file = 'vocab.txt'
        vocab = tokenization.load_vocab(vocab_file=vocab_file)
        tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)
        path = 'train_processed.txt'

        train_file = open(path, 'r', encoding='utf-8')
        lines = train_file.read().split('\n')

        max_length = 0

        for i in range(len(lines)):
            TK = lines[i].split(' \t')

            if max_length < len(TK[0]):
                max_length = len(TK[0])

        max_length += 1

        self.input_ids = np.zeros(shape=[len(lines), max_length], dtype=np.int32)
        self.input_mask = np.zeros(shape=[len(lines), max_length], dtype=np.int32)
        self.label = np.zeros(shape=[len(lines)], dtype=np.int32)

        for i in range(len(lines) - 1):
            TK = lines[i].split(' \t')
            if len(TK) != 2:
                TK = lines[i].split('\t')

            sentence = TK[0]
            token = tokenizer.tokenize(sentence)
            tk_ids = tokenization.convert_tokens_to_ids(vocab=vocab, tokens=token)

            for j in range(len(tk_ids)):
                self.input_ids[i, j + 1] = tk_ids[j]
                self.input_mask[i, j + 1] = 1
            self.input_ids[i, 0] = tokenization.convert_tokens_to_ids(vocab=vocab, tokens=['[CLS]'])[0]
            self.input_mask[i, 0] = 1
            self.label[i] = int(TK[1])

        path = 'test_processed.txt'

        test_file = open(path, 'r', encoding='utf-8')
        lines = test_file.read().split('\n')

        max_length = 0

        for i in range(len(lines)):
            TK = lines[i].split(' \t')

            if max_length < len(TK[0]):
                max_length = len(TK[0])

        print(max_length)
        max_length += 1

        self.test_input_ids = np.zeros(shape=[len(lines), max_length], dtype=np.int32)
        self.test_input_ids_masking = np.zeros(shape=[len(lines), max_length], dtype=np.int32)
        self.test_label = np.zeros(shape=[len(lines)], dtype=np.int32)

        for i in range(len(lines) - 1):
            TK = lines[i].split(' \t')
            if len(TK) != 2:
                TK = lines[i].split('\t')

            sentence = TK[0]
            token = tokenizer.tokenize(sentence)
            tk_ids = tokenization.convert_tokens_to_ids(vocab=vocab, tokens=token)

            for j in range(len(tk_ids)):
                self.test_input_ids[i, j + 1] = tk_ids[j]
                self.test_input_ids_masking[i, j + 1] = 1
            self.test_input_ids[i, 0] = tokenization.convert_tokens_to_ids(vocab=vocab, tokens=['[CLS]'])[0]
            self.test_input_ids_masking[i, 0] = 1

            self.test_label[i] = int(TK[1])

        self.Batch_Size = 8

        self.random_idx = np.array(range(self.label.shape[0]), dtype=np.int32)
        np.random.shuffle(self.random_idx)

        self.Batch_Idx = 0
        self.Test_Batch_Idx = 0

    def next_random_batch(self):
        if self.Batch_Idx >= 80000:
            np.random.shuffle(self.random_idx)
            self.Batch_Idx = 0

        x = np.zeros(shape=[self.Batch_Size, self.input_ids.shape[1]], dtype=np.int32)
        x_mask = np.zeros(shape=[self.Batch_Size, self.input_ids.shape[1]], dtype=np.int32)
        Y = np.zeros(shape=[self.Batch_Size, 2], dtype=np.float32)

        for i in range(self.Batch_Size):
            idx = self.random_idx[self.Batch_Idx]

            x[i] = self.input_ids[idx]
            x_mask[i] = self.input_mask[idx]
            Y[i, self.label[idx]] = 1

            self.Batch_Idx += 1

        return x, x_mask, Y

    def next_test_batch(self):
        if self.Test_Batch_Idx + self.Batch_Size >= self.label.shape[0]:
            return False, 0, 0, 0

        x = np.zeros(shape=[self.Batch_Size, self.test_input_ids.shape[1]], dtype=np.int32)
        x_masking = np.zeros(shape=[self.Batch_Size, self.test_input_ids.shape[1]], dtype=np.int32)
        Y = np.zeros(shape=[self.Batch_Size], dtype=np.float32)

        for i in range(self.Batch_Size):
            idx = self.Test_Batch_Idx

            x_masking[i] = self.test_input_ids_masking[idx]
            x[i] = self.test_input_ids[idx]
            Y[i] = self.test_label[idx]

            self.Test_Batch_Idx += 1

        return True, x, x_masking, Y
