#!/usr/bin/env python3
import unicodedata
import re
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import os

plt.switch_backend('agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: {}'.format(device))


class Lang:
    """ Create a class to store words and indexes to easy change between them """

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=1000):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = torch.nn.Embedding(self.output_size, self.hidden_size)
        self.attn = torch.nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = torch.nn.functional.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)

        output = torch.nn.functional.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Reader(object):
    """ Responsible for manipulating the data in a prespecified format """

    def unicodeToAscii(self, s):
        """ Truns a unicode string to plain ASCII """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self, s):
        """ Lowercase, trim, and remove non-letter characters """
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def prepareData(self, lang1, lang2, input_file, reverse=False):
        """ Reads tab sepperated file with the sentence and it's translation """
        print("Reading lines...")

        # Read the file and split into lines
        lines = open(input_file, encoding='utf-8').\
            read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[self.normalizeString(s) for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)

        # Show results
        print("Read %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs


class Trainer:
    """ Responsible for providing methods to make us of the provided model (encoder & decoder) """

    def __init__(self, SOS_token=0, EOS_token=1, teacher_forcing_ratio=0.5, max_length=1000):
        self.max_length = max_length
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.unknown_words = []

    def tensorFromSentence(self, lang, sentence):
        """ Converts a sentence to an integer and save unknown words """
        indexes = []
        for word in sentence.split(' '):
            if word in lang.word2index:
                indexes.append(lang.word2index[word])
            else:
                self.unknown_words.append(word)
        indexes.append(self.EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def tensorsFromPair(self, pair):
        """ Converts a sentence of origin and input language to integers """
        input_tensor = self.tensorFromSentence(input_lang, pair[0])
        target_tensor = self.tensorFromSentence(output_lang, pair[1])
        return (input_tensor, target_tensor)

    def train(self, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
        """ trains the model """

        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length, encoder.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == self.EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def trainIters(self, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, n_iters, plot_every=200, print_every=None):
        """ Starts the training procedure for a specific number of iterations """

        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        training_pairs = [self.tensorsFromPair(random.choice(pairs))
                          for i in range(n_iters)]
        criterion = torch.nn.NLLLoss()

        print_loss_avg = 0
        for iter in tqdm(range(1, n_iters + 1)):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self.train(input_tensor, target_tensor, encoder,
                              decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if print_every is not None:
                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('(%d %d%%) %.4f' % (iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        # Plot losses
        plt.figure()
        fig, ax = plt.subplots()

        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(plot_losses)
        plt.show()
        return plot_losses

    def evaluate(self, encoder, decoder, sentence):
        """ Make a German prediction for a specified English sentence """
        with torch.no_grad():
            input_tensor = self.tensorFromSentence(input_lang, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()

            encoder_outputs = torch.zeros(self.max_length, encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[self.SOS_token]], device=device)

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)

            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == self.EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]

    def evaluateTest(self, pairs, encoder, decoder):
        """ Itterates the test dataset to output the predicted sentence """
        bleu_from = []
        bleu_to = []
        bleu_predict = []
        for lang1, lang2 in pairs:
            print('>', lang1)
            print('=', lang2)
            output_words, attentions = self.evaluate(encoder, decoder, lang1)
            output_sentence = ' '.join(output_words).replace(' .', '.').replace(' <EOS>', '')
            print('<', output_sentence)
            print('')
            bleu_from.append(lang1)
            bleu_to.append(lang2)
            bleu_predict.append(output_sentence)
        with open("data/bleu_from.txt", "w") as outfile:
            outfile.write("\n".join(bleu_from))
        with open("data/bleu_to.txt", "w") as outfile:
            outfile.write("\n".join(bleu_to))
        with open("data/bleu_predict.txt", "w") as outfile:
            outfile.write("\n".join(bleu_predict))

    def showAttention(self, input_sentence, output_words, attentions):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_sentence.split(' ') +
                           ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()

    def evaluateAndShowAttention(self, input_sentence, encoder, decoder):
        output_words, attentions = self.evaluate(
            encoder, decoder, input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        self.showAttention(input_sentence, output_words, attentions)


MAX_LENGTH = 1000
reader = Reader()
input_lang, output_lang, pairs = reader.prepareData('ger', 'eng', 'data/german/ger-train.txt', True)
print(random.choice(pairs))

hidden_size = 256
filename = 'data/pytorchmodel.dict'

learning_rate = 0.01

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, max_length=MAX_LENGTH).to(device)
encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)

trainer = Trainer(max_length=MAX_LENGTH)

if not os.path.exists(filename):
    loss = trainer.trainIters(pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, 95000)
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict()
    }, filename)

checkpoint = torch.load(filename)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

test_input_lang, test_output_lang, test_pairs = reader.prepareData('ger', 'eng', 'data/german/ger-test.txt', True)
trainer.evaluateTest(test_pairs, encoder, decoder)
