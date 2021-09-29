import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from model.prompt_encoder import PromptEncoder
from transformers.models.bert import BertForMaskedLM, BertTokenizer


class BertForMultiMask(nn.Module):

    def __init__(self, args, template, device):
        super().__init__()
        self.args = args
        self.device = device

        self.model = BertForMaskedLM.from_pretrained(
            self.args.pretrained_model).to(self.device)

        self.tokenizer = BertTokenizer.from_pretrained(
            self.args.pretrained_model)
        self.template = template
        self.spell_length = sum(self.template)

        self.embeddings = self.model.bert.get_input_embeddings()
        self.hidden_size = self.embeddings.embedding_dim
        self.pseudo_token_id = self.tokenizer.get_vocab()[
            self.args.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.prompt_encoder = PromptEncoder(
            self.template, self.hidden_size, self.tokenizer, self.device, args).to(self.device)

    def get_sentence(self, sentence, label, prompt_token):
        if sentence.count('[SEP]'):
            first_sentence, second_sentence = sentence.strip().split('[SEP]')
            left_text, right_text = second_sentence.strip().split('[MASK]')
            label_length = len(self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(label)))
            return [
                [self.tokenizer.cls_token_id]
                + prompt_token * self.template[0]
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + left_text))
                + [self.tokenizer.mask_token_id] * label_length
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + right_text))
                + prompt_token * self.template[1]
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + first_sentence))
                + [self.tokenizer.sep_token_id]
            ]
        else:
            left_text, right_text = sentence.strip().split('[MASK]')
            label_length = len(self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(label)))
            return [
                [self.tokenizer.cls_token_id]
                + prompt_token * self.template[0]
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + left_text))
                + [self.tokenizer.mask_token_id] * label_length
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + right_text))
                + prompt_token * self.template[1]
                + [self.tokenizer.sep_token_id]
            ]

    def get_label(self, label):
        return [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(label))]

    def embed_input(self, sentences):
        batch_size = sentences.shape[0]
        sentences_for_embedding = sentences.clone()
        sentences_for_embedding[(
            sentences == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(sentences_for_embedding)

        blocked_indices = (sentences == self.pseudo_token_id).nonzero().reshape(
            (batch_size, self.spell_length, 2))[:, :, 1]  # batch_size
        replace_embeds = self.prompt_encoder()
        for bidx in range(batch_size):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i],
                           :] = replace_embeds[i, :]
        return raw_embeds

    def forward(self, sentences, labels):
        batch_size = len(sentences)
        prompt_tokens = [self.pseudo_token_id]
        sentences = [torch.LongTensor(self.get_sentence(
            sentences[i], labels[i], prompt_tokens)).squeeze(0) for i in range(batch_size)]
        sentences = pad_sequence(
            sentences, True, padding_value=self.tokenizer.pad_token_id).long().to(self.device)

        inputs_embeds = self.embed_input(sentences)

        label_ids = [self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(label)) for label in labels]
        label_ids = torch.LongTensor(
            [label for label_ in label_ids for label in label_]).to(self.device)
        label_mask = (sentences == self.tokenizer.mask_token_id).nonzero().to(
            self.device)
        labels = torch.empty_like(sentences).fill_(-100).long().to(self.device)
        for i, idx in enumerate(label_mask):
            idx = idx.tolist()
            labels[idx[0], idx[1]] = label_ids[i]
        attention_mask = sentences != self.pad_token_id
        logits_mask = labels != -100

        output = self.model(inputs_embeds=inputs_embeds.to(self.device),
                            attention_mask=attention_mask.bool().to(self.device),
                            labels=labels.to(self.device))
        loss, logits = output.loss, output.logits

        logits = logits[logits_mask, :]

        return loss, logits
