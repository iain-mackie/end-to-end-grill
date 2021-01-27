
from genre import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq
import spacy


def get_entities(sentences, model):
    """ """
    prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_fairseq(model, sentences)
    return model.sample(
        sentences,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )



def get_preprocess_text(s):
    """ """
    s = s.replace('\t', ' ').replace('\n', ' ')
    s = s.replace('{', '(').replace('}', ')').replace('[', '(').replace(']', ')')
    return " ".join(s.split())


def get_sentence_chunks(s, max_words=100):
    """ """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(s)
    sentence_chunks = []
    sentence_chunk = ''
    current_word_count = 0
    for sent in doc.sents:
        sent_word_count = len(sent.text.split())
        if (current_word_count + sent_word_count) < max_words:
            sentence_chunk += ' ' + sent.text
            current_word_count += sent_word_count
        else:
            sentence_chunks.append('. ' + sentence_chunk)
            sentence_chunk = sent.text
            current_word_count = sent_word_count

    if len(sentence_chunk) > 0:
        sentence_chunks.append('. ' + sentence_chunk)

    return sentence_chunks


def build_sentences(documents, max_words=100):
    """ Builds sentences input require for GENRE from documents dict. """

    start_i = 0
    sentences = []
    index_map = {}
    for doc_id, text in documents.items():
        preprocess_text = get_preprocess_text(s=text)

        sentence_chunks = get_sentence_chunks(s=preprocess_text, max_words=max_words)
        sentences += sentence_chunks

        end_i = start_i + len(sentence_chunks) - 1
        index_map[doc_id] = (start_i, end_i)
        start_i = end_i + 1

    return index_map, sentences


def run_e2e_linking(documents, model_path, max_words=100):
    """ """
    print('prepocessing documents into passage chunks for GENRE.')
    index_map, sentences = build_sentences(documents, max_words=max_words)

    model = GENRE.from_pretrained(model_path, eval_bleu_args='', eval_bleu_detok_args='').eval()
    entities = get_entities(sentences, model=model)
    print(entities)

if __name__ == '__main__':

    genre_path = '/Users/iain/LocalStorage/coding/github/end-to-end-grill/genre_grill/data/fairseq_e2e_entity_linking_aidayago'
    # genre_path = '/nfs/trec_robust04/end-to-end-grill/genre_grill/data/fairseq_e2e_entity_linking_aidayago'

    from static import test_documents

    run_e2e_linking(documents=test_documents, model_path=genre_path)



