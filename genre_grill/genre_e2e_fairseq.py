
from genre import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq
import spacy



def run_genre_model(sentences, model, beam=5):
    """ Run GENRE e2e model returning on sentences (list of strings). Returning GENRE formatted outputs. """
    prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_fairseq(model, sentences)
    return model.sample(sentences,
                        beam=beam,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                        )


def get_entity_links_from_genre_output(output):
    """ Parse GENRE outputs into unique entity links. """
    # Store entity links for each sentence.
    sentences_links = []
    # Loop over sentence beams (i.e. default is 5 for GENRE).
    for sentence_beams in output:
        # Store entity links for all beams.
        beam_links = []
        # Loop over sentence beams.
        for beam in sentence_beams:
            link = {}
            # Parse logic based on GENRE special characters, i.e { mention } [ link ]
            mention_bool = False
            link_bool = False
            mention_text = ''
            link_text = ''
            for i, char in enumerate(beam['text']):
                if char == '{':
                    mention_bool = True
                    # Store character index for ordering.
                    link['i'] = i
                elif char == '}':
                    mention_bool = False
                    # Add mention.
                    link['mention'] = mention_text[1:].strip()
                    mention_text = ''
                elif char == '[':
                    link_bool = True
                elif char == ']':
                    link_bool = False
                    # Add link.
                    link['link'] = link_text[1:].strip()
                    # Add score.
                    link['score'] = beam['logprob'].cpu().numpy().tolist()
                    # If new 'link' and 'mention' add 'beam_links'
                    if (link['link'] not in [d['link'] for d in beam_links]) or \
                            (link['mention'] not in [d['mention'] for d in beam_links]):
                        beam_links.append(link)
                    link_text = ''
                    link = {}

                # Updates strings if in parse mode.
                if mention_bool:
                    mention_text += char
                if link_bool:
                    link_text += char

        # Add beam_links to sentences_links (sorting by char index + remove 'i' keys)
        sentences_links.append([{'mention': d['mention'], 'link': d['link'], 'score': d['score']}
                                for d in sorted(beam_links, key=lambda k: k['i'])])

    return sentences_links


def map_sentence_links_to_documents(sentences_links, index_map):
    """ Build entity links mapping {doc_id: [entity links]}."""
    entity_links = {}
    for doc_id, indexes in index_map.items():
        doc_entity_links = []
        for i in range(indexes[0], indexes[1]+1):
            doc_entity_links += sentences_links[i]
        entity_links[doc_id] = doc_entity_links
    return entity_links


def get_preprocess_text(s):
    """ Basic string preprocessing for GENRE model, i.e. remove tabs and newlines and GENRE special characters: []{}. """
    s = s.replace('\t', ' ').replace('\n', ' ')
    s = s.replace('{', '(').replace('}', ')').replace('[', '(').replace(']', ')')
    return " ".join(s.split())


def get_sentence_chunks(s, max_words=50):
    """ Build sentence chunks for GENRE input. """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(s)
    sentence_chunks = []
    sentence_chunk = ''
    current_word_count = 0
    #TODO - what if sentence > max_words (logic will fail)

    # Loop over sentences.
    for sent in doc.sents:
        sent_word_count = len(sent.text.split())
        # If next sentence does not exceed 'max_words' append to sentence_chunk
        if (current_word_count + sent_word_count) < max_words:
            sentence_chunk += ' ' + sent.text
            current_word_count += sent_word_count
        # Add sentence 'sentence_chunk' to 'sentence_chunks'
        else:
            sentence_chunks.append('. ' + sentence_chunk)
            sentence_chunk = sent.text
            current_word_count = sent_word_count

    # Create fine sentence_chunk if any remaining.
    if len(sentence_chunk) > 0:
        sentence_chunks.append('. ' + sentence_chunk)

    return sentence_chunks


def build_sentences(documents, max_words=100):
    """ Builds sentences input require for GENRE from documents dict. """
    start_i = 0
    sentences = []
    doc_to_sentence_map = {}
    for doc_id, text in documents.items():
        preprocess_text = get_preprocess_text(s=text)
        # Updates sentences with processed and chunked sentences.
        sentence_chunks = get_sentence_chunks(s=preprocess_text, max_words=max_words)
        sentences += sentence_chunks
        # Update doc_to_sentence_map with start and end index for sentences.
        end_i = start_i + len(sentence_chunks) - 1
        doc_to_sentence_map[doc_id] = (start_i, end_i)
        start_i = end_i + 1

    return doc_to_sentence_map, sentences


def run_genre_e2e_linking(documents, model_path, max_words=50, beam=5):
    """ Build entity linking given document using GENRE. """
    print('Prepocessing documents into passage chunks for GENRE.')
    index_map, sentences = build_sentences(documents, max_words=max_words)
    print('Running GENRE model.')
    model = GENRE.from_pretrained(model_path, eval_bleu_args='', eval_bleu_detok_args='').eval()
    output = run_genre_model(sentences=sentences, model=model, beam=beam)
    print('Parsing GENRE output to entity links')
    sentences_links = get_entity_links_from_genre_output(output)
    print('Mapping entity links back to documents.')
    return map_sentence_links_to_documents(sentences_links, index_map)


if __name__ == '__main__':
    from static import test_documents

    genre_path = './data/fairseq_e2e_entity_linking_aidayago'
    entity_links = run_genre_e2e_linking(documents=test_documents, model_path=genre_path)
    print(entity_links)

