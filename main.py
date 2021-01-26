
from genre import GENRE
from genre_grill import genre_e2e_fairseq
from rel_grill import e2e
import json


def get_queries(qrels_path):
    """ Get list of unique queries from qrels file. """
    queries = []
    with open(qrels_path, 'r') as f:
        for line in f:
            q, _, _, _ = line.strip().split()
            if q not in queries:
                queries.append(q)
    return queries


def get_query_text(query, topics_path):
    """ Extract specific query text from Robust topic file for specific query (query id). """
    query_text = ""
    process_query = False
    with open(topics_path, 'r') as f_topics:
        for line in f_topics:
            if '<num>' == line[:5]:
                if line[13:].strip() == query:
                    process_query = True
                else:
                    process_query = False
            if process_query and ('<num>' != line[:5]):
                query_text += line.strip() + " "
                if '<desc>' in line:
                    query_text += '. '

    # Remove metadata
    query_text = query_text.replace('</top>', '').replace('<top>', '').replace('<title>', '') \
        .replace('<narr> Narrative:', '').replace('<desc> Description:', '').replace('<narr>', '') \
        .replace('<desc>', '').strip()
    return " ".join(query_text.split())


if __name__ == '__main__':
    qrels_path = '/Users/iain/LocalStorage/coding/complex_information_needs/robust04_runs/robust04.qrels'
    topics_path = '/Users/iain/LocalStorage/coding/complex_information_needs/robust04_runs/04.testset'
    queries = get_queries(qrels_path)

    genre_path = '/Users/iain/LocalStorage/coding/github/end-to-end-grill/genre_grill/data/fairseq_e2e_entity_linking_aidayago'
    genre_model = GENRE.from_pretrained(genre_path, eval_bleu_args='', eval_bleu_detok_args='').eval()

    rel_base_url = '/Users/iain/LocalStorage/coding/github/end-to-end-grill/rel_grill/data/'
    rel_wiki_year = '2019'
    mention_detection, tagger_ner, entity_disambiguation = e2e.get_REL_models(rel_base_url, rel_wiki_year)

    linking_data = {}
    for query in queries:

        query_text = get_query_text(query, topics_path)
        print('----------')
        print(query, query_text)
        linking_data[query] = {}
        linking_data[query]['text'] = query_text

        try:
            sentences = [query_text]
            genre_predictions = genre_e2e_fairseq.get_entities(sentences, model=genre_model)
            linking_data[query]['genre'] = [{'text': chain['text'], 'logprob': chain['logprob'].numpy().tolist()}
                                            for chain in genre_predictions[0]]
        except:
            print(query, 'genre fail')

        try:
            rel_predictions = e2e.get_entity_ranking_REL(query_text, mention_detection, tagger_ner, entity_disambiguation)
            linking_data[query]['rel'] = rel_predictions
        except:
            print(query, 'rel fail')


    out_path = '/Users/iain/LocalStorage/coding/github/end-to-end-grill/robust_queries.json'
    with open(out_path, 'w') as fp:
        json.dump(linking_data, fp, indent=4)

