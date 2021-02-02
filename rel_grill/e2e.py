
from REL.mention_detection import MentionDetection
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import load_flair_ner
import sys
import json
import os
import datetime
#import spacy


def get_REL_models(rel_base_url, rel_wiki_year):
    """ """
    rel_model_path = "{}/ed-wiki-{}/model".format(rel_base_url, rel_wiki_year)
    wiki_version = "wiki_" + str(rel_wiki_year)

    mention_detection = MentionDetection(rel_base_url, wiki_version)
    tagger_ner = load_flair_ner("ner-fast")
    config = {
        "mode": "eval",
        "model_path": rel_model_path,
    }
    entity_disambiguation = EntityDisambiguation(rel_base_url, wiki_version, config)
    return mention_detection, tagger_ner, entity_disambiguation

def get_entity_ranking_REL(input_json, mention_detection, tagger_ner, entity_disambiguation):
    """ Get entity ranking using REL. """
    #input_text = {'0': [text, []]}
    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp(text)
    # input_text = {str(i): [sent.text, []]for i, sent in enumerate(doc.sents)}
    processed_json = {doc_id: [text,[]] for doc_id, text in input_json.items()}

    mentions_dataset, n_mentions = mention_detection.find_mentions(processed_json, tagger_ner)
    predictions, timing = entity_disambiguation.predict(mentions_dataset)
    
    output_dict = {}
    for doc_id, mentions in predictions.items():
        if doc_id not in output_dict:
            output_dict[doc_id] = []

        for mention in mentions:
            valid_links = []
            valid_scores = []
            scores = mention["scores"]
            links = mention["candidates"]
            for i in range(len(scores)):
                if float(scores[i]) >= 0.0:
                    valid_links.append(links[i])
                    valid_scores.append(float(scores[i]))
            mention_dict = {
                "mention": mention["mention"],
                "pred": mention["prediction"],
                "links": valid_links,
                "scores": valid_scores
            }
            output_dict[doc_id].append(mention_dict)
    return output_dict


if __name__ == '__main__':

    #assuming running from end-to-end folder
    #model_path = './rel_grill/data'
    rel_wiki_year = '2019'
    logging_name = 'rel_log.text'
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    model_path = sys.argv[3]

    print('input_folder: {}'.format(input_folder))
    print('output_folder: {}'.format(output_folder))
    print('model_path: {}'.format(model_path))

    if os.path.isdir(output_folder) == False:
        os.mkdir(output_folder)

    log_path = os.path.join(output_folder, logging_name)
    with open(log_path, 'a+') as f_log:
        f_log.write('REL START PROCESS: {}\n'.format(datetime.datetime.now()))
    f_log.close()

    print('load models')
    mention_detection, tagger_ner, entity_disambiguation = get_REL_models(model_path, rel_wiki_year)

    for file in os.listdir(input_folder):
        if file[-5:] == ".json":
            with open(f"{input_folder}/{file}") as f:
                input_json = json.load(f)
            f.close()

            try:
                predictions = get_entity_ranking_REL(input_json, mention_detection, tagger_ner, entity_disambiguation)
                with open(log_path, 'a+') as f_log:
                    f_log.write(f"SUCCESS:\t{input_folder}\t{file}\t{datetime.datetime.now()}\n")
                f_log.close()
            except Exception as e:
                print('FAIL in input_folder: {}, file: {}'.format(input_folder, file))
                print(e)
                with open(log_path, 'a+') as f_log:
                    f_log.write(f"FAIL:\t{input_folder}\t{file}\n")
                f_log.close()
                continue

            output_path = os.path.join(output_folder, "{}_genre.json".format(file[:-5]))
            print('writing: {}'.format(output_path))
            with open(output_path,"w") as g:
                json.dump(predictions, g, indent=4)
            g.close()
    