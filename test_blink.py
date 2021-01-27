

if __name__ == '__main__':
    # print('test')
    # import sys
    #
    # sys.path.append("../BLINK")
    #
    # # '/Users/iain/LocalStorage/coding/github/end-to-end-grill'
    # # '/Users/iain/LocalStorage/coding/github/BLINK' #/blink/main_dense.py
    #
    # import blink.main_dense as main_dense
    # import argparse
    #
    # models_path = "/Users/iain/LocalStorage/coding/github/BLINK/models/"  # the path where you stored the BLINK models
    #
    # config = {
    #     "test_entities": None,
    #     "test_mentions": None,
    #     "interactive": False,
    #     "top_k": 10,
    #     "biencoder_model": models_path + "biencoder_wiki_large.bin",
    #     "biencoder_config": models_path + "biencoder_wiki_large.json",
    #     "entity_catalogue": models_path + "entity.jsonl",
    #     "entity_encoding": models_path + "all_entities_large.t7",
    #     "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
    #     "crossencoder_config": models_path + "crossencoder_wiki_large.json",
    #     "fast": False,  # set this to be true if speed is a concern
    #     "output_path": "logs/"  # logging directory
    # }
    #
    # args = argparse.Namespace(**config)
    #
    # models = main_dense.load_models(args, logger=None)
    #
    # data_to_link = [{
    #     "id": 0,
    #     "label": "unknown",
    #     "label_id": -1,
    #     "context_left": "".lower(),
    #     "mention": "Shakespeare".lower(),
    #     "context_right": "'s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty.".lower(),
    # },
    #     {
    #         "id": 1,
    #         "label": "unknown",
    #         "label_id": -1,
    #         "context_left": "Shakespeare's account of the Roman general".lower(),
    #         "mention": "Julius Caesar".lower(),
    #         "context_right": "'s murder by his friend Brutus is a meditation on duty.".lower(),
    #     }
    # ]
    #
    # _, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)
    #
    # print(predictions)
    # print(scores)

    import sys

    sys.path.append("/nfs/software/BLINK")

    import elq.main_dense as main_dense
    import argparse

    models_path = "/nfs/software/BLINK/models/"  # the path where you stored the BLINK models

    config = {
        "interactive": False,
        "biencoder_model": models_path + "elq_wiki_large.bin",
        "biencoder_config": models_path + "elq_large_params.txt",
        "cand_token_ids_path": models_path + "entity_token_ids_128.t7",
        "entity_catalogue": models_path + "entity.jsonl",
        "entity_encoding": models_path + "all_entities_large.t7",
        "output_path": "logs/",  # logging directory
        "faiss_index": "hnsw",
        "index_path": models_path + "faiss_hnsw_index.pkl",
        "num_cand_mentions": 10.0,
        "num_cand_entities": 10.0,
        "threshold_type": "joint",
        "threshold": -4.5,
    }

    args = argparse.Namespace(**config)

    models = main_dense.load_models(args, logger=None)

    data_to_link = [{
        "id": 0,
        "text": "paris is capital of which country?".lower(),
    },
        {
            "id": 1,
            "text": "paris is great granddaughter of whom?".lower(),
        },
        {
            "id": 2,
            "text": "who discovered o in the periodic table?".lower(),
        },
    ]

    predictions = main_dense.run(args, None, *models, test_data=data_to_link)
    print(predictions)