import pickle
import argparse
from genre.utils import (
    get_micro_precision,
    get_micro_recall,
    get_micro_f1,
    get_macro_precision,
    get_macro_recall,
    get_macro_f1,
)

def get_metrics(guess_entities, gold_entities):
    micro_p = float(get_micro_precision(guess_entities, gold_entities))
    micro_r = float(get_micro_recall(guess_entities, gold_entities))
    micro_f1 = float(get_micro_f1(guess_entities, gold_entities))
    macro_p = float(get_macro_precision(guess_entities, gold_entities))
    macro_r = float(get_macro_recall(guess_entities, gold_entities))
    macro_f1 = float(get_macro_f1(guess_entities, gold_entities))
    return (micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1)
    
def get_genre_eval(guess_entities, gold_entities, output_folder, run_name):
    output_file = open(f"{output_folder}{run_name}_genre_eval_output.csv","w")
    output_file.write("doc_id\tmicro_p\tmicro_r\tmicro_f1\tmacro_p\tmacro_r\tmacro_f1\n")
    split_d = {}
    for prediction in guess_entities:
        doc_id = prediction[0]
        if doc_id not in split_d:
            split_d[doc_id] = {
                "guess":[],
                "gold":[],
            }
        split_d[doc_id]["guess"].append(prediction)
        
    for prediction in gold_entities:
        doc_id = prediction[0]
        if doc_id not in split_d:
            split_d[doc_id] = {
                "guess":[],
                "gold":[],
            }
        split_d[doc_id]["gold"].append(prediction)
    
    for doc_id, to_eval in split_d.items():
        micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1 = get_metrics(to_eval["guess"], to_eval["gold"])
        output_file.write(f"{doc_id}\t{micro_p:0.4f}\t{micro_r:0.4f}\t{micro_f1:0.4f}\t{macro_p:0.4f}\t{macro_r:0.4f}\t{macro_f1:0.4f}\n")
    
    micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1 = get_metrics(guess_entities, gold_entities)
    output_file.write(f"Overall\t{micro_p:0.4f}\t{micro_r:0.4f}\t{micro_f1:0.4f}\t{macro_p:0.4f}\t{macro_r:0.4f}\t{macro_f1:0.4f}\n")
    output_file.close()
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='enter run and qrel file')
    parser.add_argument('run_file', help='run file as pickle')
    parser.add_argument('qrel_file', help='qrel file as pickle')
    parser.add_argument('output_folder')
    parser.add_argument('run_name')
    
    args = parser.parse_args()
    run_file = args.run_file
    qrel_file = args.qrel_file
    output_folder = args.output_folder
    if output_folder[-1:] != "/":
        output_folder += "/"
    run_name = args.run_name
    
    with open(run_file,'rb') as f:
        guess_entities = pickle.load(f)
    f.close()
    
    with open(qrel_file, 'rb') as g:
        gold_entities = pickle.load(g)
    g.close()
    
    get_genre_eval(guess_entities, gold_entities,output_folder, run_name)