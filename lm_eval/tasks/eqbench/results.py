import numpy as np
import math
def _process_question(predictions, references):  # This is a passthrough function
    string_label = ["entailment", "contradiction", "neutral"]
    predictions = (
        string_label.index(predictions[0]) if predictions[0] in string_label else 0
    )
    references = string_label.index(references[0])

    return (predictions, references)

def doc_to_text(doc):
    if doc['source'] == 'winogrande_xl':
        idx = doc["query"].index("_")
        return doc["query"][:idx].strip()
    elif doc['source'].startswith('mmlu'):
        choices = doc['choices']
        category = doc['source'][5:].replace('-', ' ')
        prepend_str = 'The following are multiple choice questions (with answers) about '+category+'\n\n'
        return prepend_str + doc['query'].strip() + '\nA. ' + choices[0] + '\nB. ' + choices[1] + '\nC. ' + choices[2] + '\nD. ' + choices[3] + '\nAnswer:'
    elif doc['source'].startswith('agieval'):
        return doc['query']


def doc_to_target(doc):
    if doc['source'] == 'winogrande_xl':
        return doc['gold']
    elif doc['source'].startswith('mmlu'):
        return doc['gold']
    elif doc['source'].startswith('agieval'):
        return doc['gold']


def doc_to_choice(doc):
    if doc['source'] == 'winogrande_xl':
        idx = doc["query"].index("_") + 1
        options = [doc["choices"][0][4:], doc["choices"][1][4:]]
        return [opt + doc["query"][idx:] for opt in options]
    elif doc['source'].startswith('mmlu'):
        return ["A", "B", "C", "D"]
    elif doc['source'].startswith('agieval'):
        return doc['choices']

def calc_score_for_question(pred, gold):
    # Implementing the scoring system from here:
    # https://github.com/EQ-bench/EQ-Bench/blob/main_v2_0/lib/scoring.py

    # Except with the slight variation that we are calculating each of
    # the 4 parts of each question individually.
    
    diff = abs(pred-gold)
    if diff == 0:
        scaled_diff = 0
    elif diff < 5:
        # S-shaped scaling function
        # https://www.desmos.com/calculator
        # 6.5\cdot\ \frac{1}{\left(1\ +\ e^{\left(-1.2\cdot\left(x-4\right)\right)}\right)}						
        scaled_diff = 6.5 * (1 / (1 + math.e ** (-1.2 * (diff-4))))
    else:
        scaled_diff = diff

    # this constant was calculated so that answering randomly on
    # the whole test produces a score of zero.
    adjust_const =  0.7477

    return 10 - scaled_diff * adjust_const
	

def process_question_acc(doc, results):
    lls, is_greedy = zip(*results)

    # retrieve choices in List[str] form, to compute choice lengths, etc.
    choices = doc_to_choice(doc)
    completion_len = np.array([float(len(i)) for i in choices])

    pred = np.argmax(lls)
    gold = doc_to_target(doc)    
    acc = calc_score_for_question(pred, gold)

    return acc

def process_question_acc_norm(doc, results):
    lls, is_greedy = zip(*results)

    # retrieve choices in List[str] form, to compute choice lengths, etc.
    choices = doc_to_choice(doc)
    completion_len = np.array([float(len(i)) for i in choices])

    pred_norm = np.argmax(lls / completion_len)
    gold = doc_to_target(doc)    
    acc_norm = calc_score_for_question(pred_norm, gold)

    return acc_norm


def aggregate_scores(items):    
    # This is effectively replicating the original scoring system here:
    # https://github.com/EQ-bench/EQ-Bench/blob/main_v2_0/lib/scoring.py

    # Except here we've processed the 4 parts for each question individually,
    # so the score tallying looks a little different.
    acc = (sum(items) - ((len(items) / 4) * 30)) / (len(items) / 4)
    return acc