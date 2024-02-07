import numpy as np
import math

def doc_to_text(doc):
    return doc['query']

def doc_to_target(doc):
    return doc['gold']

def doc_to_choice(doc):
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
    score = round(10 - scaled_diff * adjust_const, 4)
    #print('score:', score)

    return score
	

def process_question_acc(doc, results):
    lls, is_greedy = zip(*results)

    # retrieve choices in List[str] form, to compute choice lengths, etc.
    choices = doc_to_choice(doc)

    pred = np.argmax(lls)
    gold = doc_to_target(doc)    
    acc = calc_score_for_question(pred, gold)

    return acc

def process_question_acc_norm(doc, results):
    lls, is_greedy = zip(*results)

    # retrieve choices in List[str] form, to compute choice lengths, etc.
    choices = doc_to_choice(doc)
    completion_len = np.array([float(len(str(i))) for i in choices])

    pred_norm = np.argmax(lls / completion_len)
    gold = doc_to_target(doc)    
    acc_norm = calc_score_for_question(pred_norm, gold)

    return acc_norm

def aggregate_scores(items):
    # This is effectively replicating the original scoring system here:
    # https://github.com/EQ-bench/EQ-Bench/blob/main_v2_0/lib/scoring.py
    #with open('test.out', 'w') as f:
    #    json.dump(items, f)
    print(items)

    # Except here we've processed the 4 parts for each question individually,
    # so the score tallying looks a little different.
    acc = 10 * (sum(items) - ((len(items) / 4) * 30)) / (len(items) / 4)
    return acc

def process_results_mcqa(doc, results):
    acc = process_question_acc(doc, results)
    #acc_norm = process_question_acc_norm(doc, results)

    return {
        "eqbenchv2": acc,
        #"eqbenchv2_norm": acc_norm,
    }