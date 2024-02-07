import numpy as np
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

def process_question(doc, results):
    lls, is_greedy = zip(*results)

    # retrieve choices in List[str] form, to compute choice lengths, etc.
    choices = doc_to_choice(doc)
    completion_len = np.array([float(len(i)) for i in choices])

    pred = np.argmax(lls)
    pred_norm = np.argmax(lls / completion_len)
    gold = doc_to_target(doc)

    acc = 1.0 if pred == gold else 0.0
    acc_norm = 1.0 if pred_norm == gold else 0.0

    result_dict = {
        {"acc": acc},
        {"acc_norm": acc_norm}
    }
    return result_dict


def aggregate_scores(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        paragraph_id = doc["idx"]["paragraph"]
        question_id = doc["idx"]["question"]
        if (paragraph_id, question_id) not in question_scoring_dict:
            question_scoring_dict[(paragraph_id, question_id)] = []

        gold_label = doc["label"] == 1

        question_scoring_dict[(paragraph_id, question_id)].append(gold_label == pred)
    acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
    return acc

