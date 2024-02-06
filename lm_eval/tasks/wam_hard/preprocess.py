def doc_to_text(doc):
	if doc['source'] == 'winogrande_xl':
		idx = doc["query"].index("_") + 1
		return doc["query"][idx:].strip()
	elif doc['source'].startswith('mmlu'):
		choices = doc['choices']
		"{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:"
		return doc['query'].strip() + '\nA. ' + choices[0] + '\nB. ' + choices[1] + '\nC. ' + choices[2] + '\nD. ' + choices[3] + '\nAnswer:'
	elif doc['source'].startswith('agieval'):
		return doc['query']


def doc_to_target(doc):
	return doc['gold']


def doc_to_choice(doc):
	if doc['source'] == 'winogrande_xl':
		idx = doc["query"].index("_")
		options = [doc["choices"][0][4:], doc["choices"][1][4:]]
		return [doc["query"][:idx] + opt for opt in options]
	elif doc['source'].startswith('mmlu'):
		return ["A", "B", "C", "D"]
	elif doc['source'].startswith('agieval'):
		return doc['choices']
