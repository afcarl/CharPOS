import numpy as np
from nltk.corpus import ptb
import cPickle

import models_common as common

START = "-start-"
END = "-end-"

# Throughout, "model" refers to the Markov model for a specific POS.
# It's a dictionary of bigrams to probability distributions of characters.

# Add one input-output observation to this model
def add_count(input, output, model):
	if input not in model:
		model[input] = {}
	model[input][output] = model[input].get(output, 0) + 1

# Add bigram observations for this word to the model.
# This model will be a dictionary of bigrams into a dictionary of characters
# into counts. A[i] represents the observations for bigram i, and A[i][j]
# represents the number of observations of bigram i leading to character j.
def add_counts(word, model):
	add_count(START, word[0], model)
	
	if len(word) >= 2:
		add_count(START + word[0], word[1], model)
	
	for i in range(len(word) - 2):
		add_count(word[i:i+2], word[i+2], model)
	
	add_count(word[-1], END, model)
	add_count(word[-2:], END, model)

# Perform add-delta smoothing on the model
def smooth(model, delta):
	# Set of everything you can transition to
	all_e = set().union(*[set(model[s].keys()) for s in model.iterkeys()])
	
	# Add peusdo-counts for all those cases
	for s in model:
		for e in all_e:
			model[s][e] = model[s].get(e, 0) + delta

# Transform the model so that it becomes a dictionary of tuples. Each tuple
# contains two lists: one of characters, the other of probabilities.
def open_as_probs(model):
	for cur_char in model.iterkeys():
		counts_to_probs(cur_char, model)
		
	#for cur_char, counts in model.iteritems():
	#	out_chars = counts.keys()
	#	out_counts = counts.values()
	#	s = float(sum(out_counts))
	#	out_counts = [a / s for a in out_counts]
	#	model[cur_char] = (out_chars, out_counts)

def counts_to_probs(key, model):
	obs = model[key].keys()
	counts = model[key].values()
	s = float(sum(counts))
	counts = [a / s for a in counts]
	model[key] = (obs, counts)

def observe_closed(word, model):
	model[word] = model.get(word, 0) + 1

# Obtain models per tag, training on the given sections of the WSJ
def make_word_model(scts):
	super_model = {tag: {} for tag in common.OPEN_CLASSES | common.CLOSED_CLASSES}
	
	def parse_file(f):
		for word, tag in ptb.tagged_words(f):
			if tag in common.OPEN_CLASSES:
				add_counts(word, super_model[tag])
			elif tag in common.CLOSED_CLASSES:
				observe_closed(word, super_model[tag])
	
	common.for_all_in_ptb_scts(scts, parse_file)
	
	for tag, model in super_model.iteritems():
		if tag in common.OPEN_CLASSES:
			# smooth(model, 1)
			open_as_probs(model)
		else:
			counts_to_probs(tag, super_model)
		
	return super_model

def save_model(super_model, fname=common.DEFAULT_MODEL_FILE):
	f = open(fname, "wb")
	cPickle.dump(super_model, f)
	f.close()

def load_model(fname=common.DEFAULT_MODEL_FILE):
	f = open(fname, "rb")
	model = cPickle.load(f)
	f.close()
	return model

def generate_word(tag, super_model):
	if tag in common.OPEN_CLASSES:
		return generate_open(super_model[tag])
	else:
		return get_out(tag, super_model)

def get_out(input, model):
	outs, probs = model[input]
	return np.random.choice(outs, p=probs)

def generate_open(model):
	res = "nothing"
	while START + res not in model:
		res = get_out(START, model) # Pick a starting character
	c = get_out(START + res, model)
	
	# While we haven't picked the end, pick new characters (making sure that
	# when appended to the word they form a bigram that we can start off of)
	while c != END:
		while res[-1] + c not in model:
			if len(res) == 1:
				c = get_out(START + res, model)
			else:
				c = get_out(res[-2:], model)
			
		res += c
		c = get_out(res[-2:], model)
		
	return res

def generate_test_batch(model):
	all = {}
	for tag in common.OPEN_CLASSES:
		all[tag] = []
		for i in range(16):
			all[tag].append(generate_word(tag, model))
	return all