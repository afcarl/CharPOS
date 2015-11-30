'''
WSJ Dataset loader.
Frederic Lafrance, Michael Noseworthy
'''

from nltk.corpus import ptb
import numpy as np
import cPickle as pyk

BATCH_SIZE = 128 # Number of examples in a batch
SEQUENCE_SIZE = 96 # Number of characters in an example
CONTEXT = 2 # Number of context words to include

CHARS = [u'~', u'!', u'#', u'$', u'%', u'&', u"'", u'*', u',', u'-', u'.', 
		u'/', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u':',
		u';', u'=', u'?', u'@', u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H',
		u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R', u'S', u'T',
		u'U', u'V', u'W', u'X', u'Y', u'Z', u'\\', u'`', u'a', u'b', u'c', u'd',
		u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p',
		u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z', 
		u'{', u'}', u' '] #Special characters for our input (plus ~ for unknown)
		
TAGS = [u'#', u'$', u"''", u',', u'-LRB-', u'-NONE-', u'-RRB-', u'.', u':', 
		u'CC', u'CD', u'DT', u'EX', u'FW', u'IN', u'JJ', u'JJR', u'JJS', u'LS',
		u'MD', u'NN', u'NNP', u'NNPS', u'NNS', u'PDT', u'POS', u'PRP', u'PRP$',
		u'RB', u'RBR', u'RBS', u'RP', u'SYM', u'TO', u'UH', u'VB', u'VBD',
		u'VBG', u'VBN', u'VBP', u'VBZ', u'WDT', u'WP', u'WP$', u'WRB', u'``']

I_TO_C = {i:c for i, c in enumerate(CHARS)}
C_TO_I = {c:i for i, c in enumerate(CHARS)}

I_TO_T = {i:c for i, c in enumerate(TAGS)}
T_TO_I = {c:i for i, c in enumerate(TAGS)}

TRAIN_SCTS = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
				'11', '12', '13', '14', '15', '16', '17', '18']
DEV_SCTS = ['19', '20', '21']
TEST_SCTS = ['22', '23', '24']

def get_example(sent, i):
	s = max(i - CONTEXT, 0)
	before = " ".join([w for w, _ in sent[s:i]])
	after = " ".join([w for w, _ in sent[i+1:i+1+CONTEXT]])
	sequence = before + "{" + sent[i][0] + "}" + after
	tag = sent[i][1]
	
	# Squish within the sequence size (either cut if it's too much,
	# or pad with ~'s otherwise)
	sequence = sequence[:SEQUENCE_SIZE]
	sequence = sequence + ('~' * (SEQUENCE_SIZE - len(sequence)))
	
	# One example: (seq_size vector, scalar)
	return (np.array([C_TO_I[c] for c in sequence]), T_TO_I[tag])

def empty_example():
	x = np.zeros(shape=(SEQUENCE_SIZE,), dtype=np.float32)
	y = T_TO_I[u'-NONE-']

	return (x, y)

def empty_batch():
	return (np.ndarray(shape=(BATCH_SIZE, SEQUENCE_SIZE), dtype=np.float32),
			np.ndarray(shape=(BATCH_SIZE, ), dtype=np.int8))

# Yield batches from the given sections
def get_batches(scts):
	batch_xs, batch_ys = empty_batch()
	ex_cnt = 0
	
	for sct in scts:
		print "Section " + sct 
		fs = [f for f in ptb.fileids() if f.startswith("WSJ/" + sct)]
		for f in fs:
			print "  File " + f + "...",
			# For each word in the sentences of the file,
			# create an example and add it to the batch.
			for sent in ptb.tagged_sents(f):
				for i in range(len(sent)):
					x, y = get_example(sent, i)
					batch_xs[ex_cnt] = x
					batch_ys[ex_cnt] = y
					
					# If we reach enough examples to form a batch, yield it now,
					# then start a new batch.
					ex_cnt += 1
					if ex_cnt == BATCH_SIZE:
						yield (batch_xs, batch_ys)
						batch_xs, batch_ys = empty_batch()
						ex_cnt = 0
	
	# If we have an incomplete batch at the end, pad it with nothings
	# and yield it.
	if ex_cnt != 0:
		while ex_cnt < BATCH_SIZE:
			x, y = empty_example()
			batch_xs[ex_cnt] = x
			batch_ys[ex_cnt] = y
			ex_cnt += 1
		yield (batch_xs, batch_ys)
		
	raise StopIteration

def write_set(name, scts):
	
	all_x = []
	all_y = []
	for x, y in get_batches(scts):
		all_x.append(x)
		all_y.append(y)
		
	f = open(name, "wb")
	pyk.dump((all_x, all_y), f)
	f.close()

if __name__ == '__main__':
	write_set("train", TRAIN_SCTS)
	write_set("dev", DEV_SCTS)
	write_set("test", TEST_SCTS)
	