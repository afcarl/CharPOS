from nltk.corpus import ptb

OPEN_CLASSES = {'CD', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 
				'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
				
CLOSED_CLASSES = {'CC', 'DT', 'EX', 'FW', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP',
					'PRP$', 'RP', 'SYM', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB'}

TRAIN_SCTS = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
				'11', '12', '13', '14', '15', '16', '17', '18']

DEFAULT_MODEL_FILE = "word_model"

def for_all_in_ptb_scts(scts, call):
	for sct in scts:
		print "Section " + sct 
		fs = [f for f in ptb.fileids() if f.startswith("WSJ/" + sct)]
		for f in fs:
			print "  File " + f + "...",
			call(f)