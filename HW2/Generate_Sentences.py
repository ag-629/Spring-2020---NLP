import sys, string, math, numpy as np

def get_data(path):
	'''
	Reads in and cleans the data.
	Removes punctuation and newline characters

	input: text file
	'''
	try:
		with open(path, 'r', encoding='utf8') as f:
			#punct = string.punctuation+'’‘”“–'
			data = []
			#trans = str.maketrans('','',punct)
			for line in f.readlines():
				if not line.startswith('\n'):
					data.append('<s> ' + line.rstrip() + ' </s>')
			return data
	except IOError:
		print('Could not find file at: ', path , 'check path and try again.')
		sys.exit()

def update_counts(counts, w):
	'''
	increments word counts in dictionary

	input: word counts dictionary
	'''
	if w in counts:
		counts[w] = counts[w] + 1
	else:
		counts[w] = 1

def train(data):
	'''
	Creates the bigram distribution and word counts

	Input: text data with <s> and </s> added to each line

	Returns: bigram counts, a nested dictionary representing each row of the bigram matrix
	            word (type) counts, a dictionary of words and their counts
	'''
	bigram_counter = {}
	word_counts = {}
	for sent in data:
		sent = sent.lower().split()
		#add <s> for each iteration
		update_counts(word_counts, '<s>')
		for first, second in zip(sent[:-1], sent[1:]):
			#update only 'second' to avoid double counting
			#<s> already added
			update_counts(word_counts, second)
			if first in bigram_counter.keys():
				update_counts(bigram_counter[first], second)
			else:
				bigram_counter[first] = {second : 1}
	return bigram_counter, word_counts

def build_dist(bigram_counts, word_counts):
	'''
	builds a dictionary of distributions of bigrams
	P((w_i | w_(i-1) / w_(i-1))

	inputs: bigrams counts: counts of bigram pairs in data set
			word counts: counts of words in data

	returns: nested dictionary of the distribution whose keys are the bigrams
	'''
	dist = {}
	for k1 in bigram_counts.keys():
		for k2 in bigram_counts[k1]:
			if k1 in dist.keys():
				dist[k1][k2] = (bigram_counts[k1][k2] / word_counts[k1])
			else:
				dist[k1] = {k2: (bigram_counts[k1][k2] / word_counts[k1])}
	return dist

def generate_sentence(dist):
	'''
	take in bigrams and and words counts to generate a sentence

	inputs: bigram_counts and word (type) counts

	returns: a sentence string
	'''
	sentence = ''
	curr_key = '<s>'
	while curr_key != '</s>':
		probs = []
		words = []
		for k in dist[curr_key]:
			probs.append(dist[curr_key][k])
			words.append(k)
		word_to_add = np.random.choice(words, p = probs)
		if word_to_add != '</s>':
			sentence += word_to_add + ' '
		curr_key = word_to_add
	return sentence

def compute_probability(sentence, word_counts, bigram_counts):

 	'''
 	Comutes probability of a sentence given the bigrams and word counts from training data

	inputs: bigram pair counts, total word (type) counts, test data, 
	normalize flag to return a normalized probability

	returns: dictionaries of sentence index, probability pairs
 	'''
 	probability = 0
 	sent_len = 0
 	vocab_length = len(word_counts)
 	sent = sentence.lower().split()
 		#Create bigrams from each sentence in test data
 	sent_len = len(sent)
 	#initialize to 0 for log probability, not 1
 	sent_prob = 0
 	for bigram in zip(sent[:-1], sent[1:]):
 		if bigram[0] in bigram_counts.keys():
 			if bigram[1] in bigram_counts[bigram[0]]:
 				numerator = 1 + bigram_counts[bigram[0]][bigram[1]]
 				denominator = word_counts[bigram[0]] + vocab_length
 				frac = numerator / denominator
 				sent_prob += math.log(frac)
 			else:
 				print(bigram)
 				numerator = 1
 				denominator = word_counts[bigram[0]] + vocab_length
 				frac = numerator / denominator
 				sent_prob += math.log(frac)
 		else:
 			print('first word not found')
 			numerator = 1
 			denominator = vocab_length
 			frac = numerator / denominator
 			sent_prob += math.log(frac)
 	probability = sent_prob
 	prob_norm = probability / sent_len
 	return prob_norm

def perplexity_from_generated_sentence(sentence, word_counts, bigram_counts):
	norm_prob = compute_probability(sentence, word_counts, bigram_counts)
	return np.exp(norm_prob)



train_data = get_data('./hw2_training_sets.txt')
#build bigram- and word-count dictionaries
bigram_counts, word_counts = train(train_data)

dist = build_dist(bigram_counts, word_counts)

sentences = []
for i in range(5):
	sentences.append(generate_sentence(dist))

perplexitites = []
for s in sentences:
	perplexitites.append(perplexity_from_generated_sentence(s, word_counts, bigram_counts))

for s, p in zip(sentences, perplexitites):
	print('Generated Sentence: \n', s)
	print('Perplexity: \n', p)
	print()
