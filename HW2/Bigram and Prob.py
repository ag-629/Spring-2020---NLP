import sys, math

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


def compute_probability(test_data, word_counts, bigram_counts, normalize = False):

 	'''
 	Comutes probability of a sentence given the bigrams and word counts from training data

	inputs: bigram pair counts, total word (type) counts, test data, 
	normalize flag to return a normalized probability

	returns: dictionaries of sentence index, probability pairs
 	'''
 	probabilites = []
 	sent_len = []
 	vocab_length = len(word_counts)
 	for sent in test_data:
 		sent = sent.lower().split()
 		#Create bigrams from each sentence in test data
 		sent_len.append(len(sent))
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
 		probabilites.append(sent_prob)
 	if normalize:
 		prob_norm = [(log_prob / length) for log_prob, length in zip(probabilites, sent_len)]
 		return prob_norm
 	else:
 		return probabilites
 		
def perplexity_from_prob(prob_log):
	'''
	inputs: prob_log: list of NORMALIZED log probabilites
	
	return: list of perplexities for each given probablility
	'''
	return [math.exp(-1 * p) for p in prob_log]

def perplexity_from_data(test_data, word_counts, bigram_counts):

 	'''
 	Computed perplexity test data given the bigrams and word counts from training data

	inputs: bigram pair counts, total word (type) counts, test data, 

	returns: list of perplexities per sentence
 	'''
 	#must use normalized for perplexity
 	norm_probs = compute_probability(test_data, word_counts, bigram_counts, normalize = True)
 	return perplexity_from_prob(norm_probs)



#Retreive and clean the input text
train_data = get_data('./hw2_training_sets.txt')
test_data = get_data('test_set.txt')

#build bigram- and word-count dictionaries
bigram_counts, word_counts = train(train_data)

#Log probabilites of sentences in test data
probabilities_log = compute_probability(test_data, word_counts, bigram_counts)
prob_log_norm = compute_probability(test_data, word_counts, bigram_counts, normalize = True)
perp_from_prob = perplexity_from_prob(prob_log_norm)

print('(Using log probability)')
for i in range(len(probabilities_log)):
	print('{0}, {1}, {2}'.format(probabilities_log[i], prob_log_norm[i], perp_from_prob[i]))
