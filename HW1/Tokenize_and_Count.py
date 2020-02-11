'''
Andrew Gerlach
1/27/2020
HW 1: Count tokens, types, and most frequen words
'''
import string, sys
def getData(filename):
	try:
		with open(filename, 'r') as f:
			data = f.read()
			#Remove punctuation
			trans = str.maketrans('','',string.punctuation)
			data_no_punc = data.translate(trans)
			return data_no_punc
	except IOError:
		print("Could not read file: ",filename)
		sys.exit()


text = getData("hw1_training_sets.txt")
#Split data on white space to get tokens.
#The length of this array is the number of tokens.
text_array = text.split()
token_count = len(text_array)

text_dict = {}
#Build a dictionary of types and their counts
#The length of the dictionary is the number of types.
for w in text_array:
	key = w.lower()
	if key in text_dict:
		text_dict[key] = text_dict[key] + 1
	else:
		text_dict[key] = 1

type_count = len(text_dict)

#Sort the dictionary descending.
sorted_text_dict = sorted(text_dict.items(), key = lambda x : x[1], reverse = True)

print("Tokens:", token_count)
print("Types:", type_count)
print()
#Print top 5 values in dictioary
for i in range(5):
	print(str(sorted_text_dict[i][0])+ " " + str(sorted_text_dict[i][1]))
