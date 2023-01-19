Project topic: Twitter bot


Description of the project:

Creating a bot in Python using the Twitter API and some machine learnig.
I have set up two Twitter accounts where one represents a Twitter user (Input) and the other represents a bot answering the given question (Output).
To get a response from the bot it is necessary to tag it with the "@" sign. The main.py file, when run, generates a response to the last Tweet in which the bot account was tagged.

More specifically user questions are categorized according to the keyword detected in them, preceded by the "#" sign:
1) #hello
2) #embedding
3) #emotion
4) no keyword detected.

The operation of the program in each of the four cases will be as follows:
1) Return a welcome message.
2) Return 2D image showing words connected with the word in the question preceded by the "$" sign.
3) Return the most probable emotion recognized in the question.
4) Extend the text in the question so that the whole makes sense.

The answer to each question is generated as follows:
1) A simple welcome message.
2) There is a dataset containing word embedding data: words are assigned vectors of length 100.
The program searches for the words closest to the specified word and reduces the space from 100 dimensions to 2 dimensions (using PCA method) so that the words can be shown in a 2D image.
3) There is a dataset containing Tweets with a specific emotion attached. Tweets are processed into sequences using Tokenizer. The processed data goes to a neural network that learns to classify Tweets.
4) A function from tensorflow API allows to extend the given text. A pretreated transformer model is used for this purpose.
