import numpy as np
from sklearn.tree import DecisionTreeClassifier

def get_bigrams(word, max_bigrams=5):
    bigrams = [word[i:i+2] for i in range(len(word) - 1)]
    bigrams = sorted(set(bigrams))[:max_bigrams]
    return bigrams

class CustomDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='log2', lookahead_depth=2):
        super().__init__(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf, max_features=max_features)
        self.lookahead_depth = lookahead_depth

    def fit(self, X, y):
        super().fit(X, y)
        self._apply_lookahead_strategy()

    def _apply_lookahead_strategy(self):
        # Implement unique lookahead logic if needed
        pass

################################
# Non Editable Region Starting #
################################
def my_fit(words):
################################
#  Non Editable Region Ending  #
################################

    # Convert words to bigrams
    bigram_list = [get_bigrams(word) for word in words]

    # Flatten the list of bigrams and create a dictionary
    total_bigrams = sorted(set(bigram for bigrams in bigram_list for bigram in bigrams))

    # Create feature vectors
    x_train = [[bigram in bigrams for bigram in total_bigrams] for bigrams in bigram_list]

    # Encode labels
    y_train = list(range(len(words)))

    # Create a custom model with unique parameters
    model = CustomDecisionTreeClassifier(
        criterion='gini',      # Use Gini index for impurity calculation
        max_depth=None,        # Limit the depth of the tree (can be tuned)
        min_samples_split=2,   # Minimum number of samples required to split an internal node
        min_samples_leaf=1,    # Minimum number of samples required to be at a leaf node
        max_features='sqrt',   # Use the square root of the number of features (bigrams) for best split
        lookahead_depth=20     # Unique lookahead depth parameter
    )

    # Train the model
    model.fit(x_train, y_train)

    # Store the bigrams and words in the model for prediction
    model.total_bigrams = total_bigrams
    model.words = words

    return model  # Return the trained model

################################
# Non Editable Region Starting #
################################
def my_predict(model, bigram_list):
################################
#  Non Editable Region Ending  #
################################

    # Create feature vector for the input bigrams
    x_test = [[bigram in bigram_list for bigram in model.total_bigrams]]

    # Predict the label
    predictions = model.predict(x_test)

    # Convert the predicted indices back to the words
    guess_list = [model.words[pred] for pred in predictions]

    return guess_list[:5]  # Return up to 5 guesses as a list
