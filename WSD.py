import sys
from math import log, ceil
from collections import defaultdict, Counter
import nltk
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score

class WSD(object):
    """
    The paper I referred to:
    http://www.springer.com/cda/content/document/cda_downloaddocument
    /9783642336928-c2.pdf?SGWID=0-0-45-1356776-p174672576

    https://web.stanford.edu/~jurafsky/slp3/slides/Chapter18.wsd.pdf

    # max( sum( log( sum( alpha_k ( factorial( theta_kj )) ) ) ) )
    # where alpha_k = feature_j / all_feature_occurrence
    # theta_kj = feature_j in context of sense s_k / all features in context of sense s_k

    # using add-one smooth
    # work in log space
    """
    def __init__(self, target, fold_number):
        self.folds = []
        self.target = target
        self.fold_number = fold_number

    def pre_processing(self, filename):
        """
        Stores a list of BeautifulSoup object into self.folds.
        """
        #print "pre-processing start..."
        with open(filename, 'r') as f:
            s = f.read().strip()
            content = s.split("\n\n")
            self.length = len(content)
            self.fold_length = int(ceil(self.length * 1.0 / self.fold_number))
            for i in xrange(self.length):
                content[i] = BeautifulSoup(content[i].strip(), "lxml")
            for i in xrange(self.fold_number):
                self.folds.append(content[i * self.fold_length: min(self.length, (i + 1) * self.fold_length)])

    def train(self, fold_as_test_id):
        self.prior_count = Counter()
        self.prior_probability = {}

        # collocational features, considering a +-1 window [..., w_i, pos_i, ...]
        self.collocational_tags = {}
        # bag of words: frequency counts
        self.bag_of_words = {}

        self.answers = []

        train_length = 0
        for i in xrange(self.fold_number):
            if i == fold_as_test_id:
                continue
            #print "training fold", i + 1
            soup = self.folds[i]
            for instance in soup:
                train_length += 1
                target, answer = self.get_answer(instance)
                self.prior_count[answer] += 1

                if answer not in self.bag_of_words:
                    self.answers.append(answer)
                    self.bag_of_words[answer] = Counter()
                    self.collocational_tags[answer] = defaultdict(int)

                sentence, pos = self.get_content(instance)

                self.get_bag_of_words(pos, answer)
                collocational_feature = self.get_collocation_feature(pos)
                self.collocational_tags[answer][collocational_feature] += 1

        self.prior_probability[self.answers[0]] = self.prior_count[self.answers[0]] * 1.0 / (self.prior_count[self.answers[0]] + self.prior_count[self.answers[1]])
        self.prior_probability[self.answers[1]] = 1.0 - self.prior_probability[self.answers[0]]

        self.words1 = sum(self.bag_of_words[self.answers[0]].values())
        self.words2 = sum(self.bag_of_words[self.answers[1]].values())
        self.V = len(self.bag_of_words[self.answers[0]]) + len(self.bag_of_words[self.answers[1]])

        self.collocation1 = sum(self.collocational_tags[self.answers[0]].values())
        self.collocation2 = sum(self.collocational_tags[self.answers[1]].values())
        self.colV = len(self.collocational_tags[self.answers[0]]) + len(self.collocational_tags[self.answers[1]])


    def get_collocation_feature(self, pos):
        collocation_tags = []
        for ind in xrange(len(pos)):
            word, tag = pos[ind]
            if word == self.target:
                collocation_tags.append(pos[ind - 1][1])
                if ind < len(pos) - 1:
                    collocation_tags.append(pos[ind + 1][1])
                else:
                    collocation_tags.append("")
                break
        collocation_tags = tuple(collocation_tags)
        return collocation_tags

    def get_bag_of_words(self, pos, answer):
        #wordset = set()
        for word, tag in pos:
            self.bag_of_words[answer][word] += 1
         #   if word not in wordset:
          #      wordset.add(word)
          #      self.inverse_document_freq[word] += 1

    def get_co_occurence_feature(self, sentence):
        word_bag = sentence.split(" ")
        feature_vec = []
        for word in word_bag:
            feature_vec.append((self.bag_of_words[self.answers[0]][word], self.bag_of_words[self.answers[1]][word]))

        feature_vec = tuple(feature_vec)
        return feature_vec

    def get_content(self, instance):
        content = instance.context
        sentence = " ".join(content.get_text().strip().split()).lower()
        pos = nltk.pos_tag(nltk.word_tokenize(sentence))
        return sentence, pos

    def get_answer(self, instance):
        target, answer = instance.answer['senseid'].split("%")
        return target, answer

    def test(self, fold_as_test_id, f):
        #print "start testing..."
        test_fold = self.folds[fold_as_test_id]
        my_result = []
        prob1 = log(self.prior_probability[self.answers[0]])
        prob2 = log(self.prior_probability[self.answers[1]])
        #print prob1, prob2
        golden_result = []
        head = "Fold %d\n" % (fold_as_test_id + 1)
        f.write(head)
        for instance in test_fold:
            answer1_prob = prob1
            answer2_prob = prob2
            target, answer = self.get_answer(instance)
            golden_result.append(answer)
            id = instance.answer['instance']
            sentence, pos = self.get_content(instance)

            co_occurence_feature = self.get_co_occurence_feature(sentence)

            for v1, v2 in co_occurence_feature:
                answer1_prob += log((v1 + 1.0) / (self.words1 + self.V))
                answer2_prob += log((v2 + 1.0) / (self.words2 + self.V))

            collocation_tags = self.get_collocation_feature(pos)
            answer1_prob += log((self.collocational_tags[self.answers[0]][collocation_tags] + 1.0) / (self.collocation1 + self.colV))
            answer2_prob += log((self.collocational_tags[self.answers[1]][collocation_tags] + 1.0) / (self.collocation2 + self.colV))

            if answer1_prob >= answer2_prob:
                my_result.append(self.answers[0])
                f.write(self.output_line(id, self.answers[0]))
            else:
                my_result.append(self.answers[1])
                f.write(self.output_line(id, self.answers[1]))
        accuracy = self.evaluate(my_result, golden_result)
        print "The accuracy of fold", fold_as_test_id + 1, "is: ", accuracy
        return accuracy
    def output_line(self, id, result):
        str = id + " " + self.target + "%" + result + "\n"
        return str
    def evaluate(self, my_result, golden):
        return accuracy_score(golden, my_result)

if __name__ == "__main__":
    file = sys.argv[1]
    the_word = file[:-4]
    print the_word
    fold_number = 5
    solution = WSD(the_word, fold_number)
    solution.pre_processing(file)
    avg_accuracy = 0.0
    output_file_name = the_word + ".wsd.out"
    f = open(output_file_name, 'a')
    for i in xrange(5):
        solution.train(i)
        avg_accuracy += solution.test(i, f)
    f.close()
    avg_accuracy /= fold_number * 1.0
    print "the average accuracy is %.2f" % avg_accuracy