from typing import TypeAlias
import re, string, random, warnings, math
from collections import defaultdict, Counter


Unigram: TypeAlias = dict[str, float]
Bigram: TypeAlias = dict[str, Unigram | float]

UNKNOWN = ''
INIT = '[INIT]'


def bigram_model(filepath: str) -> Bigram:
    """
    Build a bigram language model from text file using Laplace smoothing, normalization, and initial word probabilities.

    :param filepath: Path to a text file.
    :return: Nested dictionary where:
             - Outer key is previous word (including INIT and UNKNOWN)
             - Inner key is current word (including UNKNOWN)
             - Value is P(current|previous) probability
    """
    text = list()
    with open(file=filepath, mode='r') as file:
        text = [INIT + ' ' + line.rstrip('\n') for line in file]
    
    bigrams = defaultdict(Counter)
    [bigrams[words[i - 1]].update([words[i]]) for line in text if (words := line.split()) for i in range(1, len(words))]

    probs, min_unknown_prob = dict(), 1
    for prev, ccs in bigrams.items():
        unknown_prob = 1/(total := sum(ccs.values()) + len(set(ccs.keys())))
        (smoothed_ccs := {word: (count+1)/total for word, count in ccs.items()}).update({UNKNOWN: unknown_prob})
        min_unknown_prob = min(min_unknown_prob, unknown_prob)
        probs[prev] = smoothed_ccs
    probs[UNKNOWN] = min_unknown_prob

    return probs


def sequence_generator(bigram_probs, init_word: str, length: int = 20) -> tuple[list[str], float]:
    """
    Generate a sequence of specified length starting with given word.

    :param bigram_probs: Bigram probabilities from bigram_model().
    :param init_word: First word in sequence.
    :param length: Number of words to generate.
    :return: Tuple containing:
             - list[str]: Generated sequence
             - float: Log probability of sequence using natural log
    """
    punctuation = re.compile(pattern=r'[{}]'.format(re.escape(pattern=string.punctuation)))
    punctuations = lambda seq: len(punctuation.findall(string=' '.join(seq)))
    
    vocabulary = set(bigram_probs.keys())
    [vocabulary.update(ccs.keys()) for _, ccs in bigram_probs.items() if not isinstance(ccs, float)]
    vocabulary -= {INIT, UNKNOWN}
    
    out, word_set = [init_word], {init_word}
    log_likelihood = math.log(bigram_probs[INIT][init_word] if init_word in bigram_probs[INIT] else bigram_probs[INIT][UNKNOWN])
    
    unacceptable = lambda token: punctuations(seq=out+[token]) >= length//5 or (token not in string.punctuation and token in word_set)
    
    for i in range(length-1):
        chosen, candidates = None, bigram_probs.get(out[i], bigram_probs[UNKNOWN])
        if isinstance(candidates, float):
            random_set = {word for word in vocabulary if not punctuation.search(word)} - word_set
            if len(random_set) == 0:
                warnings.warn(message=f'No more candidate words without punctuation(s) after unknown token.')
                return out, log_likelihood
            chosen, prob = random.choice(seq=list(random_set)), bigram_probs[UNKNOWN]
        else:
            for candidate, candidate_prob in sorted(candidates.items(), key=lambda item: item[1], reverse=True):
                if unacceptable(token=candidate):
                    continue
                chosen, prob = candidate, candidate_prob
                break
            if chosen is None or unacceptable(token=chosen):
                warnings.warn(message=f'No more candidate word.\n\tStart token: {init_word} | Current token: {out[i]}')
                return out, log_likelihood
        
        out, log_likelihood = out+[chosen], log_likelihood+math.log(prob)
        word_set.update({chosen} if chosen not in string.punctuation else {})

    return out, log_likelihood


if __name__ == '__main__':
    filepath = 'dat/chronicles_of_narnia.txt'
    bigram_probs = bigram_model(filepath=filepath)
    print(bigram_probs)