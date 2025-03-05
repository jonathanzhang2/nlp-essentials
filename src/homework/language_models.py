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
    prev, text = '', list()
    book, chapter = re.compile(pattern=r'\(\s\d{4}\s\)'), re.compile(pattern=r'[Cc][Hh][Aa][Pp][Tt][Ee][Rr]\s[IVXLCDM]+\n')
    with open(file=filepath, mode='r') as file:
        for line in file:
            if not prev and not chapter.search(string=line) and not book.search(string=line):
                text.append(INIT + ' ' + line.rstrip('\n'))
            prev = chapter.search(string=line)
    
    bigrams = defaultdict(Counter)
    [bigrams[words[i - 1]].update([words[i]]) for line in text if (words := line.split()) for i in range(1, len(words))]

    probs, min_unknown_prob = dict(), 1
    for prev, ccs in bigrams.items():
        unknown_prob = 1/(total := sum(ccs.values()) + len(set(ccs.keys())) + 1)
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


def sequence_generator_plus(bigram_probs: Bigram, init_word: str, length: int = 20) -> tuple[list[str], float]:
    """
        Generate a sequence of specified length starting with given word, which generates sequences with
        higher probability scores and better semantic coherence compared to sequence_generator().

        :param bigram_probs: Bigram probabilities from bigram_model().
        :param init_word: First word in sequence.
        :param length: Number of words to generate.
        :return: Tuple containing:
                 - list[str]: Generated sequence
                 - float: Log probability of sequence using natural log
        """
    # dp = [defaultdict(lambda: -math.inf) for _ in range(length)]
    dp = [dict() for _ in range(length)]
    # dp = defaultdict(lambda: defaultdict(lambda: -math.inf))
    dp[0][init_word] = math.log(bigram_probs[INIT].get(init_word, bigram_probs[INIT][UNKNOWN]))
    
    backtrack = [dict() for _ in range(length)]
    
    all_vocabulary = set(bigram_probs.keys())
    [all_vocabulary.update(ccs.keys()) for _, ccs in bigram_probs.items() if not isinstance(ccs, float)]
    
    for i in range(1, length):
        all_prev = dp[i-1].keys()
        vocabulary = set()
        for w in all_prev:
            if w in bigram_probs and not isinstance(bigram_probs[w], float):
                vocabulary.update(bigram_probs[w].keys())
        # vocabulary = set(bigram_probs[dp][i-1])
        for j, candidate in enumerate(vocabulary):
            max_log_prob, best_prev_word = float('-inf'), None
            for prev_word in dp[i-1].keys():
                if prev_word not in bigram_probs or prev_word == UNKNOWN:
                    log_prob = dp[i-1][prev_word] + bigram_probs[UNKNOWN]
                elif candidate not in bigram_probs[prev_word]:
                    continue
                else:
                    log_prob = dp[i-1][prev_word] + bigram_probs[prev_word][candidate]
                if log_prob > max_log_prob:
                    max_log_prob, best_prev_word = log_prob, prev_word
            
            if best_prev_word:
                dp[i][candidate] = max_log_prob
                backtrack[i][candidate] = best_prev_word
            # else: 
            #     # backtrack to return sequence
            #     print('error')
            #     pass
    
    last_word = max(dp[-1], key=dp[-1].get)
    max_log_prob = dp[-1].get(last_word)
    
    sequence = [last_word]
    for i in range(length - 1, 1, -1):
        last_word = backtrack[i].get(last_word)
        sequence.append(last_word)
    sequence.append(init_word)
    
    return list(reversed(sequence)), max_log_prob


def main():
    filepath = '/Users/jonathan/Desktop/Emory/CS 329/nlp-essentials/dat/chronicles_of_narnia.txt'
    import json
    with open('bigram probabilities.json' ,'w') as file:
        json.dump((probs := bigram_model(filepath=filepath)), fp=file, indent=4)
        
    vocabulary = set(probs.keys())
    vocabulary -= {INIT, UNKNOWN}
    
    seqs = dict()
    for v in vocabulary:
        out = list(sequence_generator(bigram_probs=probs, init_word=v))
        seqs[v] = out
    with open('sequence generation.json', 'w') as file:
        json.dump(seqs, file, indent=4)

if __name__ == '__main__':
    main()
    filepath = '/Users/jonathan/Desktop/Emory/CS 329/nlp-essentials/dat/chronicles_of_narnia.txt'
    bigram_probs = bigram_model(filepath)
    print(bigram_probs)
    # sequence_generator(bigram_probs, 'You')