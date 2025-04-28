import math, numpy as np


def read_word_embeddings(filepath: str) -> dict[str, np.ndarray]:
    with open(file=filepath, mode='r') as file:
        embeddings = file.readlines()
    return {(line := row.rstrip().split(sep='\t'))[0]: np.array(line[1:], dtype=np.float64) for row in embeddings}


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    def norm(x: np.ndarray) -> float:
        return math.sqrt(sum(xi**2 for xi in x))
    return 0.0 if (unorm := norm(x=u)) == 0 or (vnorm := norm(x=v)) == 0 else \
        float(sum(ui*vi for ui, vi in zip(u, v))/(unorm*vnorm))


def similar_words(word_embeddings: dict[str, np.ndarray], target_word: str, threshold: float=0.8) -> list[tuple[str, float]]:
    u = word_embeddings[target_word]
    scores = [
        (word, score) for word, v in word_embeddings.items() 
        if word != target_word and (score := cosine(u=u, v=v)) >= threshold
    ]
    return sorted(scores, key=lambda twotuple: (twotuple[1], twotuple[0]), reverse=True)


def document_similarity(word_embeddings: dict[str, np.ndarray], d1: str, d2: str) -> float:
    def rowmean(u: list[np.ndarray]) -> list[float]:
        return [float(sum(u[i][j] for i in range(len(u)))/len(u)) for j in range(len(u[0]))]
    documents = [[word_embeddings[token] for token in d.split() if token in word_embeddings] for d in [d1, d2]]
    dembeddings = [rowmean(u=document) if document else [0.0]*len(next(iter(word_embeddings.values()))) for document in documents]
    return cosine(u=dembeddings[0], v=dembeddings[1])


if __name__ == '__main__':
    word_embeddings = read_word_embeddings(filepath='/Users/jonathan/Desktop/Emory/CS 329/nlp-essentials/dat/word_embeddings.txt')
    print(similar_words(word_embeddings, 'America', 0.8))
    d1 = 'I love this movie'
    d2 = 'I hate this movie'
    print(document_similarity(word_embeddings, "Japanese computer love Sunday", "Sunday morning is Pentecostal"))
    print(document_similarity(word_embeddings, d1, d2))