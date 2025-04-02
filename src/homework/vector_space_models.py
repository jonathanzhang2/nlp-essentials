stop_words = {
    'next', 'everything', 'should', 'theirs', 'thick', 're', 'doing', 'between', 'him', 'namely', 'myself', 'been', 'without', 'under', 
    'cant', 'themselves', 'when', 'could', 'whatever', 'serious', 'ten', 'wherever', 'wherein', 'an', 'thin', 'someone', 'y', 'mustn', 
    'nothing', 'ever', 'moreover', 'other', 'herself', 'none', 'call', 'was', 'fify', 'above', 'but', 'therefore', 'yourselves', 'with', 
    'besides', 'through', 'too', 'hereby', 'would', 'only', 'being', 'weren', 'by', 'seeming', 'eight', 'nor', 'he', 'rather', 'whoever', 
    'latter', 'be', 'please', 'done', 'becoming', 'get', 'ie', 'you', 'hadn', 'twenty', 'out', 'system', 'during', 'keep', 'alone', 'found',
    'noone', 'here', 'anyway', 'became', 'seemed', 'whenever', 'five', 'well', 'thereafter', 'each', 'even', 'third', 'neither', 'bottom', 
    'toward', 'top', 'very', 'mightn', 'for', 'll', 'inc', 'cannot', 'forty', 'if', 'at', 'will', 'ours', 'what', 'we', 't', 'whence', 'name', 
    'thus', 'perhaps', 'amongst', 'whereas', 'do', 'others', 'ourselves', 'always', 'anything', 'one', 'around', 'few', 'itself', 'though', 
    'thru', 'whither', 'who', 'couldn', 'our', 'haven', 'a', 'two', 'on', 'them', 'which', 'off', 'most', 'mill', 'bill', 'doesn', 'yours', 
    'move', 'although', 'am', 'full', 'have', 'another', 'than', 'last', 'co', 'back', 'of', 'were', 'formerly', 'o', 'still', 'just', 'never', 
    'no', 've', 'this', 'front', 'ain', 'many', 'from', 'afterwards', 'often', 'over', 'whether', 'once', 'everyone', 'why', 'so', 'both', 'else', 
    'meanwhile', 'something', 'my', 'somewhere', 'they', 'won', 'ltd', 'thereupon', 'find', 'three', 'fire', 'your', 'becomes', 'eg', 'after', 
    'it', 'hundred', 'put', 'fifteen', 'hence', 'd', 'twelve', 'first', 'are', 'against', 'yourself', 'again', 'show', 'had', 'detail', 'enough', 
    'however', 'is', 'nowhere', 'her', 'shan', 'among', 'below', 'some', 'don', 'such', 'anyone', 'there', 'that', 'cry', 'as', 'wasn', 'may', 
    'does', 'further', 'whereafter', 'due', 'every', 'before', 'six', 'didn', 'us', 'himself', 'seem', 'across', 'less', 'herein', 'behind',
    'yet', 'up', 'their', 'anyhow', 'mine', 'sometimes', 'elsewhere', 'indeed', 'i', 'ma', 'throughout', 'whose', 'amount', 'sincere', 'and',
    'either', 'the', 'side', 'down', 'whereby', 'did', 'those', 'per', 'can', 's', 'much', 'sixty', 'to', 'hereupon', 'via', 'because', 'take',
    'wouldn', 'former', 'whole', 'together', 'nevertheless', 'made', 'go', 'thereby', 'describe', 'etc', 'onto', 'same', 'sometime', 'thence',
    'several', 'upon', 'beyond', 'hasnt', 'latterly', 'otherwise', 'four', 'beforehand', 'or', 'these', 'whom', 'amoungst', 'become', 'seems',
    'eleven', 'while', 'm', 'not', 'anywhere', 'nine', 'into', 'all', 'therein', 'part', 'almost', 'except', 'give', 'having', 'empty', 
    'hereafter', 'where', 'until', 'un', 'needn', 'nobody', 'hers', 'his', 'least', 'shouldn', 'in', 'con', 'about', 'along', 'towards', 
    'somehow', 'own', 'how', 'hasn', 'see', 'its', 'any', 'already', 'interest', 'whereupon', 'must', 'also', 'then', 'might', 'couldnt',
    'more', 'within', 'beside', 'she', 'aren', 'everywhere', 'me', 'since', 'isn', 'mostly', 'has', 'de', 'now', 'fill'
}



from typing import TypeAlias, Callable
Vocabulary: TypeAlias = dict[str, int]
Document: TypeAlias = list[str]
SparseVector: TypeAlias = dict[int, int | float]

import math, random
from collections import Counter
from functools import cached_property
from operator import itemgetter



class TFIDFVectorizor:
    def __init__(self, stop_words: set=None, punctuations: bool=False, stop_word_downweight: float=0):
        self.stop_words = stop_words if stop_words else set()
        from string import punctuation
        self.punctuations = punctuation if punctuations else ''
        self.stop_word_downweight = stop_word_downweight
        self.documents = None
        self.vocabulary: Vocabulary = dict()
        self._tfs, self._dfs, self._tfidfs = list(), list(), list()
        
    @cached_property
    def D(self) -> int:
        return len(self.documents)
    
    def _build_vocabulary(self, documents: list[Document]) -> Vocabulary:
        self.documents = documents
        vocabs = set(vocab for document in self.documents for vocab in document)
        self.vocabulary = {vocab: i for i, vocab in enumerate(sorted(list(vocabs)))}
    
    def _is_stop_word(self, vocab: str) -> bool:
        return vocab.lower() in self.stop_words or vocab in self.punctuations
    
    def _frequencies(self, counts: Counter, handle_stop_words: bool=False) -> SparseVector:
        if handle_stop_words and self.stop_word_downweight == 0:
            return {
                self.vocabulary[vocab]: count for vocab, count in sorted(counts.items()) 
                if vocab in self.vocabulary and not self._is_stop_word(vocab=vocab)
            }
        weighting = lambda value: value*self.stop_word_downweight if handle_stop_words else value
        return {
            self.vocabulary[vocab]: weighting(value=count) if self._is_stop_word(vocab=vocab) else count
            for vocab, count in sorted(counts.items()) if vocab in self.vocabulary
        }
    
    def _bow(self, document: Document, l1_normalize: bool=False) -> SparseVector:
        counts = Counter(document)
        frequencies = self._frequencies(counts=counts, handle_stop_words=True)
        if not l1_normalize:
            return frequencies
        total = sum(frequencies.values())
        return {k: v/total for k, v in frequencies.items()} if total > 0 else {k: 0 for k, _ in frequencies.items()}
        
    def _tf(self, document: Document):
        return self._bow(document=document, l1_normalize=True)
    
    def _compute_dfs(self) -> SparseVector:
        counts = Counter()
        for document in self.documents:
            counts.update(set(document))
        return self._frequencies(counts=counts, handle_stop_words=False)
    
    def _tfidf(self, document_tf: SparseVector) -> SparseVector:
        return {vocab: tf*math.log(self.D/self._dfs[vocab]) for vocab, tf in document_tf.items()}
        
    
    def fit(self, documents: list[Document]) -> 'TFIDFVectorizor':
        self._build_vocabulary(documents=documents)
        self._dfs = self._compute_dfs()
        self._tfs = [self._tf(document=document) for document in documents]
        return self
    
    def fit_transform(self, documents: list[Document]) -> list[SparseVector]:
        self.fit(documents=documents)
        self._tfidfs = [self._tfidf(document_tf=tf) for tf in self._tfs]
        return self._tfidfs
    
    def transform(self, documents: list[Document]) -> list[SparseVector]:
        tfs = [self._tf(document=document) for document in documents]
        return [self._tfidf(document_tf=tf) for tf in tfs]



class KNN:
    @staticmethod
    def cosine_similarity(u: SparseVector, v: SparseVector) -> float:
        dot_product = sum(x*v.get(k, 0) for k, x in u.items())
        norms = math.sqrt(sum(x**2 for _, x in u.items())*sum(x**2 for _, x in v.items()))
        return dot_product/norms if norms > 0 else 0.0
    
    def __init__(self, n_neighbors: int=3, metric: Callable[[SparseVector, SparseVector], float]=cosine_similarity):
        self.n_neighbors = n_neighbors
        self.metric = metric
        
    def fit(self, X: list[SparseVector], y: list[int]) -> 'KNN':
        self.X = X
        self.y = y
    
    def predict(self, X: list[SparseVector]) -> list[tuple[int, float]]:
        out = [None]*len(X)
        for i, document in enumerate(X):
            similarity = [self.metric(document, x) for x in self.X]
            top_k = sorted(range(len(similarity)), key=lambda j: (-similarity[j], j))[:self.n_neighbors]
            selected = itemgetter(*top_k)(self.y)
            label = sorted(Counter(selected if isinstance(selected, tuple) else (selected,)).items(), key=lambda x: (-x[1], x[0]))[0][0]
            out[i] = (label, sum(matches := [similarity[k] for k in top_k if self.y[k] == label])/len(matches))
        return out



def read(filename: str) -> list[tuple[int, str]]:
        def aux(s: str) -> tuple[int, str]:
            t = s.split('\t')
            return int(t[0]), t[1]

        return [aux(line) for line in open(filename)]


def preprocess(data: list[tuple[int, str]]) -> tuple[list[Document], list[int]]:
    return [line[1].rstrip('\n').split() for line in data], [line[0] for line in data]


def sentiment_analyzer(trn_dat: list[tuple[int, str]], tst_dat: list[tuple[int, str]]) -> list[tuple[int, float]]:
    train_documents, ytrain = preprocess(data=trn_dat)
    test_documents, _ = preprocess(data=tst_dat)
    
    tfidf = TFIDFVectorizor(stop_words=stop_words, punctuations=True, stop_word_downweight=0.05)
    xtrain = tfidf.fit_transform(documents=train_documents)
    xtest = tfidf.transform(documents=test_documents)
    
    model = KNN(n_neighbors=15)
    model.fit(X=xtrain, y=ytrain)
    return model.predict(X=xtest)


# region MLP Implementation
Vector: TypeAlias = list[int | float]
Bias: TypeAlias = list[float]
Weights: TypeAlias = list[Vector]
Activations: TypeAlias = list[Vector]



class Function:
    def __call__(self, *args, **kwds):
        raise NotImplementedError()
    
    def derivative(self, *args, **kwds):
        raise NotImplementedError()

class ReLU(Function):
    def __call__(self, x: Activations) -> Activations:
        return [[max(0.0, weight) for weight in vector] for vector in x]

    def derivative(self, x: Activations, gradient_flow: Activations) -> Activations:
        return [
            [gi if vi > 0 else 0.0 for vi, gi in zip(vector, gradient_vector)]
            for vector, gradient_vector in zip(x, gradient_flow)
        ]
        
class Softmax(Function):
    def __call__(self, x: Activations) -> Activations:
        return [self._softmax_vector(vector) for vector in x]

    def _softmax_vector(self, u: Vector) -> Vector:
        max_ui = max(u)
        exps = [math.exp(ui-max_ui) for ui in u]
        sum_exps = sum(exps)
        return [e/sum_exps for e in exps]

    def derivative(self, x: Activations, gradient_flow: Activations) -> Activations:
        gradient = [None]*len(x)
        for k, (prob_vector, gradient_vector) in enumerate(zip(x, gradient_flow)):
            dot = sum(pi*gi for pi, gi in zip(prob_vector, gradient_vector))
            gradient[k] = [pi*(gi-dot) for pi, gi in zip(prob_vector, gradient_vector)]
        return gradient
    
class CrossEntropyLoss(Function):
    def __call__(self, y_pred: Activations, y_true: list[Vector]) -> float:
        batch_size, num_classes = len(y_pred), len(y_pred[0])
        loss = [-math.log(y_pred[i][j]+1e-9) for i in range(batch_size) for j in range(num_classes) if y_true[i][j] == 1.0]
        return sum(loss)/batch_size

    def derivative(self, y_pred: Activations, y_true: list[Vector]) -> Activations:
        return [
            [-yi_true/(yi_pred+1e-9) for yi_pred, yi_true in zip(y_pred[i], y_true[i])]
            for i in range(len(y_pred))
        ]
        
        

class Linear:
    @staticmethod
    def _dot(u: SparseVector | Vector, v: Vector) -> float:
        return (
            sum(x*v[k] for k, x in u.items()) if isinstance(u, dict)
            else sum(ui*vi for ui, vi in zip(u, v))
        )
    
    @staticmethod
    def _xavier_initialization(in_dim: int, out_dim: int, zero_bias_init: bool=True) -> tuple[Weights, Bias]:
        limit = math.sqrt(6/(in_dim + out_dim))
        weights: Weights = [
            [random.uniform(-limit, limit) for _ in range(in_dim)]
            for _ in range(out_dim)
        ]
        bias: Bias = [0.0 if zero_bias_init else random.uniform(-limit, limit) for _ in range(out_dim)]
        return weights, bias
    
    def __init__(self, in_features: int, out_features: int, activation_fn: Function, lr: float, zero_bias_init: bool=True):
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr
        self.weights, self.bias = self._xavier_initialization(in_dim=self.in_features, out_dim=self.out_features, zero_bias_init=zero_bias_init)
        self.activation_fn = activation_fn
    
    def forward(self, x: list[SparseVector | Vector]) -> Activations:
        self.input = x
        self.logits = (out := [
            [self._dot(x[i], self.weights[j]) + self.bias[j] for j in range(self.out_features)]
            for i in range(len(x))
        ])
        return self.activation_fn(out)
    
    def backward(self, gradient_flow: Activations) -> list[Vector]:
        access_xi = lambda xi, k: xi[k]
        if isinstance(self.input[0], dict):
            access_xi = lambda xi, k: xi.get(k, 0.0)
        
        batch_size = len(gradient_flow)
        weights_gradient = [[0.0 for _ in range(self.in_features)] for _ in range(self.out_features)]
        bias_gradient = [0.0 for _ in range(self.out_features)]
        
        gradient_input = [[0.0 for _ in range(self.in_features)] for _ in range(batch_size)]
        
        for i in range(batch_size):
            # backward gradient flow
            xi, gi_flow = self.input[i], gradient_flow[i]
            for j in range(self.out_features):
                for k in range(self.in_features):
                    gradient_input[i][k] += self.weights[j][k]*gi_flow[j]
                    weights_gradient[j][k] += gi_flow[j]*access_xi(xi=xi, k=k)
                bias_gradient[j] += gi_flow[j]
            
        # gradient update
        for j in range(self.out_features):
            for k in range(self.in_features):
                self.weights[j][k] -= self.lr*weights_gradient[j][k]/batch_size
            self.bias[j] -= self.lr*bias_gradient[j]/batch_size

        return gradient_input



class MLP:
    def __init__(self, in_features: int, out_features: int, hidden_dims: list[int]=None, lr: float=0.1):
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr
        dims = [self.in_features] + (hidden_dims if hidden_dims else []) + [out_features]
        self.layers = [
            Linear(in_features=dim, out_features=dims[i+1], activation_fn=ReLU(), lr=self.lr)
            for i, dim in enumerate(dims[:-2])
        ]
        self.layers.append(Linear(in_features=dims[-2], out_features=dims[-1], activation_fn=Softmax(), lr=self.lr))
        self.loss_fn = CrossEntropyLoss()
        
    @cached_property
    def num_parameters(self) -> int:
        return sum(layer.out_features*layer.in_features+layer.out_features for layer in self.layers)
    
    def forward(self, x: list[SparseVector | Vector]) -> Activations:
        self.inputs, out = [x], x
        for layer in self.layers:
            self.inputs.append(out := layer.forward(out))
        return out
    
    def backward(self, y_pred: Activations, y_true: list[Vector]):
        gradient = self.loss_fn.derivative(y_pred=y_pred, y_true=y_true)
        for i in reversed(range(len(self.layers))):
            gradient = self.layers[i].activation_fn.derivative(x=self.inputs[i+1], gradient_flow=gradient)
            gradient = self.layers[i].backward(gradient_flow=gradient)



def onehot(labels: list[int], num_classes: int) -> list[Vector]:
    return [[1.0 if j == labels[i] else 0.0 for j in range(num_classes)] for i in range(len(labels))]
    
    
def train(
    model: MLP, xtrain: list[Vector | SparseVector], ytrain: list[Vector], 
    epochs: int=20, batch_size: int=32, verbose: bool=True
) -> MLP:
    batches = ((samples := len(xtrain))+batch_size-1)//batch_size
    log = {
        'Average Training CE Loss': [None]*epochs,
        'Average Training Accuracy': [None]*epochs
    }
    
    if verbose: print(f'Training MLP, number of parameters: {model.num_parameters:,}')
    for epoch in range(epochs):
        random.shuffle((combined := list(zip(xtrain, ytrain))))
        xtrain[:], ytrain[:] = zip(*combined)
        
        loss, correct = 0.0, 0
        for batch_idx in range(batches):
            xbatch = xtrain[(l := batch_idx*batch_size):(u := min(l+batch_size, samples))]
            if not xbatch: continue
            ybatch = ytrain[l:u]
            batch_probabilities = model.forward(x=xbatch)
            loss += model.loss_fn(y_pred=batch_probabilities, y_true=ybatch)*len(xbatch)
            model.backward(y_pred=batch_probabilities, y_true=ybatch)
            
            predicted = [max(enumerate(vector), key=lambda x: x[1])[0] for vector in batch_probabilities]
            true = [vector.index(1.0) for vector in ybatch]
            correct += sum(u == v for u, v in zip(predicted, true))
            
        log['Average Training CE Loss'][epoch] = (av_loss := loss/samples)
        log['Average Training Accuracy'][epoch] = (av_acc := correct/samples*100)
        
        if verbose: print(f'Epoch {epoch+1:>3}/{epochs:<3} | Loss: {av_loss:.7f} | Accuracy: {av_acc:.4f}%')

    return model


def predict(model: MLP, x: list[Vector | SparseVector]) -> list[tuple[int, float]]:
    return [
        [(i := max(enumerate(probability_vector), key=lambda x: x[1]))[0], i[1]] 
        for probability_vector in model.forward(x)
    ]
# endregion



# region Naive Bayes Implementation
class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = set()
        self.class_count = {}
        self.feature_count = {}
        self.total_feature_count = {}
        self.prior = {}
        self.likelihood = {}
        self.vocabulary = set()
        
    def fit(self, X, y):
        n = len(y)
        self.classes = set(y)
        self.class_count = Counter(y)
        for xi, yi in zip(X, y):
            if yi not in self.feature_count:
                self.feature_count[yi] = {}
                self.total_feature_count[yi] = 0
            for feature, value in xi.items():
                self.vocabulary.add(feature)
                self.feature_count[yi][feature] = self.feature_count[yi].get(feature, 0) + value
                self.total_feature_count[yi] += value
        
        vocab_size = len(self.vocabulary)
        for yi in self.classes:
            self.prior[yi] = self.class_count[yi]/n
            
        for yi in self.classes:
            self.likelihood[yi] = {}
            total = self.total_feature_count[yi]
            for feature in self.vocabulary:
                count = self.feature_count[yi].get(feature, 0)
                self.likelihood[yi][feature] = (count + self.alpha) / (total + self.alpha * vocab_size)

    def _predict_instance(self, x):
        vocab_size = len(self.vocabulary)
        log_probs = {}

        for yi in self.classes:
            total = self.total_feature_count[yi]
            log_prob = math.log(self.prior[yi])
            for feature, value in x.items():
                if feature in self.vocabulary:
                    prob = self.likelihood[yi].get(feature, self.alpha / (total + self.alpha * vocab_size))
                else:
                    prob = self.alpha / (total + self.alpha * vocab_size)
                log_prob += value * math.log(prob)
            log_probs[yi] = log_prob

        max_log = max(log_probs.values())
        exp_probs = {label: math.exp(lp - max_log) for label, lp in log_probs.items()}
        total_exp = sum(exp_probs.values())
        probs = {label: exp_val / total_exp for label, exp_val in exp_probs.items()}

        best_label = max(probs, key=probs.get)
        best_prob = probs[best_label]
        return best_label, best_prob

    def predict(self, X):
        return [self._predict_instance(x) for x in X]
# endregion


def sentiment_analyzer_extra(trn_dat: list[tuple[int, str]], tst_dat: list[tuple[int, str]]) -> list[tuple[int, float]]:
    train_documents, ytrain = preprocess(data=trn_dat)
    test_documents, _ = preprocess(data=tst_dat)
    
    tfidf = TFIDFVectorizor(stop_words=stop_words, punctuations=True, stop_word_downweight=0.05)
    xtrain = tfidf.fit_transform(documents=train_documents)
    xtest = tfidf.transform(documents=test_documents)
    
    nb = NaiveBayes(alpha=1.0)
    nb.fit(X=xtrain, y=ytrain)
    return nb.predict(X=xtest)
    
    ytrain = onehot(labels=ytrain, num_classes=len(set(ytrain)))
    model = MLP(in_features=len(tfidf.vocabulary), out_features=len(ytrain[0]), hidden_dims=[8], lr=0.2)
    model = train(model=model, xtrain=xtrain, ytrain=ytrain, epochs=6, batch_size=64, verbose=True)
    return predict(model=model, x=xtest)


if __name__ == '__main__':
    trn_dat = read('/Users/jonathan/Desktop/Emory/CS 329/nlp-essentials/dat/sentiment_treebank/sst_trn.tsv')
    dev_dat = read('/Users/jonathan/Desktop/Emory/CS 329/nlp-essentials/dat/sentiment_treebank/sst_dev.tsv')
    preds = sentiment_analyzer(trn_dat, dev_dat)

    correct = 0
    for i, (pred, score) in enumerate(preds):
        if pred == dev_dat[i][0]:
            correct += 1

    print('Accuracy: {} ({}/{})'.format(100 * correct / len(dev_dat), correct, len(dev_dat)))