from typing import Callable
from vector_space_models import Document, TFIDFVectorizor, KNN, preprocess, stop_words, read
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import optuna, multiprocessing



def CVAccuracy(documents: list[Document], labels: list[int], stop_word_downweight: float, n_neighbors: int, n_splits: int) -> float:
    scores = [None]*n_splits
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

    for k, (train_idx, val_idx) in enumerate(cv.split(X=documents, y=labels)):
        train_documents, val_documents = [documents[i] for i in train_idx], [documents[i] for i in val_idx]
        ytrain, yval = [labels[i] for i in train_idx], [labels[i] for i in val_idx]
        
        tfidf = TFIDFVectorizor(stop_words=stop_words, punctuations=True, stop_word_downweight=stop_word_downweight)
        xtrain = tfidf.fit_transform(documents=train_documents)
        xval = tfidf.transform(documents=val_documents)
        
        model = KNN(n_neighbors=n_neighbors)
        model.fit(X=xtrain, y=ytrain)
        out = model.predict(X=xval)
        
        scores[k] = accuracy_score(y_true=yval, y_pred=[y for y, _ in out])
    
    return sum(scores)/len(scores)


def make_objective(documents: list[Document], labels: list[int], n_splits: int=5) -> Callable:
    def objective(trial):
        stop_word_downweight = trial.suggest_categorical(
            'stop_word_downweight', [0, 0.005, 0.01, 0.02, 0.05, 0.1]
        )
        n_neighbors = trial.suggest_int('n_neighbors', 1, 25)

        return CVAccuracy(
            documents=documents, labels=labels, 
            stop_word_downweight=stop_word_downweight, n_neighbors=n_neighbors, 
            n_splits=n_splits
        )
        
    return objective


def optuna_worker(documents: list[Document], labels: list[int], storage_url: str, n_trials: int=4):
    study = optuna.create_study(
        study_name='knn', direction='maximize', storage=storage_url, load_if_exists=True
    )
    study.optimize(func=make_objective(documents, labels), n_trials=n_trials)



if __name__ == '__main__':
    trn_dat = read('dat/sentiment_treebank/sst_trn.tsv')
    train_documents, ytrain = preprocess(trn_dat)
    
    num_workers, trials_per_worker = 8, 5
    
    storage_url = 'sqlite:///optuna-knn.db'
    optuna.create_study(study_name='knn', direction='maximize', storage=storage_url, load_if_exists=True)
    with multiprocessing.get_context('spawn').Pool(processes=num_workers) as pool:
        pool.starmap(
            func=optuna_worker, 
            iterable=[(train_documents, ytrain, storage_url, trials_per_worker)]*num_workers
        )
