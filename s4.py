"""Implements self supervised semantic shift functions
It uses poisoning attacks to learn landmarks in a self-supervised way
At each iteration, generate perturbation on the data, generating positive
and negative samples
Learn the separation between them (using any classifier)
Apply the classifier to the original (non-perturbated) data
Negatives -> landmarks
Positives -> semantically changed
We can begin by aligning on all words, and then learn better landmarks from
there. Alternatively, one can start from random landmarks."""


# STL/Third party modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss, pairwise_distances
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Local modules
from WordVectors import WordVectors, intersection
from alignment import align


# Initialize random seeds
np.random.seed(1)
tf.random.set_seed(1)
# tf.random.set_random_seed(1)  # This is for TF1


def cumulative_choice(array, p=None):
    """
    Performs fast choice based on the cumulative distribution of elements in `array`.
    Arguments:
        array: (array-like) Elements to choose from
        p:  (array-like) Probabilities of choosing each element from `array`.
            if `None`, then every element is chosen with the same probability.
    Returns:
        x: (scalar) The chosen element(s).
    """
    assert len(array) == len(p) or p is None
    
    r = np.random.random()
    cum = 0  # cumulative sum

    for i, p_ in enumerate(p):
        cum += p_
        if r < cum:
            break
    return array[i]                


def negative_samples(words, size, p=None):
    """
    Returns negative samples of semantic change
    May use distribution of cosine distance as sampling distribution
    """
    neg_samples = np.random.choice(words, size, p=p)

    return neg_samples


def inject_change_single(wv, w, words, v_a, alpha, replace=False,
                         max_tries=50,
                         choice_method='random',
                         num_choices=1,
                         random_vector=False,
                         distances=None):
    """
    Injects change to word w in wv by randomly selecting a word t in wv
    and injecting the sense of t in to w.
    The modified vector of w must have a higher cosine distance to v_a
    than its original version. This is done by sampling t such that the cosine similarity of
    (w, t) is not greater than that of v_a and wv(w) or until a max_tries, 
    where v_a is the vector of word w in the parallel corpus (not wv).

    Arguments:
            wv      -   WordVectors of the corpus to be modified
            w       -   (str) Word to be modified
            words   -   (list) Pool of words to sample from, injecting sense
            v_a     -   (np.ndarray) word vector of w in the parallel source to wv
            alpha   -   (float) Rate of injected change
            replace -   (bool) Whether to replace w with t instead of 'moving' w towards t
            max_tries - (int) Maximum number of attempts made in order to achieve the minimum perturbation threshold
            choice_method - (str) How to choose the destination words one of {'random', 'close', 'far'}
                            - 'random' uniformly chooses a random word
                            - 'close' chooses a word based on the cosine similarity distribution
                            - 'far' chooses a word based on the cosine distance distribution
            num_choices - (int) Number of words to select and perturb towards. Will perturb towards the mean vector of the selected words.
            random_vector - (bool) If True, will perturb towards a `random` vector instead of sampling a word.
            distances - (array-like) List of distances between the target vector wv[w] and every other word in wv (used for close and far)
    Returns:
            x       -   (np.ndarray) modified vector of w
    """
    cos_t = cosine(v_a, wv[w])  # cosine distance threshold we want to surpass

    c = 0
    tries = 0
    w_id = wv.word_id[w]
    v_b = np.copy(wv.vectors[w_id])  # vb stores the modified vector

    if choice_method != 'random' and distances is None:
        distances = np.fromiter((cosine(v_b, v) for v in wv.vectors), dtype=float)

    if not random_vector:
        while c < cos_t and tries < max_tries:
            tries += 1

            if choice_method == 'random':
                selected = np.random.choice(words, size=num_choices)  # select word with new sense
            elif choice_method == 'far':
                p_cos = distances/(distances.sum())  # larger distances are sampled more
                selected = np.random.choice(words, size=num_choices, p=p_cos)
                # selected = cumulative_choice(words, p=p_cos)
            elif choice_method == 'close':
                cos_sim = 1-distances  # Convert cosine distances to similarities
                p_sim = (cos_sim + 1)/((cos_sim + 1).sum())  # Shift similarities co-domain from [-1, 1] to [0,2]
                selected = np.random.choice(words, size=num_choices, p=p_sim)
            else:
                print("S4 Error: invalid choice_method")
                return v_b
            v_target = np.mean([wv[s] for s in selected], axis=0)
            
            if not replace:
                b = wv[w] + alpha*v_target
                v_b = b
            else:
                v_b = v_target

            c = cosine(v_a, v_b)
    else:
        v_target = np.random.normal(size=v_b.shape)
        if not replace:
            b = wv[w] + alpha*v_target
            v_b = b
        else:
            v_b = v_target

    return v_b


def inject_change_batch(wv, changes, alpha, replace=True):
    """
    Given a WordVectors object and a list of words, perform fast injection
    of semantic change by using the update rule from Word2Vec
    wv - WordVectors (input)
    changes - list of n tuples (a, b) that drives the change such that b->a
          i.e.: simulates using b in the contexts of a
    alpha - degree in which to inject the change
              if scalar: apply same alpha to every pair
              if array-like: requires size n, specifies individual alpha values
                              for each pair
    replace  - (bool) if True, words are replaced instead of moved
                e.g.: if pair is (dog, car), then v_car <- v_dog
    Returns a WordVectors object with the change
    """
    wv_new = WordVectors(words=wv.words, vectors=np.copy(wv.vectors))
    for i, pair in enumerate(changes):
        t, w = pair
        t_i = wv.word_id[t]  # target word
        w_i = wv.word_id[w]  # modified word
        # Update vector with alpha and score
        # Higher score means vectors are already close, thus apply less change
        # Alpha controls the rate of change
        if not replace:
            b = wv_new[w] + alpha*(1)*wv[t]
            wv_new.vectors[w_i] = b
        else:
            wv_new.vectors[w_i] = wv[t]
        # print("norm b", np.linalg.norm(b))
    return wv_new


def get_features(x, names=("cos",)):
    """
    Compute features given input training data (concatenated vectors)
    Default features is cosine. Accepted features: cosine (cos).
    Attributes:
            x   - size n input training data as concatenated word vectors
            names - size d list of features to compute
    Returns:
            n x d feature matrix (floats)
    """
    x_out = np.zeros((len(x), len(names)), dtype=float)
    for i, p in enumerate(x):
        for j, feat in enumerate(names):
            if feat == "cos":
                x_ = cosine(p[:len(p)//2], p[len(p)//2:])
                x_out[i][j] = x_
    return x_out


def build_sklearn_model():
    """
    Build SVM using sklearn model
    The model uses an RBF kernel and the features are given by difference
    between input vectors u-v.
    Return: sklearn SVC
    """
    model = SVC(random_state=0, probability=True)
    return model


def build_keras_model(dim):
    """
    Builds the keras model to be used in self-supervision.
    Return: Keras-Tensorflow2 model
    """
    h1_dim = 100
    h2_dim = 100
    model = keras.Sequential([
                             keras.layers.Input(shape=(dim)),
                             keras.layers.Dense(h1_dim, activation="relu",
                                                activity_regularizer=keras.regularizers.l2(1e-2)),
                             # keras.layers.Dense(h2_dim, activation="relu",
                             #                    activity_regularizer=keras.regularizers.l2(1e-2)),
                             keras.layers.Dense(1, activation="sigmoid")
                            ])
    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def threshold_crossvalidation(wv1, wv2, iters=100,
                                        n_fold=1,
                                        n_targets=100,
                                        n_negatives=100,
                                        fast=True,
                                        rate=0.5,
                                        t=0.5,
                                        landmarks=None,
                                        t_overlap=1,
                                        debug=False,
                                        inject_choice='random'):
    """
    Runs crossvalidation over self-supervised samples, carrying out a model
    selection to determine the best cosine threshold to use in the final
    prediction.

    Arguments:
        wv1, wv2        - input WordVectors. They are required to be INTERSECTED and ALIGNED before call
        plot            - 1: plot functions in the end 0: do not plot
        iters           - max no. of iterations
        n_fold          - n-fold crossvalidation (1 - leave one out, 10 - 10-fold cv, etc.)
        n_targets       - number of positive samples to generate
        n_negatives     - number of negative samples
        fast            - use fast semantic change simulation
        rate            - rate of semantic change injection
        t               - classificaiton threshold (0.5)
        t_overlap       - overlap threshold for (stop criterion)
        landmarks       - list of words to use as landmarks (classification only)
        debug           - toggles debugging mode on/off. Provides reports on several metrics. Slower.
        inject_choice   - choice method for sense injection {'random', 'far', 'close', 'new'}  
    Returns:
        t - selected cosine threshold t
    """

    wv2_original = WordVectors(words=wv2.words, vectors=wv2.vectors.copy())
    landmark_set = set(landmarks)
    non_landmarks = [w for w in wv1.words if w not in landmark_set]

    d_matrix = pairwise_distances(wv2_original.vectors, metric='cosine', n_jobs=-1)

    for iter in range(iters):

        replace = dict()  # replacement dictionary
        pos_samples = list()
        pos_vectors = dict()

        # Randomly sample words to inject change to
        # If no word is flagged as non_landmarks, sample from all words
        # In practice, this should never occur when selecting landmarks
        # but only for classification when aligning on all words
        if len(non_landmarks) > 0:
            targets = np.random.choice(non_landmarks, n_targets)
            # Make targets deterministic
            #targets = non_landmarks
        else:
            targets = np.random.choice(wv1.words, n_targets)

        for target in targets:

            # Simulate semantic change in target word
            v = inject_change_single(wv2_original, target, wv1.words,
                                     wv1[target], rate, distances=d_matrix[wv2_original.word_id[target]])

            pos_vectors[target] = v

            pos_samples.append(target)
        # Convert to numpy array
        pos_samples = np.array(pos_samples)
        # Get negative samples from landmarks
        neg_samples = negative_samples(landmarks, n_negatives, p=None)
        neg_vectors = {w: wv2_original[w] for w in neg_samples}
        # Create dictionary of supervision samples (positive and negative)
        # Mapping word -> vector
        sup_vectors = {**neg_vectors, **pos_vectors}

        # Prepare training data
        words_train = np.concatenate((pos_samples, neg_samples))
        # assign labels to positive and negative samples
        y_train = [1] * len(pos_samples) + [0] * len(neg_samples)

        # Stack columns to shuffle data and labels together
        train = np.column_stack((words_train, y_train))
        # Shuffle batch
        np.random.shuffle(train)
        # Detach data and labels
        words_train = train[:, 0]
        y_train = train[:, -1].astype(int)

        # Calculate cosine distance of training samples
        x_train = np.array([cosine(wv1[w], sup_vectors[w]) for w in words_train])

        # t_pool = [0.2, 0.7]
        t_pool = np.arange(0.2, 1, 0.1)

        best_acc = 0
        best_t = 0
        for t_ in t_pool:
            acc = 0
            for i in range(0, len(x_train), n_fold):
                x_cv = x_train[i:i+n_fold]
                y_true = y_train[i:i+n_fold]
                y_hat = x_cv > t_
                acc += sum(y_hat == y_true)/len(x_cv)
            acc = acc/(len(x_train)//n_fold)
            if acc > best_acc:
                best_acc = acc
                best_t = t_
                print("- New best t", t_, acc)

    return best_t


def s4(wv1, wv2, verbose=0, plot=0, cls_model="nn",
          iters=100,
          n_targets=10,
          n_negatives=10,
          fast=True,
          rate=0,
          t=0.5,
          t_overlap=1,
          landmarks=None,
          update_landmarks=True,
          inject_choice='random',
          num_choices=1,
          random_vector=False,
          return_model=False,
          debug=False):
    """
    Performs self-supervised learning of semantic change.
    Generates negative samples by sampling from landmarks.
    Generates positive samples via simulation of semantic change on random non-landmark words.
    Trains a classifier, fine-tune it across multiple iterations.
    If update_landmarks is True, then it learns landmarks from that step. In this case,
    the returned values are landmarks, non_landmarks, Q (transform matrix)
    Otherwise, landmarks are fixed from a starting set and the returned value
    is the learned classifier - landmarks must be passed.
    Arguments:
        wv1, wv2    - input WordVectors - required to be intersected before call
        verbose     - 1: display log, 0: quiet
        plot        - 1: plot functions in the end 0: do not plot
        cls_model   - classification model to use {"nn", "svm_auto", "svm_features"}
        iters       - max no. of iterations
        n_targets   - number of positive samples to generate
        n_negatives - number of negative samples
        fast        - (deprecated) use fast semantic change simulation
        rate        - rate of semantic change injection
        t           - classificaiton threshold (0.5)
        t_overlap   - overlap threshold for (stop criterion)
        landmarks   - list of words to use as landmarks (classification only)
        update_landmarks - if True, learns landmarks. Otherwise, learns classification model.
        inject_choice   - Method of sense injection {'random', 'far', 'close', 'new'}
        'return_model'- If True, returns the classification model
        debug       - toggles debugging mode on/off. Provides reports on several metrics. Slower.
    Returns:
        if update_landmarks is True:
            landmarks - list of landmark words
            non_landmarks - list of non_landmark words
            Q           - transformation matrix for procrustes alignment
        if update_landmarks is False:
            model       - binary classifier
    """

    # Define verbose prints
    if verbose == 1:
        def verbose_print(*s, end="\n"):
            print(*s, end=end)
    elif verbose == 0:
        def verbose_print(*s, end="\n"):
            return None

    # Measure times
    t_sampling = 0
    t_train = 0
    t_update = 0
    t_predict = 0
    t_realign = 0
    t_begin = time.time()
    t0 = time.time()

    wv2_original = WordVectors(words=wv2.words, vectors=wv2.vectors.copy())

    avg_window = 0  # number of iterations to use in running average

    # Begin alignment
    if update_landmarks:
        # Check if landmarks is initialized
        if landmarks is None:
            wv1, wv2, Q = align(wv1, wv2)  # start form global alignment
            landmark_dists = [euclidean(u, v) for u, v in zip(wv1.vectors, wv2.vectors)]
            landmark_args = np.argsort(landmark_dists)
            landmarks = [wv1.words[i] for i in landmark_args[:int(len(wv1.words)*0.5)]]
            # landmarks = np.random.choice(wv1.words, int(len(wv1)*0.5))
        landmark_set = set(landmarks)
        non_landmarks = np.array([w for w in wv1.words if w not in landmark_set])
    else:
        if landmarks is None:  # If no landmarks are given, infer candidates for positive/negative sampling
            landmark_dists = [euclidean(u, v) for u, v in zip(wv1.vectors, wv2.vectors)]
            landmark_args = np.argsort(landmark_dists)
            landmarks = [wv1.words[i] for i in landmark_args[:int(len(wv1.words)*0.5)]]
        landmark_set = set(landmarks)
        non_landmarks = [w for w in wv1.words if w not in landmark_set]

    wv1, wv2, Q = align(wv1, wv2, anchor_words=landmarks)

    if cls_model == "nn":
        model = build_keras_model(wv1.dimension*2)
    elif cls_model == "svm_auto" or cls_model == "svm_features":
        model = build_sklearn_model()  # get SVC
    else:
        model = cls_model  # callable cls_model

    landmark_hist = list()  # store no. of landmark history
    loss_hist = list()  # store self-supervision loss history
    alignment_loss_hist = list()  # store landmark alignment loss
    alignment_out_hist = list()  # store alignment loss outside of lm
    alignment_all_hist = list()

    cumulative_out_hist = list()
    cumulative_alignment_hist = list()  # store cumulative loss alignment
    overlap_hist = list()  # store landmark overlap history
    cumulative_overlap_hist = list()  # mean overlap history
    cumulative_loss = 0

    # History of cosines
    cos_loss_in_hist = list()
    cos_loss_out_hist = list()
    cumulative_cos_in = list()
    cumulative_cos_out = list()

    prev_landmarks = set(landmarks)
    t_init = time.time() - t0
    
    d_matrix = pairwise_distances(wv2_original.vectors, metric='cosine', n_jobs=-1)

    for iter in range(iters):

        t0 = time.time()
        pos_samples = list()
        pos_vectors = dict()

        # Randomly sample words to inject change to
        # If no word is flagged as non_landmarks, sample from all words
        # In practice, this should not occur when selecting landmarks
        # but only for classification when aligning on all words
        if len(non_landmarks) > 0:
            targets = np.random.choice(non_landmarks, n_targets)
            # Make targets deterministic
            #targets = non_landmarks
        else:
            targets = np.random.choice(wv1.words, n_targets)

        for target in targets:

            # Simulate semantic change in target word
            v = inject_change_single(wv2_original, target, wv1.words,
                                     wv1[target], rate,
                                     distances=d_matrix[wv2_original.word_id[target]],
                                     choice_method=inject_choice,
                                     num_choices=num_choices,
                                     random_vector=random_vector)

            pos_vectors[target] = v

            pos_samples.append(target)
        # Convert to numpy array
        pos_samples = np.array(pos_samples)
        # Get negative samples from landmarks
        neg_samples = negative_samples(landmarks, n_negatives, p=None)
        neg_vectors = {w: wv2_original[w] for w in neg_samples}
        # Create dictionary of supervision samples (positive and negative)
        # Mapping word -> vector
        sup_vectors = {**neg_vectors, **pos_vectors}

        t_sampling += time.time() - t0
        t0 = time.time()

        # Prepare training data
        words_train = np.concatenate((pos_samples, neg_samples))
        # assign labels to positive and negative samples
        y_train = [1] * len(pos_samples) + [0] * len(neg_samples)

        # Stack columns to shuffle data and labels together
        train = np.column_stack((words_train, y_train))
        # Shuffle batch
        np.random.shuffle(train)
        # Detach data and labels
        words_train = train[:, 0]
        y_train = train[:, -1].astype(int)

        x_train = np.array([np.append(wv1[w], sup_vectors[w]) for w in words_train])

        # Append history
        landmark_hist.append(len(landmarks))
        v1_land = np.array([wv1[w] for w in landmarks])
        v2_land = np.array([wv2_original[w] for w in landmarks])
        v1_out = np.array([wv1[w] for w in non_landmarks])
        v2_out = np.array([wv2_original[w] for w in non_landmarks])

        alignment_loss = np.linalg.norm(v1_land-v2_land)**2/(len(v1_land) + 1e-5)
        alignment_loss_hist.append(alignment_loss)
        cumulative_alignment_hist.append(np.mean(alignment_loss_hist[-avg_window:]))

        # out loss
        alignment_out_loss = np.linalg.norm(v1_out-v2_out)**2/(len(v1_out) + 1e-5)
        alignment_out_hist.append(alignment_out_loss)
        cumulative_out_hist.append(np.mean(alignment_out_hist[-avg_window:]))

        # all loss
        alignment_all_loss = np.linalg.norm(wv1.vectors-wv2_original.vectors)**2/len(wv1.words)
        alignment_all_hist.append(alignment_all_loss)

        if debug:
            # cosine loss
            cos_in = np.mean([cosine(u, v) for u, v in zip (v1_land, v2_land)])
            cos_out = np.mean([cosine(u, v) for u, v in zip(v1_out, v2_out)])
            cos_loss_in_hist.append(cos_in)
            cos_loss_out_hist.append(cos_out)
            cumulative_cos_in.append(np.mean(cos_loss_in_hist))
            cumulative_cos_out.append(np.mean(cos_loss_out_hist))

        # Begin training of neural network
        if cls_model == "nn":
            history = model.train_on_batch(x_train, y_train, reset_metrics=False)
            # history = model.fit(x_train, y_train, epochs=5, verbose=0)
            # history = [history.history["loss"][0]]
        elif cls_model == "svm_auto":
            model.fit(x_train, y_train)
            pred_train = model.predict_proba(x_train)
            history = [log_loss(y_train, pred_train)]
        elif cls_model == "svm_features":
            x_train_ = get_features(x_train)  # retrieve manual features
            model.fit(x_train_, y_train)
            pred_train = model.predict_proba(x_train_)
            y_hat_t = (pred_train[:, 0] > 0.5)
            acc_t = accuracy_score(y_train, y_hat_t)
            history = [log_loss(y_train, pred_train), acc_t]
        else:
            training_loss = model.fit(x_train, y_train)
            proba = model.predict(x_train)
            y_hat_t = (proba[:, 0] > 0.5)
            acc_t = accuracy_score(y_train, y_hat_t)
            history = [training_loss, acc_t]

        t_train += time.time() - t0
        t0 = time.time()

        loss_hist.append(history[0])

        # Apply model on original data to select landmarks
        x_real = np.concatenate((wv1.vectors, wv2_original.vectors), axis=1)
        if cls_model == "nn":
            predict_real = model.predict(x_real)
        elif cls_model == "svm_auto":
            predict_real = model.predict_proba(x_real)
            predict_real = predict_real[:, 1]
        elif cls_model == "svm_features":
            x_real_ = get_features(x_real)
            predict_real = model.predict_proba(x_real_)
            predict_real = predict_real[:, 1]
        else:
            predict_real = model.predict(x_real)

        t_predict += time.time() - t0
        t0 = time.time()

        if update_landmarks:
            predict_real = predict_real.flatten()
            mask = predict_real < t
            landmarks = wv1.words[mask]
            non_landmarks = wv1.words[~mask]

        # Update landmark overlap using Jaccard Index
        isect_ab = set.intersection(prev_landmarks, set(landmarks))
        union_ab = set.union(prev_landmarks, set(landmarks))
        j_index = len(isect_ab)/len(union_ab)
        overlap_hist.append(j_index)

        cumulative_overlap_hist.append(np.mean(overlap_hist[-avg_window:]))  # store mean

        prev_landmarks = set(landmarks)

        verbose_print("> %3d | L %4d | l(in): %.2f | l(out): %.2f | loss: %.2f | overlap %.2f | acc: %.2f" %
                      (iter, len(landmarks), cumulative_alignment_hist[-1],
                       cumulative_out_hist[-1], history[0], cumulative_overlap_hist[-1], history[1]),
                      end="\r")
        t_update += time.time() - t0
        t0 = time.time()

        if len(landmarks) == 0:
            break
        wv1, wv2_original, Q = align(wv1, wv2_original, anchor_words=landmarks)
        t_realign += time.time() - t0

        # Check if overlap difference is below threshold
        if np.mean(overlap_hist) > t_overlap:
            break

    # Print new line
    verbose_print()

    if plot == 1:
        iter += 1  # add one to iter for plotting
        plt.plot(range(iter), landmark_hist, label="landmarks")
        plt.hlines(len(wv1.words), 0, iter, colors="red")
        plt.ylabel("No. of landmarks")
        plt.xlabel("Iteration")
        plt.show()
        plt.plot(range(iter), loss_hist, c="red", label="loss")
        plt.ylabel("Loss (binary crossentropy)")
        plt.xlabel("Iteration")
        plt.legend()
        plt.show()
        plt.plot(range(iter), cumulative_alignment_hist, label="in (landmarks)")
        plt.plot(range(iter), cumulative_out_hist, label="out")
        plt.plot(range(iter), alignment_all_hist, label="all")
        plt.ylabel("Alignment loss (MSE)")
        plt.xlabel("Iteration")
        plt.legend()
        plt.show()

        if debug:
            plt.plot(range(iter), cumulative_cos_in, label="cos in")
            plt.plot(range(iter), cumulative_cos_out, label="cos out")
            plt.legend()
            plt.show()

        plt.plot(range(iter), cumulative_overlap_hist, label="overlap")

        plt.ylabel("Jaccard Index", fontsize=16)
        plt.xlabel("Iteration", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # plt.legend()
        plt.tight_layout()
        plt.savefig("overlap.pdf", format="pdf")

    t_total = time.time() - t_begin
    if verbose == 1:
        print("--- RUNNING TIME ---")
        print("init: %.3f seconds" % t_init)
        print("sampling: %.3f seconds" % t_sampling)
        print("training: %.3f seconds" % t_train)
        print("predict: %.3f seconds" % t_predict)
        print("update: %.3f seconds" % t_update)
        print("realign: %.3f seconds" % t_update)
        print("---"*30)
        print("total: %.3f seconds" % t_total)

    if update_landmarks:
        if not return_model:
            return landmarks, non_landmarks, Q
        else:
            return landmarks, non_landmarks, Q, model
    else:
        return model


def main():
    """
    Runs main experiments using self supervised alignment.
    """
    # wv_source = "wordvectors/latin/corpus1/0.vec"
    # wv_target = "wordvectors/latin/corpus2/0.vec"
    # wv_source = "wordvectors/source/theguardianuk.vec"
    # wv_target = "wordvectors/source/thenewyorktimes_1.vec"
    wv_source = "wordvectors/semeval/latin-corpus1.vec"
    wv_target = "wordvectors/semeval/latin-corpus2.vec"
    # wv_source = "wordvectors/usuk/bnc.vec"
    # wv_target = "wordvectors/usuk/coca_mag.vec"
    # wv_source = "wordvectors/artificial/NYT-0.vec"
    # wv_target = "wordvectors/artificial/NYT-500_random.vec"
    plt.style.use("seaborn")

    # Read WordVectors
    normalized = False
    wv1 = WordVectors(input_file=wv_source, normalized=normalized)
    wv2 = WordVectors(input_file=wv_target, normalized=normalized)

    wv1, wv2 = intersection(wv1, wv2)

    landmarks, non_landmarks, Q = s4(wv1, wv2,
                                                            cls_model="nn",
                                                            n_targets=100,
                                                            n_negatives=100,
                                                            rate=1,
                                                            t=0.5,
                                                            iters=100,
                                                            verbose=1,
                                                            plot=1)
    wv1, wv2, Q = align(wv1, wv2, anchor_words=landmarks)
    d_l = [cosine(wv1[w], wv2[w]) for w in landmarks]
    d_n = [cosine(wv1[w], wv2[w]) for w in non_landmarks]
    sns.distplot(d_l, color="blue")
    sns.distplot(d_n, color="red")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
