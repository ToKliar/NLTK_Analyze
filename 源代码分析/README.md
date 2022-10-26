# **源代码分析**

这部分对NLTK库源代码分析   

分析算法的具体实现和代码架构

## **代码架构**

NLTK的各个功能实现在各个包里，大部分包（chunk、tag、tokenize、classify等等）集成了解决对应自然语言处理任务的多种机器学习算法的实现。   

这些包整体上按照接口模式进行架构，每一个包对应一种自然语言处理任务：
+ api.py文件中定义了接口，包含特定自然语言任务需要的方法和常量。通过接口，使用者可以用统一的方式使用统一的方法完成相应的任务。包中的所有模型都应该实现接口中定义的所有方法。
+ util.py文件中定义了特定自然语言处理任务需要的通用的工具方法。
+ 其余每个文件对应了一种特定自然语言处理任务的算法的实现，每个模型都需要基于接口定义的方法来实现对应的算法。

### **架构示例**

下面以tag包为例，展示这种架构

tag包中的api.py文件定义了TaggerI接口，其中注解为@abstractmethod是需要模型实现的抽象方法，其他没有注解的方法则是基于抽象方法实现的功能方法。

python语言没有接口相关的定义，python通过抽象类实现Interface，通过抽象方法定义接口的派生类需要实现的方法。

#### **接口定义**

TaggerI接口可以看成是对整个Tag任务的功能抽象，可以对Tag任务进行进一步抽象，如将基于特征集进行Tag任务的模型进行抽象，得到FeaturesetTaggerI接口，这里需要注意FeaturesetTaggerI是基于TaggerI定义的，是TaggerI的子类。

```python
class TaggerI(metaclass=ABCMeta):
    """
    给token序列中的每个token打上标记
    """

    @abstractmethod
    def tag(self, tokens):
        """
        采用某一标记算法给token序列打上标记，返回标记后的token序列
        """
        if overridden(self.tag_sents):
            return self.tag_sents([tokens])[0]

    def tag_sents(self, sentences):
        """
        给句子序列中的每一句（一句对应一个token序列）打标记
        """
        return [self.tag(sent) for sent in sentences]

    @deprecated("Use accuracy(gold) instead.")
    def evaluate(self, gold):
        return self.accuracy(gold)

    def accuracy(self, gold):
        """
        将gold作为正确的标注结果，计算实际标注结果的准确率
        """
        tagged_sents = self.tag_sents(untag(sent) for sent in gold)
        gold_tokens = list(chain.from_iterable(gold))
        test_tokens = list(chain.from_iterable(tagged_sents))
        return accuracy(gold_tokens, test_tokens)

    @lru_cache(maxsize=1)
    def _confusion_cached(self, gold):
        """
        将gold作为正确的标注结果，计算实际标注结果对应的混淆矩阵
        """
        tagged_sents = self.tag_sents(untag(sent) for sent in gold)
        gold_tokens = [token for _word, token in chain.from_iterable(gold)]
        test_tokens = [token for _word, token in chain.from_iterable(tagged_sents)]
        return ConfusionMatrix(gold_tokens, test_tokens)

    def confusion(self, gold):
        """
        封装_confusion_cached，给外界调用
        """

        return self._confusion_cached(tuple(tuple(sent) for sent in gold))

    def recall(self, gold) -> Dict[str, float]:
        """
        将gold作为正确的标注结果，计算实际标注结果的召回率
        """

        cm = self.confusion(gold)
        return {tag: cm.recall(tag) for tag in cm._values}

    def precision(self, gold):
        """
        将gold作为正确的标注结果，计算实际标注结果的精确率
        """

        cm = self.confusion(gold)
        return {tag: cm.precision(tag) for tag in cm._values}

    def f_measure(self, gold, alpha=0.5):
        """
        将gold作为正确的标注结果，计算实际标注结果的f-score，alpha为计算中false-negative的权重
        """
        cm = self.confusion(gold)
        return {tag: cm.f_measure(tag, alpha) for tag in cm._values}

    def evaluate_per_tag(self, gold, alpha=0.5, truncate=None, sort_by_count=False):
        """
        对于每个tag计算其对应的recall、precision和f-score，并用表格的形式展示
        """
        cm = self.confusion(gold)
        return cm.evaluate(alpha=alpha, truncate=truncate, sort_by_count=sort_by_count)

    def _check_params(self, train, model):
        if (train and model) or (not train and not model):
            raise ValueError("Must specify either training data or trained model.")
```

#### **工具方法**

util.py文件中则定义了Tag任务需要使用的工具方法

```python
def str2tuple(s, sep="/"):
    """
    将带标记的字符串形式的token转换为tuple
    """
    loc = s.rfind(sep)
    if loc >= 0:
        return (s[:loc], s[loc + len(sep) :].upper())
    else:
        return (s, None)


def tuple2str(tagged_token, sep="/"):
    """
    将带标记的tuple形式的token转换为字符串
    """
    word, tag = tagged_token
    if tag is None:
        return word
    else:
        assert sep not in tag, "tag may not contain sep!"
        return f"{word}{sep}{tag}"


def untag(tagged_sentence):
    """
    对于已经标记的句子，返回这个句子未标记的版本
    """
    return [w for (w, t) in tagged_sentence]
```

#### **模型实现**

在tag包中基于感知机``percptron``、条件随机场``crf``、隐马尔可夫链``HMM``等算法实现Tagger来处理词性标注Tag任务，每一个实际进行Tag任务的Tagger类都需要以TaggerI或FeatureTaggerI为父类

```python
class HiddenMarkovModelTagger(TaggerI):
    pass

class CRFTagger(TaggerI):
    pass 

class BrillTagger(TaggerI):
    pass 

class PerceptronTagger(TaggerI):
    pass 

class StanfordTagger(TaggerI):
    pass
```

### **算法实现示例 —— HMM**

这里以基于隐马尔可夫链算法实现的模型HiddenMarkovModelTagger为例，分析NLTK对特定机器学习算法的实现 

#### **基本性质** 

隐马尔可夫模型HMM主要用于为数据序列中的每一个数据分配正确的标签和评估根据数据序列生成给定标签序列的概率。   

HMM通过从可观察的标签序列中确定标签序列生成过程中的隐含的状态，利用这些状态和状态-标签之间转换的概率进行进一步的分析。

HMM模型是有限状态机，其中的特征是状态、状态之间的转换和每个状态输出的标签符号。  

HMM中observation代表着某个状态对应的一个概率函数。   

HMM基于马尔可夫链的假设，从一个状态转换到另一个状态的概率只取决于当前状态。  

HMM基于Viterbi算法找到给定状态序列生成的最大概率的标签序列。

基于以上内容，可以看出HMM具有时不变性质，即将HMM中的状态序列进行位移生成的标签序列等于HMM基于位移前的状态序列生成的标签序列在内容上是一致的。（时间的变化不会影响标签序列的生成）

#### **HMM特征表示**

HMM本质上是一个有向图，每条边的权重为概率（状态到标签的概率，状态到状态的概率），每个状态会非确定性的生成一个标签，外界可以看到HMM的输出的标签序列，对应的状态序列不可见，
![](./image/HMM.png)

计算过程中HMM的特征如下：
+ 输出的标签序列
+ 状态集 
+ 从某一状态转换到另一个状态的概率 $a_{ij} = P(s_t = j | s_{t-1} = i), s_t$为t时刻的状态
+ 从某一状态生成某一标签的概率 $b_i(k) = P(o_t = k | s_t = i), o_t$为t时刻的标签
+ 初始状态概率分布，第一个状态是某一状态的概率 

#### **具体实现**

HMM分为两个部分，第一部分是使用HMM进行标签序列的生成，第二部分是训练HMM，找到最合适的的状态参数。

下面解析HiddenMarkovModelTagger中的主要方法和实现方式。

```python
class HiddenMarkovModelTagger(TaggerI):
    """
    HiddenMarkovModelTagger的初始化需要以下参数
    symbols：标签序列
    states：状态序列，需要自己给出
    transitions：从某一状态转换到另一个状态的概率，基于ConditionalProbDistI接口
    outputs：从某一状态生成某一标签的概率，基于ConditionalProbDistI
    priors：初始状态的概率分布，基于ProbDistI
    transform：
    :param transform: 用于转换状态序列，可选，默认不做任何转换
    """

    def __init__(
        self, symbols, states, transitions, outputs, priors, transform=_identity
    ):
        # 将状态集和标签集去重
        self._symbols = unique_list(symbols) 
        self._states = unique_list(states) 
        self._transitions = transitions
        self._outputs = outputs
        self._priors = priors
        self._cache = None
        self._transform = transform

    @classmethod
    def _train(
        cls,
        labeled_sequence,
        test_sequence=None,
        unlabeled_sequence=None,
        transform=_identity,
        estimator=None,
        **kwargs,
    ):

        if estimator is None:

            def estimator(fd, bins):
                return LidstoneProbDist(fd, 0.1, bins)

        labeled_sequence = LazyMap(transform, labeled_sequence)
        symbols = unique_list(word for sent in labeled_sequence for word, tag in sent)
        tag_set = unique_list(tag for sent in labeled_sequence for word, tag in sent)

        trainer = HiddenMarkovModelTrainer(tag_set, symbols)
        hmm = trainer.train_supervised(labeled_sequence, estimator=estimator)
        hmm = cls(
            hmm._symbols,
            hmm._states,
            hmm._transitions,
            hmm._outputs,
            hmm._priors,
            transform=transform,
        )

        if test_sequence:
            hmm.test(test_sequence, verbose=kwargs.get("verbose", False))

        if unlabeled_sequence:
            max_iterations = kwargs.get("max_iterations", 5)
            hmm = trainer.train_unsupervised(
                unlabeled_sequence, model=hmm, max_iterations=max_iterations
            )
            if test_sequence:
                hmm.test(test_sequence, verbose=kwargs.get("verbose", False))

        return hmm

    @classmethod
    def train(
        cls, labeled_sequence, test_sequence=None, unlabeled_sequence=None, **kwargs
    ):
        """
        使用cls(HiddenMarkovModelTrainer对象)根据带标签的序列和无标签的序列训练得到一个HMM模型

        labeled_sequence: 带标签的训练序列
        test_sequence: 带标签的测试序列
        unlabeled_sequence: 不带标签的训练序列
        kwargs中可选参数为
        estimator 
        :param estimator: an optional function or class that maps a
            condition's frequency distribution to its probability
            distribution, defaults to a Lidstone distribution with gamma = 0.1

        verbose: 是否展示训练信息
        max_iterations: Baum-Welch训练算法的迭代次数
        """
        return cls._train(labeled_sequence, test_sequence, unlabeled_sequence, **kwargs)

    def probability(self, sequence):
        """
        Returns the probability of the given symbol sequence. If the sequence
        is labelled, then returns the joint probability of the symbol, state
        sequence. Otherwise, uses the forward algorithm to find the
        probability over all label sequences.

        :return: the probability of the sequence
        :rtype: float
        :param sequence: the sequence of symbols which must contain the TEXT
            property, and optionally the TAG property
        :type sequence:  Token
        """
        return 2 ** (self.log_probability(self._transform(sequence)))

    def log_probability(self, sequence):
        """
        Returns the log-probability of the given symbol sequence. If the
        sequence is labelled, then returns the joint log-probability of the
        symbol, state sequence. Otherwise, uses the forward algorithm to find
        the log-probability over all label sequences.

        :return: the log-probability of the sequence
        :rtype: float
        :param sequence: the sequence of symbols which must contain the TEXT
            property, and optionally the TAG property
        :type sequence:  Token
        """
        sequence = self._transform(sequence)

        T = len(sequence)

        if T > 0 and sequence[0][_TAG]:
            last_state = sequence[0][_TAG]
            p = self._priors.logprob(last_state) + self._output_logprob(
                last_state, sequence[0][_TEXT]
            )
            for t in range(1, T):
                state = sequence[t][_TAG]
                p += self._transitions[last_state].logprob(
                    state
                ) + self._output_logprob(state, sequence[t][_TEXT])
                last_state = state
            return p
        else:
            alpha = self._forward_probability(sequence)
            p = logsumexp2(alpha[T - 1])
            return p

    def _tag(self, unlabeled_sequence):
        path = self._best_path(unlabeled_sequence)
        return list(zip(unlabeled_sequence, path))

    def _output_logprob(self, state, symbol):
        """
        :return: the log probability of the symbol being observed in the given
            state
        :rtype: float
        """
        return self._outputs[state].logprob(symbol)

    def _create_cache(self):
        """
        The cache is a tuple (P, O, X, S) where:

          - S maps symbols to integers.  I.e., it is the inverse
            mapping from self._symbols; for each symbol s in
            self._symbols, the following is true::

              self._symbols[S[s]] == s

          - O is the log output probabilities::

              O[i,k] = log( P(token[t]=sym[k]|tag[t]=state[i]) )

          - X is the log transition probabilities::

              X[i,j] = log( P(tag[t]=state[j]|tag[t-1]=state[i]) )

          - P is the log prior probabilities::

              P[i] = log( P(tag[0]=state[i]) )
        """
        if not self._cache:
            N = len(self._states)
            M = len(self._symbols)
            P = np.zeros(N, np.float32)
            X = np.zeros((N, N), np.float32)
            O = np.zeros((N, M), np.float32)
            for i in range(N):
                si = self._states[i]
                P[i] = self._priors.logprob(si)
                for j in range(N):
                    X[i, j] = self._transitions[si].logprob(self._states[j])
                for k in range(M):
                    O[i, k] = self._output_logprob(si, self._symbols[k])
            S = {}
            for k in range(M):
                S[self._symbols[k]] = k
            self._cache = (P, O, X, S)

    def _update_cache(self, symbols):
        # add new symbols to the symbol table and repopulate the output
        # probabilities and symbol table mapping
        if symbols:
            self._create_cache()
            P, O, X, S = self._cache
            for symbol in symbols:
                if symbol not in self._symbols:
                    self._cache = None
                    self._symbols.append(symbol)
            # don't bother with the work if there aren't any new symbols
            if not self._cache:
                N = len(self._states)
                M = len(self._symbols)
                Q = O.shape[1]
                # add new columns to the output probability table without
                # destroying the old probabilities
                O = np.hstack([O, np.zeros((N, M - Q), np.float32)])
                for i in range(N):
                    si = self._states[i]
                    # only calculate probabilities for new symbols
                    for k in range(Q, M):
                        O[i, k] = self._output_logprob(si, self._symbols[k])
                # only create symbol mappings for new symbols
                for k in range(Q, M):
                    S[self._symbols[k]] = k
                self._cache = (P, O, X, S)

    def best_path(self, unlabeled_sequence):
        """
        Returns the state sequence of the optimal (most probable) path through
        the HMM. Uses the Viterbi algorithm to calculate this part by dynamic
        programming.

        :return: the state sequence
        :rtype: sequence of any
        :param unlabeled_sequence: the sequence of unlabeled symbols
        :type unlabeled_sequence: list
        """
        unlabeled_sequence = self._transform(unlabeled_sequence)
        return self._best_path(unlabeled_sequence)

    def _best_path(self, unlabeled_sequence):
        T = len(unlabeled_sequence)
        N = len(self._states)
        self._create_cache()
        self._update_cache(unlabeled_sequence)
        P, O, X, S = self._cache

        V = np.zeros((T, N), np.float32)
        B = -np.ones((T, N), int)

        V[0] = P + O[:, S[unlabeled_sequence[0]]]
        for t in range(1, T):
            for j in range(N):
                vs = V[t - 1, :] + X[:, j]
                best = np.argmax(vs)
                V[t, j] = vs[best] + O[j, S[unlabeled_sequence[t]]]
                B[t, j] = best

        current = np.argmax(V[T - 1, :])
        sequence = [current]
        for t in range(T - 1, 0, -1):
            last = B[t, current]
            sequence.append(last)
            current = last

        sequence.reverse()
        return list(map(self._states.__getitem__, sequence))

    def best_path_simple(self, unlabeled_sequence):
        """
        Returns the state sequence of the optimal (most probable) path through
        the HMM. Uses the Viterbi algorithm to calculate this part by dynamic
        programming.  This uses a simple, direct method, and is included for
        teaching purposes.

        :return: the state sequence
        :rtype: sequence of any
        :param unlabeled_sequence: the sequence of unlabeled symbols
        :type unlabeled_sequence: list
        """
        unlabeled_sequence = self._transform(unlabeled_sequence)
        return self._best_path_simple(unlabeled_sequence)

    def _best_path_simple(self, unlabeled_sequence):
        T = len(unlabeled_sequence)
        N = len(self._states)
        V = np.zeros((T, N), np.float64)
        B = {}

        # find the starting log probabilities for each state
        symbol = unlabeled_sequence[0]
        for i, state in enumerate(self._states):
            V[0, i] = self._priors.logprob(state) + self._output_logprob(state, symbol)
            B[0, state] = None

        # find the maximum log probabilities for reaching each state at time t
        for t in range(1, T):
            symbol = unlabeled_sequence[t]
            for j in range(N):
                sj = self._states[j]
                best = None
                for i in range(N):
                    si = self._states[i]
                    va = V[t - 1, i] + self._transitions[si].logprob(sj)
                    if not best or va > best[0]:
                        best = (va, si)
                V[t, j] = best[0] + self._output_logprob(sj, symbol)
                B[t, sj] = best[1]

        # find the highest probability final state
        best = None
        for i in range(N):
            val = V[T - 1, i]
            if not best or val > best[0]:
                best = (val, self._states[i])

        # traverse the back-pointers B to find the state sequence
        current = best[1]
        sequence = [current]
        for t in range(T - 1, 0, -1):
            last = B[t, current]
            sequence.append(last)
            current = last

        sequence.reverse()
        return sequence

    def random_sample(self, rng, length):
        """
        Randomly sample the HMM to generate a sentence of a given length. This
        samples the prior distribution then the observation distribution and
        transition distribution for each subsequent observation and state.
        This will mostly generate unintelligible garbage, but can provide some
        amusement.

        :return:        the randomly created state/observation sequence,
                        generated according to the HMM's probability
                        distributions. The SUBTOKENS have TEXT and TAG
                        properties containing the observation and state
                        respectively.
        :rtype:         list
        :param rng:     random number generator
        :type rng:      Random (or any object with a random() method)
        :param length:  desired output length
        :type length:   int
        """

        # sample the starting state and symbol prob dists
        tokens = []
        state = self._sample_probdist(self._priors, rng.random(), self._states)
        symbol = self._sample_probdist(
            self._outputs[state], rng.random(), self._symbols
        )
        tokens.append((symbol, state))

        for i in range(1, length):
            # sample the state transition and symbol prob dists
            state = self._sample_probdist(
                self._transitions[state], rng.random(), self._states
            )
            symbol = self._sample_probdist(
                self._outputs[state], rng.random(), self._symbols
            )
            tokens.append((symbol, state))

        return tokens

    def _sample_probdist(self, probdist, p, samples):
        cum_p = 0
        for sample in samples:
            add_p = probdist.prob(sample)
            if cum_p <= p <= cum_p + add_p:
                return sample
            cum_p += add_p
        raise Exception("Invalid probability distribution - " "does not sum to one")

    def entropy(self, unlabeled_sequence):
        """
        Returns the entropy over labellings of the given sequence. This is
        given by::

            H(O) = - sum_S Pr(S | O) log Pr(S | O)

        where the summation ranges over all state sequences, S. Let
        *Z = Pr(O) = sum_S Pr(S, O)}* where the summation ranges over all state
        sequences and O is the observation sequence. As such the entropy can
        be re-expressed as::

            H = - sum_S Pr(S | O) log [ Pr(S, O) / Z ]
            = log Z - sum_S Pr(S | O) log Pr(S, 0)
            = log Z - sum_S Pr(S | O) [ log Pr(S_0) + sum_t Pr(S_t | S_{t-1}) + sum_t Pr(O_t | S_t) ]

        The order of summation for the log terms can be flipped, allowing
        dynamic programming to be used to calculate the entropy. Specifically,
        we use the forward and backward probabilities (alpha, beta) giving::

            H = log Z - sum_s0 alpha_0(s0) beta_0(s0) / Z * log Pr(s0)
            + sum_t,si,sj alpha_t(si) Pr(sj | si) Pr(O_t+1 | sj) beta_t(sj) / Z * log Pr(sj | si)
            + sum_t,st alpha_t(st) beta_t(st) / Z * log Pr(O_t | st)

        This simply uses alpha and beta to find the probabilities of partial
        sequences, constrained to include the given state(s) at some point in
        time.
        """
        unlabeled_sequence = self._transform(unlabeled_sequence)

        T = len(unlabeled_sequence)
        N = len(self._states)

        alpha = self._forward_probability(unlabeled_sequence)
        beta = self._backward_probability(unlabeled_sequence)
        normalisation = logsumexp2(alpha[T - 1])

        entropy = normalisation

        # starting state, t = 0
        for i, state in enumerate(self._states):
            p = 2 ** (alpha[0, i] + beta[0, i] - normalisation)
            entropy -= p * self._priors.logprob(state)
            # print('p(s_0 = %s) =' % state, p)

        # state transitions
        for t0 in range(T - 1):
            t1 = t0 + 1
            for i0, s0 in enumerate(self._states):
                for i1, s1 in enumerate(self._states):
                    p = 2 ** (
                        alpha[t0, i0]
                        + self._transitions[s0].logprob(s1)
                        + self._outputs[s1].logprob(unlabeled_sequence[t1][_TEXT])
                        + beta[t1, i1]
                        - normalisation
                    )
                    entropy -= p * self._transitions[s0].logprob(s1)
                    # print('p(s_%d = %s, s_%d = %s) =' % (t0, s0, t1, s1), p)

        # symbol emissions
        for t in range(T):
            for i, state in enumerate(self._states):
                p = 2 ** (alpha[t, i] + beta[t, i] - normalisation)
                entropy -= p * self._outputs[state].logprob(
                    unlabeled_sequence[t][_TEXT]
                )
                # print('p(s_%d = %s) =' % (t, state), p)

        return entropy

    def point_entropy(self, unlabeled_sequence):
        """
        Returns the pointwise entropy over the possible states at each
        position in the chain, given the observation sequence.
        """
        unlabeled_sequence = self._transform(unlabeled_sequence)

        T = len(unlabeled_sequence)
        N = len(self._states)

        alpha = self._forward_probability(unlabeled_sequence)
        beta = self._backward_probability(unlabeled_sequence)
        normalisation = logsumexp2(alpha[T - 1])

        entropies = np.zeros(T, np.float64)
        probs = np.zeros(N, np.float64)
        for t in range(T):
            for s in range(N):
                probs[s] = alpha[t, s] + beta[t, s] - normalisation

            for s in range(N):
                entropies[t] -= 2 ** (probs[s]) * probs[s]

        return entropies

    def _exhaustive_entropy(self, unlabeled_sequence):
        unlabeled_sequence = self._transform(unlabeled_sequence)

        T = len(unlabeled_sequence)
        N = len(self._states)

        labellings = [[state] for state in self._states]
        for t in range(T - 1):
            current = labellings
            labellings = []
            for labelling in current:
                for state in self._states:
                    labellings.append(labelling + [state])

        log_probs = []
        for labelling in labellings:
            labeled_sequence = unlabeled_sequence[:]
            for t, label in enumerate(labelling):
                labeled_sequence[t] = (labeled_sequence[t][_TEXT], label)
            lp = self.log_probability(labeled_sequence)
            log_probs.append(lp)
        normalisation = _log_add(*log_probs)

        entropy = 0
        for lp in log_probs:
            lp -= normalisation
            entropy -= 2 ** (lp) * lp

        return entropy

    def _exhaustive_point_entropy(self, unlabeled_sequence):
        unlabeled_sequence = self._transform(unlabeled_sequence)

        T = len(unlabeled_sequence)
        N = len(self._states)

        labellings = [[state] for state in self._states]
        for t in range(T - 1):
            current = labellings
            labellings = []
            for labelling in current:
                for state in self._states:
                    labellings.append(labelling + [state])

        log_probs = []
        for labelling in labellings:
            labelled_sequence = unlabeled_sequence[:]
            for t, label in enumerate(labelling):
                labelled_sequence[t] = (labelled_sequence[t][_TEXT], label)
            lp = self.log_probability(labelled_sequence)
            log_probs.append(lp)

        normalisation = _log_add(*log_probs)

        probabilities = _ninf_array((T, N))

        for labelling, lp in zip(labellings, log_probs):
            lp -= normalisation
            for t, label in enumerate(labelling):
                index = self._states.index(label)
                probabilities[t, index] = _log_add(probabilities[t, index], lp)

        entropies = np.zeros(T, np.float64)
        for t in range(T):
            for s in range(N):
                entropies[t] -= 2 ** (probabilities[t, s]) * probabilities[t, s]

        return entropies

    def _transitions_matrix(self):
        """Return a matrix of transition log probabilities."""
        trans_iter = (
            self._transitions[sj].logprob(si)
            for sj in self._states
            for si in self._states
        )

        transitions_logprob = np.fromiter(trans_iter, dtype=np.float64)
        N = len(self._states)
        return transitions_logprob.reshape((N, N)).T

    def _outputs_vector(self, symbol):
        """
        Return a vector with log probabilities of emitting a symbol
        when entering states.
        """
        out_iter = (self._output_logprob(sj, symbol) for sj in self._states)
        return np.fromiter(out_iter, dtype=np.float64)

    def _forward_probability(self, unlabeled_sequence):
        """
        Return the forward probability matrix, a T by N array of
        log-probabilities, where T is the length of the sequence and N is the
        number of states. Each entry (t, s) gives the probability of being in
        state s at time t after observing the partial symbol sequence up to
        and including t.

        :param unlabeled_sequence: the sequence of unlabeled symbols
        :type unlabeled_sequence: list
        :return: the forward log probability matrix
        :rtype: array
        """
        T = len(unlabeled_sequence)
        N = len(self._states)
        alpha = _ninf_array((T, N))

        transitions_logprob = self._transitions_matrix()

        # Initialization
        symbol = unlabeled_sequence[0][_TEXT]
        for i, state in enumerate(self._states):
            alpha[0, i] = self._priors.logprob(state) + self._output_logprob(
                state, symbol
            )

        # Induction
        for t in range(1, T):
            symbol = unlabeled_sequence[t][_TEXT]
            output_logprob = self._outputs_vector(symbol)

            for i in range(N):
                summand = alpha[t - 1] + transitions_logprob[i]
                alpha[t, i] = logsumexp2(summand) + output_logprob[i]

        return alpha

    def _backward_probability(self, unlabeled_sequence):
        """
        Return the backward probability matrix, a T by N array of
        log-probabilities, where T is the length of the sequence and N is the
        number of states. Each entry (t, s) gives the probability of being in
        state s at time t after observing the partial symbol sequence from t
        .. T.

        :return: the backward log probability matrix
        :rtype:  array
        :param unlabeled_sequence: the sequence of unlabeled symbols
        :type unlabeled_sequence: list
        """
        T = len(unlabeled_sequence)
        N = len(self._states)
        beta = _ninf_array((T, N))

        transitions_logprob = self._transitions_matrix().T

        # initialise the backward values;
        # "1" is an arbitrarily chosen value from Rabiner tutorial
        beta[T - 1, :] = np.log2(1)

        # inductively calculate remaining backward values
        for t in range(T - 2, -1, -1):
            symbol = unlabeled_sequence[t + 1][_TEXT]
            outputs = self._outputs_vector(symbol)

            for i in range(N):
                summand = transitions_logprob[i] + beta[t + 1] + outputs
                beta[t, i] = logsumexp2(summand)

        return beta

    def test(self, test_sequence, verbose=False, **kwargs):
        """
        Tests the HiddenMarkovModelTagger instance.

        :param test_sequence: a sequence of labeled test instances
        :type test_sequence: list(list)
        :param verbose: boolean flag indicating whether training should be
            verbose or include printed output
        :type verbose: bool
        """

        def words(sent):
            return [word for (word, tag) in sent]

        def tags(sent):
            return [tag for (word, tag) in sent]

        def flatten(seq):
            return list(itertools.chain(*seq))

        test_sequence = self._transform(test_sequence)
        predicted_sequence = list(map(self._tag, map(words, test_sequence)))

        if verbose:
            for test_sent, predicted_sent in zip(test_sequence, predicted_sequence):
                print(
                    "Test:",
                    " ".join(f"{token}/{tag}" for (token, tag) in test_sent),
                )
                print()
                print("Untagged:", " ".join("%s" % token for (token, tag) in test_sent))
                print()
                print(
                    "HMM-tagged:",
                    " ".join(f"{token}/{tag}" for (token, tag) in predicted_sent),
                )
                print()
                print(
                    "Entropy:",
                    self.entropy([(token, None) for (token, tag) in predicted_sent]),
                )
                print()
                print("-" * 60)

        test_tags = flatten(map(tags, test_sequence))
        predicted_tags = flatten(map(tags, predicted_sequence))

        acc = accuracy(test_tags, predicted_tags)
        count = sum(len(sent) for sent in test_sequence)
        print("accuracy over %d tokens: %.2f" % (count, acc * 100))

    def __repr__(self):
        return "<HiddenMarkovModelTagger %d states and %d output symbols>" % (
            len(self._states),
            len(self._symbols),
        )


class HiddenMarkovModelTrainer:
    """
    Algorithms for learning HMM parameters from training data. These include
    both supervised learning (MLE) and unsupervised learning (Baum-Welch).

    Creates an HMM trainer to induce an HMM with the given states and
    output symbol alphabet. A supervised and unsupervised training
    method may be used. If either of the states or symbols are not given,
    these may be derived from supervised training.

    :param states:  the set of state labels
    :type states:   sequence of any
    :param symbols: the set of observation symbols
    :type symbols:  sequence of any
    """

    def __init__(self, states=None, symbols=None):
        self._states = states if states else []
        self._symbols = symbols if symbols else []

    def train(self, labeled_sequences=None, unlabeled_sequences=None, **kwargs):
        """
        Trains the HMM using both (or either of) supervised and unsupervised
        techniques.

        :return: the trained model
        :rtype: HiddenMarkovModelTagger
        :param labelled_sequences: the supervised training data, a set of
            labelled sequences of observations
            ex: [ (word_1, tag_1),...,(word_n,tag_n) ]
        :type labelled_sequences: list
        :param unlabeled_sequences: the unsupervised training data, a set of
            sequences of observations
            ex: [ word_1, ..., word_n ]
        :type unlabeled_sequences: list
        :param kwargs: additional arguments to pass to the training methods
        """
        assert labeled_sequences or unlabeled_sequences
        model = None
        if labeled_sequences:
            model = self.train_supervised(labeled_sequences, **kwargs)
        if unlabeled_sequences:
            if model:
                kwargs["model"] = model
            model = self.train_unsupervised(unlabeled_sequences, **kwargs)
        return model

    def _baum_welch_step(self, sequence, model, symbol_to_number):

        N = len(model._states)
        M = len(model._symbols)
        T = len(sequence)

        # compute forward and backward probabilities
        alpha = model._forward_probability(sequence)
        beta = model._backward_probability(sequence)

        # find the log probability of the sequence
        lpk = logsumexp2(alpha[T - 1])

        A_numer = _ninf_array((N, N))
        B_numer = _ninf_array((N, M))
        A_denom = _ninf_array(N)
        B_denom = _ninf_array(N)

        transitions_logprob = model._transitions_matrix().T

        for t in range(T):
            symbol = sequence[t][_TEXT]  # not found? FIXME
            next_symbol = None
            if t < T - 1:
                next_symbol = sequence[t + 1][_TEXT]  # not found? FIXME
            xi = symbol_to_number[symbol]

            next_outputs_logprob = model._outputs_vector(next_symbol)
            alpha_plus_beta = alpha[t] + beta[t]

            if t < T - 1:
                numer_add = (
                    transitions_logprob
                    + next_outputs_logprob
                    + beta[t + 1]
                    + alpha[t].reshape(N, 1)
                )
                A_numer = np.logaddexp2(A_numer, numer_add)
                A_denom = np.logaddexp2(A_denom, alpha_plus_beta)
            else:
                B_denom = np.logaddexp2(A_denom, alpha_plus_beta)

            B_numer[:, xi] = np.logaddexp2(B_numer[:, xi], alpha_plus_beta)

        return lpk, A_numer, A_denom, B_numer, B_denom

    def train_unsupervised(self, unlabeled_sequences, update_outputs=True, **kwargs):
        """
        Trains the HMM using the Baum-Welch algorithm to maximise the
        probability of the data sequence. This is a variant of the EM
        algorithm, and is unsupervised in that it doesn't need the state
        sequences for the symbols. The code is based on 'A Tutorial on Hidden
        Markov Models and Selected Applications in Speech Recognition',
        Lawrence Rabiner, IEEE, 1989.

        :return: the trained model
        :rtype: HiddenMarkovModelTagger
        :param unlabeled_sequences: the training data, a set of
            sequences of observations
        :type unlabeled_sequences: list

        kwargs may include following parameters:

        :param model: a HiddenMarkovModelTagger instance used to begin
            the Baum-Welch algorithm
        :param max_iterations: the maximum number of EM iterations
        :param convergence_logprob: the maximum change in log probability to
            allow convergence
        """

        # create a uniform HMM, which will be iteratively refined, unless
        # given an existing model
        model = kwargs.get("model")
        if not model:
            priors = RandomProbDist(self._states)
            transitions = DictionaryConditionalProbDist(
                {state: RandomProbDist(self._states) for state in self._states}
            )
            outputs = DictionaryConditionalProbDist(
                {state: RandomProbDist(self._symbols) for state in self._states}
            )
            model = HiddenMarkovModelTagger(
                self._symbols, self._states, transitions, outputs, priors
            )

        self._states = model._states
        self._symbols = model._symbols

        N = len(self._states)
        M = len(self._symbols)
        symbol_numbers = {sym: i for i, sym in enumerate(self._symbols)}

        # update model prob dists so that they can be modified
        # model._priors = MutableProbDist(model._priors, self._states)

        model._transitions = DictionaryConditionalProbDist(
            {
                s: MutableProbDist(model._transitions[s], self._states)
                for s in self._states
            }
        )

        if update_outputs:
            model._outputs = DictionaryConditionalProbDist(
                {
                    s: MutableProbDist(model._outputs[s], self._symbols)
                    for s in self._states
                }
            )

        model.reset_cache()

        # iterate until convergence
        converged = False
        last_logprob = None
        iteration = 0
        max_iterations = kwargs.get("max_iterations", 1000)
        epsilon = kwargs.get("convergence_logprob", 1e-6)

        while not converged and iteration < max_iterations:
            A_numer = _ninf_array((N, N))
            B_numer = _ninf_array((N, M))
            A_denom = _ninf_array(N)
            B_denom = _ninf_array(N)

            logprob = 0
            for sequence in unlabeled_sequences:
                sequence = list(sequence)
                if not sequence:
                    continue

                (
                    lpk,
                    seq_A_numer,
                    seq_A_denom,
                    seq_B_numer,
                    seq_B_denom,
                ) = self._baum_welch_step(sequence, model, symbol_numbers)

                # add these sums to the global A and B values
                for i in range(N):
                    A_numer[i] = np.logaddexp2(A_numer[i], seq_A_numer[i] - lpk)
                    B_numer[i] = np.logaddexp2(B_numer[i], seq_B_numer[i] - lpk)

                A_denom = np.logaddexp2(A_denom, seq_A_denom - lpk)
                B_denom = np.logaddexp2(B_denom, seq_B_denom - lpk)

                logprob += lpk

            # use the calculated values to update the transition and output
            # probability values
            for i in range(N):
                logprob_Ai = A_numer[i] - A_denom[i]
                logprob_Bi = B_numer[i] - B_denom[i]

                # We should normalize all probabilities (see p.391 Huang et al)
                # Let sum(P) be K.
                # We can divide each Pi by K to make sum(P) == 1.
                #   Pi' = Pi/K
                #   log2(Pi') = log2(Pi) - log2(K)
                logprob_Ai -= logsumexp2(logprob_Ai)
                logprob_Bi -= logsumexp2(logprob_Bi)

                # update output and transition probabilities
                si = self._states[i]

                for j in range(N):
                    sj = self._states[j]
                    model._transitions[si].update(sj, logprob_Ai[j])

                if update_outputs:
                    for k in range(M):
                        ok = self._symbols[k]
                        model._outputs[si].update(ok, logprob_Bi[k])

                # Rabiner says the priors don't need to be updated. I don't
                # believe him. FIXME

            # test for convergence
            if iteration > 0 and abs(logprob - last_logprob) < epsilon:
                converged = True

            print("iteration", iteration, "logprob", logprob)
            iteration += 1
            last_logprob = logprob

        return model

    def train_supervised(self, labelled_sequences, estimator=None):
        """
        Supervised training maximising the joint probability of the symbol and
        state sequences. This is done via collecting frequencies of
        transitions between states, symbol observations while within each
        state and which states start a sentence. These frequency distributions
        are then normalised into probability estimates, which can be
        smoothed if desired.

        :return: the trained model
        :rtype: HiddenMarkovModelTagger
        :param labelled_sequences: the training data, a set of
            labelled sequences of observations
        :type labelled_sequences: list
        :param estimator: a function taking
            a FreqDist and a number of bins and returning a CProbDistI;
            otherwise a MLE estimate is used
        """

        # default to the MLE estimate
        if estimator is None:
            estimator = lambda fdist, bins: MLEProbDist(fdist)

        # count occurrences of starting states, transitions out of each state
        # and output symbols observed in each state
        known_symbols = set(self._symbols)
        known_states = set(self._states)

        starting = FreqDist()
        transitions = ConditionalFreqDist()
        outputs = ConditionalFreqDist()
        for sequence in labelled_sequences:
            lasts = None
            for token in sequence:
                state = token[_TAG]
                symbol = token[_TEXT]
                if lasts is None:
                    starting[state] += 1
                else:
                    transitions[lasts][state] += 1
                outputs[state][symbol] += 1
                lasts = state

                # update the state and symbol lists
                if state not in known_states:
                    self._states.append(state)
                    known_states.add(state)

                if symbol not in known_symbols:
                    self._symbols.append(symbol)
                    known_symbols.add(symbol)

        # create probability distributions (with smoothing)
        N = len(self._states)
        pi = estimator(starting, N)
        A = ConditionalProbDist(transitions, estimator, N)
        B = ConditionalProbDist(outputs, estimator, len(self._symbols))

        return HiddenMarkovModelTagger(self._symbols, self._states, A, B, pi)
```
