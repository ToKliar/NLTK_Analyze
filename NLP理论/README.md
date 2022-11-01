# NLP理论

这部分对NLTK库涉及的机器学习理论进行分析，这里以HMM模型介绍相关理论和实现解析

## **算法实现示例 —— HMM**

这里以基于隐马尔可夫链算法实现的模型HiddenMarkovModelTagger为例，分析NLTK对特定机器学习算法的实现 

参考资料：
+ https://blog.csdn.net/u014688145/article/details/53046765
+ https://zhuanlan.zhihu.com/p/88362664

### **基本性质** 

隐马尔可夫模型HMM主要用于为数据序列中的每一个数据分配正确的标签和评估根据数据序列生成给定标签序列的概率。   

#### **HMM的理论基础**
HMM通过从可观察的标签序列中确定标签序列生成过程中的隐含的状态，利用这些状态和状态-标签之间转换的概率进行进一步的分析。

HMM基于马尔可夫链的假设，从一个状态转换到另一个状态的概率只取决于当前状态。

#### **HMM的特征和计算**
HMM模型是有限状态机，其中的特征是状态、状态之间的转换和每个状态输出的标签符号。  

HMM基于Viterbi算法找到给定状态序列生成的最大概率的标签序列。

#### **HMM的特征**

基于以上内容，可以看出HMM具有时不变性质，即将HMM中的状态序列进行位移生成的标签序列等于HMM基于位移前的状态序列生成的标签序列在内容上是一致的。（时间的变化不会影响标签序列的生成）

### **HMM特征表示**

HMM本质上是一个有向图，每条边的权重为概率（状态到标签的概率，状态到状态的概率），每个状态会非确定性的生成一个标签，外界可以看到HMM的输出的标签序列，对应的状态序列不可见，
![](./image/HMM.png)

计算过程中HMM的特征如下：
+ 输出的标签序列
+ 状态集 
+ 从某一状态转换到另一个状态的概率 $a_{ij} = P(s_t = j | s_{t-1} = i), s_t$为t时刻的状态
+ 从某一状态生成某一标签的概率 $b_i(k) = P(o_t = k | s_t = i), o_t$为t时刻的标签
+ 初始状态概率分布，第一个状态是某一状态的概率 

### **具体实现**

HMM分为两个部分，第一部分是使用HMM进行标签序列的生成，第二部分是训练HMM，找到最合适的的状态参数。

下面解析HiddenMarkovModelTagger中的主要方法和实现方式。

这里需要注意，对于词性标注任务，状态是word，标签是tag

NLTK中HMM的生成和训练如下：
```python
class HiddenMarkovModelTagger(TaggerI):
    """
    HiddenMarkovModelTagger的初始化需要以下参数
    symbols：标签序列
    states：状态序列，需要自己给出
    transitions：从某一状态转换到另一个状态的概率，基于ConditionalProbDistI接口
    outputs：从某一状态生成某一标签的概率，基于ConditionalProbDistI
    priors：初始状态的概率分布，基于ProbDistI
    transform：用于转换状态序列，可选，默认不做任何转换
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
        cls, labeled_sequence, test_sequence=None, unlabeled_sequence=None, transform=_identity, estimator=None, **kwargs,
    ):
        # HMM的训练过程，这里需要注意，对于带有标签的状态序列使用监督学习模式
        if estimator is None:
            def estimator(fd, bins):
                return LidstoneProbDist(fd, 0.1, bins)

        labeled_sequence = LazyMap(transform, labeled_sequence)
        symbols = unique_list(word for sent in labeled_sequence for word, tag in sent)
        tag_set = unique_list(tag for sent in labeled_sequence for word, tag in sent)

        trainer = HiddenMarkovModelTrainer(tag_set, symbols)
        # 监督学习方法根据带标签的状态序列，训练生成多个概率矩阵（初始概率，状态转移概率，状态生成标签概率）
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
            # 如果存在无标签的状态序列，在原模型的基础上使用无监督学习加强训练，并进行测试
            hmm = trainer.train_unsupervised(
                unlabeled_sequence, model=hmm, max_iterations=max_iterations
            )
            if test_sequence:
                hmm.test(test_sequence, verbose=kwargs.get("verbose", False))

        return hmm
```

HMM的应用主要有三个方面的问题：

#### **应用问题**：
已知模型参数和标签序列的情况下（状态序列是否已知不重要），计算标签序列出现的概率

HMM通过probability方法计算标签序列出现的概率，采用前向或者后向算法进行计算，代码中采用前向算法

```python
    def probability(self, sequence):
        return 2 ** (self.log_probability(self._transform(sequence)))

    def log_probability(self, sequence):
        """
        如果sequence是带有标签的状态序列，返回sequence中的状态序列生成标签序列的概率，

        如果sequence是不带状态的标签序列，计算标签序列出现概率的方法是，算出每个可能的状态序列生成标签序列的概率并求和。
        等价于：返回标签序列生成最后一个状态（最后一个状态可能是标签集合中任意一个标签）的概率之和

        因为序列长度比较长时，序列生成概率比较小，所以这里用对数的形式计算使得最后的结果更精准
        """
        sequence = self._transform(sequence) # 对序列进行转换，默认不转换

        T = len(sequence)

        if T > 0 and sequence[0][_TAG]:
            # 第一个标签的概率的对数为生成第一个状态的概率的对数加上第一个状态生成第一个标签的概率
            last_state = sequence[0][_TAG]
            p = self._priors.logprob(last_state) + self._output_logprob(
                last_state, sequence[0][_TEXT]
            ) 
            # 生成第i个标签的概率的对数为第i-1个状态转换到第i个状态的概率的对数加上第i个状态生成第i个标签的概率
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
```
前向算法是一个递推算法
+ 标签序列记为$[t_1, t_2, ..., t_T]$，状态序列为$[s_1, s_2, ..., s_T]$，状态集为$[o_1, o_2, ..., o_N]$，这里$t_i$是固定值，$s_i$非定值
+ $transition[i, k] = log(P(s_n = o_k | s_{n-1} = o_i)), n - 1 > 0$
+ $\lambda$表示模型中的参数

```
注意，为了方便推导，推导过程中的所有矩阵的下标和代码里的下标不一致
推到过程中矩阵下标从1开始，代码里从0开始
```

前向算法用alpha矩阵计算标签序列的概率，alpha矩阵为 T x N矩阵（T为序列长度，N为状态集大小），矩阵中的某个元素用来表示标签序列的某个前缀序列对应的状态序列最后一个状态为某个特定状态的概率。因为生成概率经过不断的乘法累积会变得比较小，所以用对数表示会更加方便计算。
+ $alpha[n, k] = log(P(t_1, t_2, ..., t_n, s_n = o_k | \lambda))$
+ $alpha[1, k] = log(P(s_1 = o_k | \lambda) * P(t_1 | s_1 = o_k))$

注意到对于长度为n的标签序列，其状态序列中最后一个状态为o_k的概率，为所有的前继状态（第n-1个状态）的概率乘以前继状态转移到o_k的概率再乘以o_k生成第n个标签的概率之和。具体推导如下：
+ $P(t_1, t_2, ..., t_n, s_n = o_k | \lambda) = \sum_{i=1}^N P(t_1, t_2, ..., t_{n-1}, s_n = o_i | \lambda) * P(s_n = o_k | s_{n-1} = o_i) * P(t_n | s_n = o_k) $
+ $alpha[n, k] = log(2^{output[k, t_n]} * \sum_{i=1}^N 2^{alpha[n-1,i] + transition[i, k]})$ 

```python
     def _forward_probability(self, unlabeled_sequence):
        T = len(unlabeled_sequence)
        N = len(self._states)
        alpha = _ninf_array((T, N))

        transitions_logprob = self._transitions_matrix()

        # Initialization
        # alpha[0, k]等于状态k的初始生成概率和状态k生成第一个标签的概率
        symbol = unlabeled_sequence[0][_TEXT]
        for i, state in enumerate(self._states):
            alpha[0, i] = self._priors.logprob(state) + self._output_logprob(
                state, symbol
            )

        # 不断前向传播，alpha[t, k]的概率为 第t-1个状态为任意状态i的概率乘以状态i转移到状态k的概率再求和，最后乘以状态k生成第k个标签的概率
        # 这里注意保存的概率是概率的对数，所以在计算下一个概率的对数的时候需要用logsumexp2计算
        for t in range(1, T):
            symbol = unlabeled_sequence[t][_TEXT]
            output_logprob = self._outputs_vector(symbol)

            for i in range(N):
                summand = alpha[t - 1] + transitions_logprob[i]
                alpha[t, i] = logsumexp2(summand) + output_logprob[i]

        return alpha
```

后向算法也是递推算法
+ 标签序列记为$[t_1, t_2, ..., t_T]$，状态序列为$[s_1, s_2, ..., s_T]$，状态集为$[o_1, o_2, ..., o_N]$，这里$t_i$是固定值，$s_i$非定值
+ $transition[i, k] = log(P(s_n = o_k | s_{n-1} = o_i)), n - 1 > 0$
+ $prior[i] = log(P(s_1 = o_i | \lambda))$

后向算法从后往前进行递推，beta矩阵表示某个后缀标签序列的前一个状态为某个特定状态的概率，即后面的标签序列为$[t_{n+1}, ..., t_T]$时，第n个状态为某一特定状态的概率，表达式如下：
+ $beta[n, k] = log(P(t_T, t_{T-1}, ..., t_{n+1} | s_n = o_k , \lambda))$
+ $beta[T, k] = log(1)$

递推方式的推导过程如下：
+ $P(t_T, t_{T-1}, ..., t_{n+1} | s_n = o_k , \lambda) = \sum_{i=1}^N P(t_T, t_{T-1}, ..., t_{n+1}, s_{n+1} = o_i | s_n = o_k , \lambda) = \sum_{i=1}^N P(t_T, t_{T-1}, ..., t_{n+2} | s_{n+1} = o_i, \lambda) * P (s_{n+1} = o_i | s_n = o_k) * P(t_{n+1} | s_{n + 1} = o_i) $
+ $beta[n, k] = log(P(t_T, t_{T-1}, ..., t_{n+1} | s_n = o_k , \lambda)) = log(\sum_{i=1}^N 2^{transition[i, k] * P(t_{n+1} | s_{n + 1} = o_i) * beta[n + 1, i]})$
+ 最终概率$P = \sum_{i=1}^N beta[1, i] * prior[i] * P(t_1 | s_1 = o_i)$

```python
    def _backward_probability(self, unlabeled_sequence):
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
```

#### **预测问题**：
也叫做解码问题，已知模型参数和标签序列，找到最可能的状态序列，使用Viterbi算法求解  

对于长度为T的标签序列，状态集的大小为N，则可能的状态序列的数量为$N^T$个，每个状态序列生成这个标签序列的概率是不同的，找到概率最大的那个状态序列。  
对应的Viterbi算法如下：
+ 标签序列记为$[t_1, t_2, ..., t_T]$，可能的状态序列为$[s_1, s_2, ..., s_T]$，状态集为$[o_1, o_2, ..., o_N]$
+ $P(t_1, t_2, ..., t_n | s_1, ..., s_n) = P(t_1, t_2, ..., t_{n-1} | s_1, s_2, ..., s_{n-1}) * P(s_n | s_{n-1}) * P(t_n | s_n) $
+ $V[n, k] = max(P(t_1, t_2, ..., t_n, s_n = o_k)) $即第n个状态为$o_k$的序列中最可能生成$[t_1, t_2, ..., t_n]$的状态序列的概率
+ $V[n, k] = max(V[n - 1, i] * X(o_i, o_k)) * O(o_k, t_n), i \in {1, ..., N}$

```python
    def _best_path(self, unlabeled_sequence):
        """
        unlabeled_sequence为一个标签序列，该函数负责找到最有可能的状态序列（状态序列生成这个标签序列的可能性最大）
        The cache is a tuple (P, O, X, S) where:
          - S 代表标签字典，一个标签对应一个序号
          - O 代表状态生成标签概率矩阵
          - X 代表状态转移概率矩阵
          - P 代表第一个状态是状态k的概率
        整体的算法如下：动态规划算法
        标签序列记为[t1, t2]
        最重要找的状态序列是[a1, a2, ..., at]这个序列在所有的序列中生成这个标签序列的可能性最大
        """
        unlabeled_sequence = self._transform(unlabeled_sequence)
        T = len(unlabeled_sequence)
        N = len(self._states)
        # 使用cache缓存计算所需的参数
        self._create_cache()
        self._update_cache(unlabeled_sequence)
        P, O, X, S = self._cache

        V = np.zeros((T, N), np.float32) # V[t, k] 根据前t个标签的序列生成的若干状态序列[a1, a2, ..., at = k]中概率最大的状态序列的概率 
        B = -np.ones((T, N), int) # B[t, k]  V[t, k]对应的序列中第t-1个状态

        V[0] = P + O[:, S[unlabeled_sequence[0]]]
        for t in range(1, T):
            for j in range(N):
                vs = V[t - 1, :] + X[:, j]
                best = np.argmax(vs) # 找到概率最大的状态序列
                V[t, j] = vs[best] + O[j, S[unlabeled_sequence[t]]]
                B[t, j] = best

        # 倒序找到概率最大的状态序列
        current = np.argmax(V[T - 1, :])
        sequence = [current]
        for t in range(T - 1, 0, -1):
            last = B[t, current]
            sequence.append(last)
            current = last

        sequence.reverse()
        return list(map(self._states.__getitem__, sequence))
```
#### **熵**
HMM支持计算标签序列的熵
+ 信息熵的计算为 $\sum p * log(p)$
+ 熵的计算方法为$\sum P(s_1, s_2, ..., s_T | t_1, t_2, ..., t_T) log(P(s_1, s_2, ..., s_n| t_1, t_2, ..., t_T)$
+ 这里计算就是根据标签序列生成状态序列的概率计算熵
+ $P(t_1, t_2, ..., t_T)$即标签序列的产生概率前面已经论述过计算方法
+ $P(s_1, s_2, ..., s_T | t_1, t_2, ..., t_T) = \frac{P(s_1, s_2, ..., s_T, t_1, t_2, ..., t_T)}{P(t_1, t_2, ..., t_T)}$

现在论述怎么计算熵，首先根据上面的推导，得到以下公式
$$alpha[n, k] = log(P(t_1, t_2, ..., t_n, s_n = o_k | \lambda))$$
$$beta[n, k] = log(P(t_T, t_{T-1}, ..., t_{n+1} | s_n = o_k , \lambda))$$
$$alpha[n, k] + beta[n, k] = log(P(t_1, t_2, ..., t_n, s_1 = o_k | \lambda) * P(t_T, t_{T-1}, ..., t_{n+1} | s_n = o_k , \lambda)) = log(P(t_1, t_2, ..., t_T, s_n = o_k | \lambda)) \tag{1}$$
$$alpha[n, k] + beta[n+1, i] + transition[k, i] + output[i, t_{n+1}] = log(P(t_1, ..., t_n, s_n = o_k | \lambda) * P(t_T, t_{T-1}, ..., t_{n+2} | s_{n+1} = o_i , \lambda) * P(s_{n+1}=o_i | s_n=o_k) * P(t_{n+1} | s_{n+1}=o_i)) = log(P(t_1, ..., t_T, s_n = o_k, s_{n+1} = o_i) \tag{2}$$

具体的计算过程：
$$entropy = \sum P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T)log(P(s_1,s_2,...,s_t | t_1, t_2,...,t_T))$$
$$可知\sum P(s_1, s_2, ..., s_T | t_1, t_2, ..., t_T) = 1 $$
$$又可知P(s_1, s_2,...,s_T|t_1, t_2, ...,t_T) = P(s_1, s_2, ..., s_T, t_1, t_2, ..., t_T) / P(t_1, t_2, ...,t_T)$$
$$令Z = P(t_1, t_2, ..., t_T)对应函数中的2^{normalisation}$$
$$\begin{aligned}
entropy &= \sum P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T)log(P(s_1, s_2, ..., s_T, t_1, t_2, ..., t_T) / P(t_1, t_2, ...,t_T)) \\
&= \sum P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T)log(P(s_1, s_2, ..., s_T, t_1, t_2, ..., t_T) / Z) \\
&= Z \sum P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T) - \sum P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T) log(P(s_1, s_2, ..., s_T, t_1, t_2, ..., t_T)) \\
&= Z - \sum P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T) log(P(s_1, s_2, ..., s_T, t_1, t_2, ..., t_T))
\end{aligned}$$
$$P(s_1, s_2, ..., s_T, t_1, t_2, ..., t_T) = \prod_{i=1}^{T} P(t_i | s_i) * \prod_{i=1}^{T-1} P(s_{i+1} | s_{i}) * P(s_1)$$
$$log((s_1, s_2, ..., s_T, t_1, t_2, ..., t_T)) = \sum_{i=1}^{T} log(P(t_i | s_i)) * \sum_{i=1}^{T-1} log(P(s_{i+1} | s_{i})) * log(P(s_1)) $$
将原式按照log项合并同类项可得，
$$entrypy = \sum_{i=1}^{N} (log(P(s_1=o_i)) \sum_{s_1=o_i} P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T)) +  \sum_{i=1}^{T} \sum_{j=1}^{N} (log(t_i | s_i = o_j) \sum_{s_i=o_j} P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T)) + \sum_{i=1}^{T-1} \sum_{p=1}^{N} \sum_{q=1}^{N} (log(s_{i+1}=o_q | s_i = o_p) \sum_{s_{i+1}=o_q , s_i = o_p} P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T))$$
对于entropy的新表达式可以分为三个部分，第一部分如下，代表初始状态生成的熵：
$$\begin{aligned} 
\sum_{s_1=o_i} P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T) &= P(s_1 = o_i | t_1, t_2, ..., t_T) \\
&= P(s_1 = o_i, t_1, t_2, .., t_T) | P(t_1, t_2, ...m t_T) \\
&= 2^{alpha[0][i] + alpha[0][i] - normalisation} & \text{根据公式(1)}
\end{aligned}$$
$$又因为prior[i] = log(P(s_1=o_i))$$
$$\sum_{i=1}^{N} (log(P(s_1=o_i)) \sum_{s_1=o_i} P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T)) = \sum_{i=1}^{N} prior[i] * 2^{alpha[1][i] + alpha[1][i] - normalisation}$$
第二部分推导如下，代表状态生成标签的熵：
$$\begin{aligned} 
\sum_{s_i=o_j} P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T) &= P(s_i = o_j | t_1, t_2, ..., t_T) \\
&= P(s_i=o_j, t_1, t_2, ..., t_T) / P(t_1, t_2, ..., t_T) \\
&= 2^{alpha[i,j]+beta[i,j]-normalisation}  & \text{根据公式(1)}
\end{aligned}$$
$$又因为output[o_j][t_i] = log(t_i | s_i = o_j)$$
$$\sum_{i=1}^{T} \sum_{j=1}^{N} (log(t_i | s_i = o_j) \sum_{s_i=o_j} P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T)) = \sum_{i=1}^{T} \sum_{j=1}^{N} output[o_j][t_i] 2^{alpha[i,j]+beta[i,j]-normalisation}$$
第三部分推导如下，代表状态转移的熵：
$$\begin{aligned} 
\sum_{s_{i+1}=o_q , s_i = o_p} P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T) &= P(s_{i+1}=o_q, s_i = o_p | t_1, t_2, ..., t_T) \\
&= P(s_{i+1} = o_q, s_i = o_p, t_1, t_2, ..., t_T) / P(t_1, t_2, ..., t_T) \\
&= 2^{alpha[i, p] + beta[i+1, q] + transition[p][q] + output[q][t_{n+1}] - normalisation} & \text{根据公式(2)}
\end{aligned}$$
$$又因为log(s_{i+1}=o_q | s_i = o_p) = transition[p][q]$$
$$\sum_{i=1}^{T-1} \sum_{p=1}^{N} \sum_{q=1}^{N} (log(s_{i+1}=o_q | s_i = o_p) \sum_{s_{i+1}=o_q , s_i = o_p} P(s_1, s_2, ..., s_T| t_1, t_2, ..., t_T)) = \sum_{i=1}^{T-1} \sum_{p=1}^{N} \sum_{q=1}^{N} transition[p][q] * 2^{alpha[i, p] + beta[i+1, q] + transition[p, q] + output[q, t_{n+1}] - normalisation}$$
综上所述，标签序列的熵 = 初始状态生成的熵 + 状态生成标签的熵 + 状态转移的熵：
$$entropy = normalisation - \sum_{i=1}^{N} prior[i] * 2^{alpha[1][i] + alpha[1][i] - normalisation} - \sum_{i=1}^{T} \sum_{j=1}^{N} output[o_j][t_i] 2^{alpha[i,j]+beta[i,j]-normalisation} - \sum_{i=1}^{T-1} \sum_{p=1}^{N} \sum_{q=1}^{N} transition[p][q] * 2^{alpha[i, p] + beta[i+1, q] + transition[p][q] + output[q][t_{n+1}] - normalisation}$$

```python
    def entropy(self, unlabeled_sequence):
        unlabeled_sequence = self._transform(unlabeled_sequence)

        T = len(unlabeled_sequence)
        N = len(self._states)

        alpha = self._forward_probability(unlabeled_sequence)
        beta = self._backward_probability(unlabeled_sequence)
        normalisation = logsumexp2(alpha[T - 1]) # 标签序列的概率

        entropy = normalisation

        # 计算初始状态生成的熵
        for i, state in enumerate(self._states):
            p = 2 ** (alpha[0, i] + beta[0, i] - normalisation)
            entropy -= p * self._priors.logprob(state)

        # 计算状态转移的熵
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

        # 计算状态生成标签的熵
        for t in range(T):
            for i, state in enumerate(self._states):
                p = 2 ** (alpha[t, i] + beta[t, i] - normalisation)
                entropy -= p * self._outputs[state].logprob(
                    unlabeled_sequence[t][_TEXT]
                )

        return entropy
```
NLTK在标签序列的熵的基础上进行扩展，推导过程即将上述熵进行简单扩展，不再详细推导
+ point_entropy代表标签序列上每个位置的标签的熵（根据每个位置的标签可能的状态的概率计算信息熵）
+ exhaustive_entropy通过计算标签序列生成每个可能状态序列的概率来计算标签序列的熵
+ exhaustive_entropy通过计算标签序列生成每个可能状态序列的概率来计算标签序列中每个位置的标签的熵

#### **学习问题**：
在模型参数未知的情况下，根据标签序列（可能带有状态序列，也可能没有）推断模型参数。  

有两种可能的场景：
+ 监督学习的场景，已知诸多标签序列和对应的状态序列，推断模型参数，学习方法相对简单。
+ 非监督学习的场景，只知道诸多标签序列，推断模型参数，一般使用EM期望最大化方法进行迭代求解。

```python
class HiddenMarkovModelTrainer:
    """
    训练器的初始化只需要可能的状态序列和标签序列
    """
    def __init__(self, states=None, symbols=None):
        self._states = states if states else []
        self._symbols = symbols if symbols else []

    def train(self, labeled_sequences=None, unlabeled_sequences=None, **kwargs):
        """
        根据标签序列是否带有状态序列选择有选择还是无选择洗脸
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
```
**有监督学习**  
对于有监督学习，本质上是根据传入的带有状态的标签序列计算频率分布
+ 统计初始状态的频率分布
+ 统计状态转移的频率分布
+ 统计状态生成标签的频率分布

之后，根据上述计算得到频率分布数据计算概率，HMM使用某种estimator进行概率计算   

先生成HMMTrainer再进行训练的情况下默认使用MLE（最大似然估计），用频率分布中的频率近似估计概率

使用HMMTagger的静态方法训练得到模型的情况下，默认使用LidStone平滑法则进行平滑，在频率分布的基础上对分子加上一个系数$\gamma$，对分母加上标签表或者状态表的大小与$\gamma$的乘积，计算方式如下：
$$从状态o_1转移到o_2的概率P(o_2|o_1) = \frac{C(o_2 | o_1) + \gamma}{C(o_1) + |V| * \gamma}  \text{其中|V|为状态表的大小} $$

```python
    def train_supervised(self, labelled_sequences, estimator=None):
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

**无监督学习**：  
在标签序列已知，无对应的状态序列，需要自己估计状态和各种概率矩阵  

在NLTK的实现方式下，HMM模型无监督学习的学习算法为Baum-Welch算法，是对EM算法的一个扩充，接下来介绍NLTK对于Baum-Welch算法的实现

Baum-Welch算法的基本步骤如下：
```
注意，为了推导方便，这里和上面的推导使用了不同的符号和标注，本质上是一样的只是符号不同罢了
```
标签序列记为$O = (o_1, o_2, ..., o_T)$，状态序列记为$I = (i_1, i_2, ..., i_T)$，模型参数为$\lambda$，完全数据的对数似然函数为$log(P(O, I | \lambda))$，状态集为${q_1, q_2, ..., q_N}$，标签集为$v_1, ..., v_k$
$\pi_{i}$为初始状态为$q_i$的概率，$b_{i}(j)$为根据$q_i$生成$v_j$的概率，对应transition矩阵的指数，$a_i(j)$为$q_i$转移到$q_j$的概率，对应于output矩阵的指数

EM算法中的E步，求Q函数，其中$]lambda$是当前参数估计值，$\hat{\lambda}$是要最大化的模型参数
$$Q(\lambda, \hat{\lambda}) = \sum_I log(P(O, I | \lambda))P(O, I | \hat{\lambda})$$

需要注意$P(O, I | \hat{\lambda})$是确定值，根据$\hat{\lambda}$进行估计 

$$P(O, I | \lambda) = \pi_{i_1} b_{i_1}(o_1) a_{i_1}(i_2) b_{i_2}(o_2)...a_{i_{T-1}}(i_T) b_{i_T}(o_T) $$
因此函数$Q(\lambda, \hat{\lambda})$可以写成：
$$Q(\lambda, \hat{\lambda}) = \sum_I log(\pi_{i_1})P(O, I | \hat{\lambda}) + \sum_I (\sum_{t=1}^{T-1}log(a_{i_t}(i_{t+1}))P(O, I | \hat{\lambda}) + \sum_I (\sum_{t=1}^{T} log(b_{i_t}(o_{t}))P(O, I | \hat{\lambda})$$

(1) 上式中第一项可以写成：
$$\sum_I log(\pi_{i_1})P(O, I | \hat{\lambda}) = \sum_{i=1}^{N} log(\pi_{i_1})P(O, i_1 = i | \hat{\lambda})$$
由于$\pi_i$满足约束条件$\sum_{i=1}^N \pi_i = 1$，利用拉格朗日乘子法，写出拉格朗日函数：
$$\sum_{i=1}^{N} log(\pi_{i_1})P(O, i_1 = i | \hat{\lambda}) + \gamma (\sum_{i=1}^N \pi_{i} - 1)$$
求$\pi_i$偏导数并令其为0，可得
$$P(O, i_1 = i | \hat{\lambda}) + \gamma \pi_i = 0$$
对i求和可得
$$\gamma = -P(O | \hat{\lambda})$$
所以可得
$$\pi_i = \frac{P(O, i_1 = i | \hat{\lambda})}{P(O | \hat{\lambda})}$$
(2) 上式中的第二项可以写成
$$\sum_I (\sum_{t=1}^{T-1}log(a_{i_t}(i_{t+1}))P(O, I | \hat{\lambda}) = \sum_{i=1}^{N} \sum_{j=1}^{N} \sum_{t=1}^{T-1} log(a_i(j))P(O, I | \hat{\lambda})$$
用第一项类似的方法根据$\sum_{j=1}^N a_i(j) = 1$的约束，使用拉格朗日乘子法可以求出
$$a_i(j) = \frac{\sum_{t=1}^{T-1} P(O, i_t=i, i_{t+1}=j | \hat{\lambda})}{\sum_{t=1}^{T-1} P(O, i_t = i | \hat{\lambda})}$$
(3) 上式中第三项可以写成：
$$\sum_I (\sum_{t=1}^{T} log(b_{i_t}(o_{t}))P(O, I | \hat{\lambda}) = \sum_{j=1}^{N} \sum_{t=1}^{T} log(b_{j}(o_{t}))P(O, i_t=j | \hat{\lambda})$$
根据$\sum_{k=1}^M b_j(k) = 1$的约束，使用拉格朗日乘子法。注意只有$o_t = v_k$时$b_j(o_t)$对$b_j(k)$的偏导数不为0，这里用$I(o_t = v_k)$表示:  
$$b_j(k) = \frac{\sum_{t=1}^{T} P(O, i_t=j | \hat{\lambda}) I(o_t = v_k)}{\sum_{t=1}^{T} P(O, i_t = j | \hat{\lambda})}$$

这些概率的计算可以通过之前陈述的前向、后向算法和标签序列熵推导用到的部分公式进行计算
$$alpha[t, i] + beta[t, i] = log(P(O, i_t = q_i | \lambda)) \tag{1}$$
$$alpha[t, i] + beta[t+1, j] + transition[i, j] + output[j][o_{t+1}] = log(P(O, i_t=q_i, i_{t+1} = q_j) \tag{2}$$
根据上面两个公式可以得出:
$$\begin{aligned}
\gamma_{t}(i) &= P(i_t = q_i |O; \lambda) \\
&= \frac{P(i_t = q_i, O; \lambda)}{P(O; \lambda)} \\
&= \frac{2^{alpha[t, i] + beta[t, i]}}{\sum_{j=1}^{N} 2^{alpha[t, j] + beta[t, j]}}
\end{aligned}$$
$$\begin{aligned}
\xi_{t}(i, j) &= P(i_t = q_i, i_{t+1} = q_j |O; \lambda) \\
&= \frac{P(i_t = q_i, i_{t+1} = q_j, O; \lambda)}{P(O; \lambda)} \\
&= \frac{P(i_t = q_i, i_{t+1} = q_j, O; \lambda)}{\sum_{i=1}^{N} \sum_{j=1}^{N} P(i_t = q_i, i_{t+1} = q_j, O; \lambda)}\\
&= \frac{2^{alpha[t, i] + transition[i, j] + beta[t + 1, j] + output[j, o_{t+1}]}}{\sum_{i=1}^{N} \sum_{j=1}^{N} 2^{alpha[t, i] + transition[i, j] + beta[t + 1, j] + output[j, o_{t+1}]}}
\end{aligned}$$
综上所述，概率矩阵的更新公式为：
$$\pi_i = \gamma_1{i}$$
$$a_i(j) = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T} \gamma_t(i)}$$
$$b_j(k) = \frac{\sum_{t=1}^{T} \gamma_t(j) I(o_t = v_k)}{\sum_{t=1}^{T} \gamma_t(j)}$$

NLTK的实现基于以上公式，使用numpy中的ndarray张量形式进行向量和矩阵运算  

在Baum-Welch的实现中，一次迭代中遍历训练集中所有序列，每个序列会使用Baum-Welch算法对于前一次迭代最终的transition和output矩阵进行更新，根据每个序列得到的更新后的概率矩阵会存储起来，然后进行标准化(根据这个序列的熵进行加权)，产生此次迭代最终更新的概率矩阵。
```python
    def _baum_welch_step(self, sequence, model, symbol_to_number):

        N = len(model._states)
        M = len(model._symbols)
        T = len(sequence)

        # 根据标签序列使用前向算法和后向算法计算alpha和beta概率矩阵
        alpha = model._forward_probability(sequence)
        beta = model._backward_probability(sequence)

        # 计算的到标签序列的生成概率
        lpk = logsumexp2(alpha[T - 1])

        A_numer = _ninf_array((N, N)) 
        B_numer = _ninf_array((N, M))
        A_denom = _ninf_array(N)
        B_denom = _ninf_array(N)

        transitions_logprob = model._transitions_matrix().T

        for t in range(T):
            symbol = sequence[t][_TEXT]  
            next_symbol = None
            if t < T - 1:
                next_symbol = sequence[t + 1][_TEXT]  
            xi = symbol_to_number[symbol]

            # 每个状态生成下一个标签的概率向量
            next_outputs_logprob = model._outputs_vector(next_symbol) 
            # 由之前的推导，alpha[t] + beta[t] 为该标签序列中t位置对应的状态概率向量
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
        # 如果给了部分训练的模型，直接读取各个概率矩阵
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

        # 读取状态集和标签集
        self._states = model._states
        self._symbols = model._symbols

        N = len(self._states)
        M = len(self._symbols)
        symbol_numbers = {sym: i for i, sym in enumerate(self._symbols)}

        # 根据状态集和标签集更新HMM模型的状态转移概率矩阵和状态生成标签概率矩阵
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

        converged = False
        last_logprob = None
        iteration = 0
        max_iterations = kwargs.get("max_iterations", 1000)
        epsilon = kwargs.get("convergence_logprob", 1e-6)

        # 每个迭代使用Baum-Welch算法更新状态转移概率矩阵和（可能）状态生成标签概率矩阵
        # 持续迭代直到收敛或者达到最大迭代次数
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
                # 调用Baum-Welch算法得到每个序列
                (
                    lpk,
                    seq_A_numer,
                    seq_A_denom,
                    seq_B_numer,
                    seq_B_denom,
                ) = self._baum_welch_step(sequence, model, symbol_numbers)

                # 累加更新的概率矩阵
                for i in range(N):
                    A_numer[i] = np.logaddexp2(A_numer[i], seq_A_numer[i] - lpk)
                    B_numer[i] = np.logaddexp2(B_numer[i], seq_B_numer[i] - lpk)
                # 累加标准化的权重参数
                A_denom = np.logaddexp2(A_denom, seq_A_denom - lpk)
                B_denom = np.logaddexp2(B_denom, seq_B_denom - lpk)

                logprob += lpk

            # probability values
            for i in range(N):
                logprob_Ai = A_numer[i] - A_denom[i]
                logprob_Bi = B_numer[i] - B_denom[i]

                # 将所有序列更新的概率矩阵加权平均产生新的概率矩阵
                logprob_Ai -= logsumexp2(logprob_Ai)
                logprob_Bi -= logsumexp2(logprob_Bi)

                # 更新transition和output概率矩阵
                si = self._states[i]

                for j in range(N):
                    sj = self._states[j]
                    model._transitions[si].update(sj, logprob_Ai[j])

                if update_outputs:
                    for k in range(M):
                        ok = self._symbols[k]
                        model._outputs[si].update(ok, logprob_Bi[k])

                # 目前的实现中不会更新prior概率矩阵，NLTK的开发者对此有不同的意见

            # 测试是否收敛
            if iteration > 0 and abs(logprob - last_logprob) < epsilon:
                converged = True

            print("iteration", iteration, "logprob", logprob)
            iteration += 1
            last_logprob = logprob

        return model
```


