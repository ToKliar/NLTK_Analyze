# **代码整体分析**

这部分分析NLTK库的程序构造原理、类库结构和高效能编程实践

## **代码架构**

NLTK的各个功能实现在各个包里，大部分包（chunk、tag、tokenize、classify等等）集成了解决对应自然语言处理任务的多种机器学习算法的实现。   

NLTK的实现上，每个功能包会暴露一个默认的功能方法，因此可以通过这些方法直接调用默认的模型执行某些自然语言处理任务，如tag包就会将pos_tag方法和pos_tag_sents方法暴露出去，这两个方法使用感知器作为默认Tagger进行词性标注，可以在主包中直接调用两个方法进行词性标注。
```python
import NLTK
NLTK.pos_tag(tokens)
NLTK.pos_tag_sents(sents)
```

这些包整体上按照接口模式进行架构，每一个包对应一种自然语言处理任务：
+ api.py文件中定义了接口，包含特定自然语言任务需要的方法和常量。通过接口，使用者可以用统一的方式使用统一的方法完成相应的任务。包中的所有模型都应该实现接口中定义的所有方法。
+ util.py文件中定义了多个模型需要的通用的工具方法。
+ 其余每个文件对应了一种特定自然语言处理任务的算法的实现，每个模型都需要基于接口定义的方法来实现对应的算法。文件的内容包含两个部分：第一部分是算法实现有关的类，第二部分是类需要的工具方法和使用示例方法。

### **架构示例**

下面以tag包为例，展示这种架构

tag包中的api.py文件定义了TaggerI接口，其中注解为@abstractmethod是需要模型实现的抽象方法，其他没有注解的方法则是基于抽象方法实现的功能方法。

python语言没有接口相关的定义，python通过抽象类实现Interface，通过抽象方法定义接口的派生类需要实现的方法。

#### **接口定义**

TaggerI接口可以看成是对整个Tag任务的功能抽象，可以对Tag任务进行进一步抽象，如将基于特征集进行Tag任务的模型进行抽象，得到FeaturesetTaggerI接口，这里需要注意FeaturesetTaggerI是基于TaggerI定义的，是TaggerI的子类。

TaggerI接口中包含了Tag任务中需要的所有的功能。

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

## **高性能实践**

### **懒惰机制**

为了减少运行时间，NLTK中大量使用了懒惰机制，即将部分工作推迟到不得不进行该工作的时候进行。这样可以减少代码的运行时间和存储消耗，提高性能。

#### **LazyImport** 

为了加速代码的启动时间，NLTK使用LazyImport的方式，直到代码从模块的命名空间中请求某个属性（对象或方法）时，这一属性才会实际导入到项目中。

LazyImport的实现在``lazyimport.py``文件中

主要两个方法如下，无论是获取属性还是修改属性，都会在实际请求属性（即__getattr__和__setattr__调用时）import对应的模块属性

```python
def __getattr__(self, name):

        """Import the module on demand and get the attribute."""
        if self.__lazymodule_loaded:
            raise AttributeError(name)
        if _debug:
            print(
                "LazyModule: "
                "Module load triggered by attribute %r read access" % name
            )
        module = self.__lazymodule_import()
        return getattr(module, name)

    def __setattr__(self, name, value):

        """Import the module on demand and set the attribute."""
        if not self.__lazymodule_init:
            self.__dict__[name] = value
            return
        if self.__lazymodule_loaded:
            self.__lazymodule_locals[self.__lazymodule_name] = value
            self.__dict__[name] = value
            return
        if _debug:
            print(
                "LazyModule: "
                "Module load triggered by attribute %r write access" % name
            )
        module = self.__lazymodule_import()
        setattr(module, name, value)
```

#### **Lazy Collection**

为了提高Collection集合工具的性能，NLTK通过懒惰机制，重写了主要的几个Collection，LazyCollection的特点在于根据需要计算collection中的值，主要用于对语料库（corpus）的访问，只需要从磁盘文件中需要的语料库部分内容，而不是将整个语料库加载到内存中。

这些LazyCollection都基于虚类``AbstractLazySequence``进行实现，包括
+ ``LazySubsequence``
+ ``LazyConcatenation``
+ ``LazyMap``
+ ``LazyZip``
+ ``LazyEnumerate``
+ ``LazyIteratorList``

``LazyMap``可以看成是一个序列和对某个序列在某个函数的上映射，在``LazyMap``的实现中只有在访问到对应的值时才会使用函数对这个值进行映射，未访问情况下只会存储原始序列。而python内置的Map在初始化时就会计算序列中所有值的映射值，占据更多的内存。  
如对corpus中的tokens进行分类，对应的featureset可能会占用大量内存，所以可以在训练和使用分类器时使用``LazyMap``来减小内存的占用量。
```python
class LazyMap(AbstractLazySequence):
    def iterate_from(self, index):
        # Special case: one lazy sublist
        if len(self._lists) == 1 and self._all_lazy:
            for value in self._lists[0].iterate_from(index):
                yield self._func(value)
            return

        # Special case: one non-lazy sublist
        elif len(self._lists) == 1:
            while True:
                try:
                    yield self._func(self._lists[0][index])
                except IndexError:
                    return
                index += 1

        # Special case: n lazy sublists
        elif self._all_lazy:
            iterators = [lst.iterate_from(index) for lst in self._lists]
            while True:
                elements = []
                for iterator in iterators:
                    try:
                        elements.append(next(iterator))
                    except:  # FIXME: What is this except really catching? StopIteration?
                        elements.append(None)
                if elements == [None] * len(self._lists):
                    return
                yield self._func(*elements)
                index += 1

        # general case
        else:
            while True:
                try:
                    elements = [lst[index] for lst in self._lists]
                except IndexError:
                    elements = [None] * len(self._lists)
                    for i, lst in enumerate(self._lists):
                        try:
                            elements[i] = lst[index]
                        except IndexError:
                            pass
                    if elements == [None] * len(self._lists):
                        return
                yield self._func(*elements) # 在遍历的时候才会调用func
                index += 1
```

#### **Lazy DataLoader**

此外，NLTK通过``LazyLoader``支持对于数据集访问的懒惰访问，即只有需要读取对应的数据时才会将数据从磁盘Load到内存。  
通过这种方式加载数据一方面减小了内存占用量（可以一次加载少量数据），另外一方面可以加速代码的运行（加载数据和数据处理如模型的训练等并行）

### 自定义数据结构

为了统一数据的管理，NLTK自定义了很多数据结构并对python原生的数据结构进行扩展，规范化数据的处理形式和展示形式，举例如下：
+ 概率分布、频率分布相关的数据结构定义在``probability.py``文件中，数据结构有：``FreqDist``和基于``ProbDistI``扩展的各种概率分布数据结构如``MLEProbDist``、``LidStoneProbDist``等等
+ 与特征有关的数据结构定义在``featstruct.py``中，主要是``FeatStruct``、``Feature``、``CustomFeatureValue``这三个基本的数据结构与其扩展的数据结构
+ ``decorators.py``中扩展了python原生的装饰器

通过这种方式，NLTK将自然语言处理相关的数据结构规范化，可以通过NLTK的数据结构和模型完成从文本读取到自然语言处理任务执行整个流程而无需引入额外的包。大大减小了对外部库的依赖性和用户的学习成本，提高了使用上的便利性。