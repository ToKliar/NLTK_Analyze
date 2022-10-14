# 功能分析

这部分对NLTK库的功能进行分析 

## NLTK的主要模块和功能 

下面两个表展示了NLTK包下的各个子包和子模块的功能和对应的自然语言处理任务 （按照字典序排序）  

对于NLTK包功能的详细分析聚焦于主要的自然语言处理任务 

| NLTK的子包   | 自然语言处理任务 | 功能                                             |
|-----------|----------|------------------------------------------------|
| app       | 应用       | 图形化交互式NLTK工具                                   |
| ccg       | 语义解析     | 组合范畴文法                                         |
| chat      | 应用       | 聊天机器人                                          |
| chunk     | 分块       | 使用正则表达式、n-gram、命名实体等方式进行分块                     |
| classify  | 分类任务     | 对文本进行分类（打标签）                                   |
| cluster   | 聚类任务     | K-means、EM、GAAC方法对文本进行聚类                       |
| corpus    | 访问语料库    | 通过该模块访问NLTK提供的语料库和词典                           |
| draw      | 应用       | 绘制NLTK的解析树                                     |
| inference | 语义解释     | 用于定理证明和模型检验                                    |
| lm        | 语言模型     | 目前只包含n-gram模型                                  |
| metrics   | 指标评估     | 对自然语言处理结果使用召回率等指标进行评估                          |
| parse     | 语法解析     | 使用图表或概率解析的方式生成语法树                              |
| sem       | 语义解释     | 使用一阶逻辑公式表示语义结构，在集合理论模型中进行评估                    |
| sentiment | 情感分析     | 使用NLTK内置特征和分类器对文本进行情感分析                        |
| stem      | 词干提取     | 抽取词的词干或词根形式                                    |
| tag       | 词性标注     | 使用n-gram，backoff，HMM等算法进行词性标注                  |
| tbl       | 机器学习     | 基于转换的机器学习(Transformation Based)，被BrillTagger使用 |
| test      | 测试       | NLTK内模块的单元测试                                   |
| tokenize  | 词元化      | 将文本按照不同粒度划分                                    |
| translate | 机器翻译     | 用于机器翻译任务的特征                                    |
| tree      | 文本结构     | 用于表示文本的语言结构，如语法树                               |
| twitter   | 应用       | 使用Twitter API检索Tweet文档                         |


| NLTK的子模块          | 功能                                  |
|-------------------|-------------------------------------|
| book              | NLTK的指导书                            |
| cli               | NLTK的命令行工具                          |
| collections       | NLTK实现的collection                   |
| collocations      | 识别词组的工具                             |
| compat            | 版本兼容                                |
| data              | 查找和加载NLTK提供的资源                      |
| decorators        | NLTK内置的decorator                    |
| downloader        | NLTK语料库和模块下载器                       |
| featstruct        | 文本特征结构的基本数据类和对特征结构进行基本操作的类          |
| grammar           | 表示上下文无关语法的基本数据类。                    |
| help              | NLTK使用帮助                            |
| internals         | NLTK内置的与java、字符串交互的工具               |
| jsontags          | 给数据打上JSON标记，便于downloader查找文件        |
| lazyimport        | 延迟模块导入，加速启动时间                       |
| probability       | 表示和处理概率信息的类                         |
| text              | 各种用于文本分析的功能：索引、正则表达式搜索等             |
| tgrep             | 用Tgrep算法搜索NLTK中的树结构                 |
| toolbox           | 用于读取、写入和操作工具箱数据库和设置文件，处理SIL工具箱格式的数据 |
| treeprettyprinter | 对于NLTK中树结构的漂亮的绘制                    |
| treetransforms    | 解析自然语言中的语法转换的方法集合                   |
| util              | 一些实用方法                              |
| wsd               | 为上下文中的歧义单词返回一个同义词集                  |



