# NLTK_Analyze
对自然语言处理库：NLTK功能、源代码和相关机器学习理论的分析

## 任务要求 

本仓库对于NLTK的分析基于翻译技术与实践课程中的作业，作业的要求如下 

基础任务:
1. 知道NLTK是啥，能做什么 
2. 了解所有可能的用途，跑通部分测试用例 

进阶任务：
1. 阅读代码，了解程序构造原理，类库结构
2. 分析背后的机器学习理论、方法和高效能编程最佳实践

## NLTK 介绍 

NLTK，全称Natural Language Toolkit，自然语言处理工具包，由宾夕法尼亚大学的Steven Bird和Edward Loper开发的模块。  
NLTK是构建Python程序以使用人类语言数据的领先平台。它为50多种语料库和词汇资源（如WordNet）提供了易于使用的界面，还提供了一套用于分类，标记化，词干化，标记，解析和语义推理的文本处理库。
NLTK是Python上著名的⾃然语⾔处理库 ⾃带语料库，具有词性分类库 ⾃带分类，分词，等等功能。NLTK被称为“使用Python进行教学和计算语言学工作的绝佳工具”，以及“用自然语言进行游戏的神奇图书馆”。

## NLTK 安装方法
安装方法：
```shell
pip install --user -U nltk
```
进入python后，使用nltk.download()根据弹出的界面自动下载需要的数据和package
```python
import nltk
nltk.download()
```