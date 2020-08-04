# 项目

基于nltk+深度神经网络实现电影台词情感分析

# 说明

## 环境配置

```html
电脑环境：windows10

python3.6.5

第三方库：requirements.txt
```

## 数据集

数据集来源于电影中的台词文本。文件`positive.txt`, `negative.txt`分别存储有5331条正面情感的台词文本数据，331条负面情感的台词文本数据。 

> 提示：项目中**已经包含数据集**，不需要你再下载！

## 仓库

本仓库包括以下：

- `requirements.txt`：第三方库；
- `negative.txt`：负面情感文本数据；
- `positive.txt`：正面情感文本数据；
- `SentimentNeuralNetwork.py`：基于nltk+深度神经网络实现电影台词情感分析主程序；

# 使用

请先配置`python`运行环境：
```html
pip install -r requirements.txt
```

安装之后运行出现错误：

```html
LookupError:
**********************************************************************
  Resource [93mwordnet[0m not found.
  Please use the NLTK Downloader to obtain the resource:

  [31m>>> import nltk
  >>> nltk.download('wordnet')
  [0m
  For more information see: https://www.nltk.org/data.html

  Attempted to load [93mcorpora/wordnet[0m

  Searched in:
    - 'C:\\Users\\Userwzz/nltk_data'
    - 'E:\\python365\\nltk_data'
    - 'E:\\python365\\share\\nltk_data'
    - 'E:\\python365\\lib\\nltk_data'
    - 'C:\\Users\\Userwzz\\AppData\\Roaming\\nltk_data'
    - 'C:\\nltk_data'
    - 'D:\\nltk_data'
    - 'E:\\nltk_data'
**********************************************************************
```

解决方法：
先去[nltk_data](https://github.com/nltk/nltk_data) 下载zip包，然后将缺少的包（错误信息中有提示：`>>> nltk.download('wordnet')`），将缺少的包所在的文件夹复制到**Searched in**的任意路径下的`nltk_data`文件夹下，比如我的放置位置为：

```html
C:\\nltk_data\\
├── collections
├── packages
├── tools
├── index.xml
├── index.xsl
├── Makefile
├── corpora
|   └── wordnet
└── README.txt
```

运行程序：
```html
python SentimentNeuralNetwork.py
```







