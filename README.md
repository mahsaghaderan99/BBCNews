# BBC News Analysis
An NLP-based project to analyse BBC  news articles. 
Each section describes briefly in the following sections. The final section includes a link containing the complete document.

## Word2VEC
Word2Vec models with variouse number of parameters(5000, 10000, 15000,..., 35000) are trained on each class of the dataset. In this section Headline is concatenated with Body of each news as input text of the model. Each models trains through 40000 iterations. Trained models are saved in [models/word2vec](models/word2vec) directory. 30 most repeated words distribution on  a 2D map is illustrated on the following Image.
| <img src="reports/word2vec/word_vectors_هنر.png" alt="ART distribution" width="400"/> | 
|:--:| 
| *30 most repeated word distribution of ART class* |

Analysis of ART class distribution and other classes are discussed in prject [documentation](documents/main_v2.pdf).

This part generally is based on [assignment2_cs224n_2021 Stanford NLP course](http://web.stanford.edu/class/cs224n/assignments/a2.zip). 

## Tokenization
In this part 2 different methods of tokenization is implemented based on SentencePiece packet. This packet is an unsupervised text tokenizer and detokenizer python project, which is mainly used for text generation tasks.It uses a model to train what tokens are best to tokenize a given corpus. 

The first method is tokenizing words and subwords. In this method, the model looks for most repeated words and sub-words and even letters. As the result, we can see almost every word in the final version of chosen tokens. 

On the contrary, the second method is just tokenizing words, this model looks for the most repeated words.

## Language Model 
In this section, the corpus is divided into three parts, train set, dev set and test set. LanguageModel is trained on each label separately to evaluate the capability of each model to generate specific news. Models are saved in the”models/lm” directory. Language models' artichecture are adopted from pytorch/examples project.

Models are trained on GTX1070 GPU which training process is 10 times faster than a quad-core CPU.

### Models:
- **LSTM:** The best perplexity record is 47 on the ”Science” label. In this trial, there are many tokens in output-generated text.
- **GRU:**   LSTM and GRU have similar learning power as evidence perplexity of GRU model on the test set are 43, when the label is ”Science”. As expected GRU model training time is less than LSTM model. In this trial, again there are many tokens in output-generated text.
- **Transformer:** This model constructs an Encoder̲ and Decoder̲.
  - The encoder consists of an Embedding layer.
  - Decoder consists Linear layer. Test perplexity is 37.41 when the label is ”Science” which is 3 scores better than previous models. The number of tokens is decreased and there more meaningful sentences are generated.

In conclusion, LSTM and GRU both have almost similar learning power, but it is possible to access better results with transformer architecture.

## Finetuning LM 
In this part, ParsBert V3 is used as a pre-trained Bert model. Due to hardware limitations, the train batch size is set to 4 and the evaluation batch size is set to 8. In the first experiment, all samples in the dataset are used to fine-tune the BERT model. As a result, the BERT model is fine-tuned on the news dataset. The next step is to extract one layer before the last one as an embedding (a representation for each word) and replace the task 4 embedding part with this model.

## Dataset
Persian BBC News website is crawled on May/16/2021. Generated CSV file is located in [data](data) directory. Also other versions of dataset exits there. In [data/splited](data/splited) directory, preprocesed samples are classified by their news label and in each label train, dev and test data is splited. 

## More Details About This Project

Different version of fulldocuments are available [here](documents).

[Prpject Document](https://docs.google.com/document/d/1PBN1QmrI4QIE2bqm3R3kIKlj2fsblJLaVOc6nzjqGDM/edit?usp=sharing)
