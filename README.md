# Classification-Airline-Reviews
In this project, I worked on a [dataset](https://www.kaggle.com/datasets/chaudharyanshul/airline-reviews) from kaggle about britich airline customer reviews. I first did some visualisations to discover and to see the relations between variables. Then, I started working on the main task that interests me which is Sequence Classification. 

In this dataset, there is a column called *ReviewBody* that contains as sounds the body review of the customer on a specific travel, thus aircraft. This, is the sequence that I want to classify. The second column which I considered as the target variable is *OverallRating*. This rating is a score out of 10, which makes the task a bit harder, especially with the small number of lines contained in the dataset (3700 datarows). 

The model I choosed to use is **BertForSequenceClassification** from Hugging Face 🤗 Transformers library. The model at the first starts to overfit easily as the validation loss was increasing while the accuracy was decreasing. It means that the model becomes too specialized in my training data and is unable to generalize well to validation data. I've tried different learning rates as well as different weight decays in my Trainer object to address this problem, but nothing has changed. Therefore, I saw that the best think to do is augmenting the data first. 

I tried doing so with **nlpaug** library using the **SynonymAug** method of the **word** class, and I choosed **Wordnet** English lexical database as argument of the src parameter. The results were not so impressives, and the loaded synonyms were not so accurates and precises. Therefore, I looked for another method to solve this issue, and I decided finally to apply MLM (Masked Language Modeling) to perform this task. First, I used the task *fill-mask* with **Roberta** as a pre-trained model. I noted that it doesn't fill all masked tokens in one time, but it did it separatly by giving for each token 5 different predictions ranked by score from best to worst. So, it's obligatory to create a function that automate this replacement task.

 I coded for that two functions, the first one mask randomly a number of tokens where you can specify the minimum number of tokens that you want to be masked. An other parameter exists in this function which is boolean, it specifies whether you want to mask stop words also or ignore them. The second function involves a loop in which each mask token is replaced randomly with one of the five tokens that have been predicted for it. When I run the second function on all the sentences in one time, it produces an error related to the input length, because there are some sentences that have more than 512 words. So, to get rid of this problem, I decided to choose the customizable way which define separatly the tokenizer with all of its parameters and generate the predicted output directly by decoding the mask tokens logits.

 When comparing this two functions on sentences that have the same length, the difference in time was noticeable:

- Sentence with 34 words: 0.02s for the first function and 0.02s for the second one. 
- Sentence with 308 words: 0.25s for the first function (I've tried to test it on more longuer sentences but it didn't work) and 0.16s for the second one.
- Sentence with 654 words (the longuest one in the dataset): 0.22s for the second function.

I run this process on every sentence of the dataset, so, the number of lines has been duplicated. And, when I retrained the model on this augmented version, the difference was impressive. The accuracy increased from **34%** to **85%**.

 The project is not finished yet, and I still want a more higher accuracy.
 If you have some instructive ideas that can help achieve that, don't hesitate to text me, It would be great to discuss them with you! 