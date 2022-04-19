# NLUProject
In this paper, we will explore bias mitigation in pre-trained masked language models and how it suffers from biases that existed during the pre-training. Our project will aim to remove toxicity induced bias in a dataset by a pre-processing pipeline while fine-tuning for specific tasks. We will focus our efforts by experimenting with gender bias, but this pipeline can serve as a prototype for more complex studies such as racial bias.

Using the RobBERTa based transformer, we will first learn how to classify toxicity and then identify toxic spans in a given corpus by training a transformer, generic enough to detoxify given text for a  downstream tasks.  Once identified, we will attempt to remove toxicity, in order to train a fairer model during subsequent fine-tuning, by masking these elements with a common token.

## Data:
1. ToxicSpanDetectionData: Dataset consisting of toxic span indexes over tweets data, used in the paper Chhablani et al., 2021.
2. ToxicTextClassifierData: Dataset multi-class toxicity dataset. https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
3. BiasPredictionScores : This has the output predictions from the external evaluations run over raw unmasked datapoints and the masked outputs of the transformer. We use this in building the evaluation metrics for the transformers performance.
4. GenderWordData: 
5. SentimentAnalysisData:

## Transformer: 
1. Training Transformer: The POC of training procedures are stored in python notebooks independently executable. Were eventually run on HPC for training.
2. Models: The saved models serialized files stored here which are then loaded into the analysis notebooks to generate outputs of the transformer.
3. Analysis: Read the model files and run the flow of execution for the detoxification and masking of toxic elements. The evaluation of the external model analysis is also present here where the raw input and the masked output of the transformer is passed through an external evaluation metrics to generate the toxicity prediction outputs and generating the evaluation metrics on how much datapoints was the transformer able to detoxify by masking toxic spans.

## Faireness Metrics:
