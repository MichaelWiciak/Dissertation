# Dissertation

Abstract: This study explored the performance of different Transformer Architectures on code completion using the method of Masked Language Modelling
(MLM) within the context of domain-specific fine-tuning. Code completion, a pivotal feature in enhancing developer productivity, 
involves predicting and suggesting code snippets based on the existing context, thus reducing manual typing efforts. The study 
scrutinises various models' performance dynamics in this domain, with a particular emphasis on Transformers, which are renowned 
for their prowess in natural language processing tasks. Notably, the study highlights the performance of pre-trained models 
fine-tuned for MLM objectives, where models like CodeBERT-base exhibit superior accuracy in code completion tasks across diverse 
programming languages, showcasing the efficacy of supervised learning approaches.
\linebreak
\linebreak 
The findings also shed light on the nuanced performance nuances across different models. While CodeComments, leveraging CodeBERT-base, emerges as a standout performer in overall accuracy, ASTComments, powered by UniXCoder-base, showcases commendable speed in token generation, particularly excelling in C++ contexts. Conversely, the study reveals the limitations of simpler models like the RoBERTa-based Code model, which consistently lags in performance metrics. Thus, the paper underscores the critical importance of selecting models tailored to specific project requirements, advocating for domain-specific fine-tuning to address the unique intricacies and challenges posed by individual software projects.
