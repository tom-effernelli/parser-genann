## ParserGenann

This repo is implementing a parser Model using Genann Library.

## How it works

It is intended to detect the name of the user and his/her legal problem. For example, a prompt could be "My name is Sarah Miller and I want to report workplace harassment.".\
It uses glove.6B.50d.txt file as a word embedding file, then uses tokens.txt and labels.txt as the training dataset. It finally stores the model in model.ann.\

## How it is used

Just download the repo, compile the parsingGenann.c with gcc and launch it with argv[1]=bool_is_model_training (set at true if the model is intended to be trained on the dataset, false if you want to use the model), argv[2]=prompt (string format).\

## What it sends back

It sends back a table with the labels associated to each token of the prompt (here : each words) with the following matching pattern :
Labels matching :
 - 0 : not relevant token
 - 1 : B-PERSON (the person's name beginning)
 - 2 : I-PERSON (inside of the person's name)
 - 3 : B-LEGAL_ISSUE (the legal problem's beginning)
 - 4 : I-LEGAL_ISSUE (inside of the legal problem)
