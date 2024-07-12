# ML

## Intro ML
Hi everyone, and welcome to our introduction to machine learning. This video is the second in our series where we'll be implementing alphafold from scratch. 
In this video, you'll learn how to build a two-layer feed-forward neural network for the classification of handwritten digits ground-up. We'll use nothing but the basic tensor operations we've learned in the last section.

There are already a bunch of great videos out there that teach you basically the same thing. I particularly enjoyed the videos from 3blue1brown and Sebastian Lague, which I'll link in the description. Like all of them, we'll try to be concise while still giving you the full theoretical background and appreciating the beauty of the algorithm. There are three points that make this video a little different and potentially more suitable for you: First, we'll leverage our knowledge on tensors from the last video. This means that we don't have to build the theory up using single numbers and neurons; instead, we can directly use matrices and vectors. This makes the formulation a lot cleaner and closer to how it's done in reality. Second, as with all videos in this series, we'll provide you with a Jupyter Notebook where you can do the whole implementation by yourself, ensuring you truly understand it. And, third, it just fits a little more snugly into the rest of the series, aligning with the methods we use and preparing you for the next steps in completing AlphaFold.

So, with that out of the way, let's dive right in!



## Non-Linearities
But our ideas weren’t stupid. All the benefits of hierarchical features we came up with are real, and the collapse of the two layers is just a technical issue. A simple linear weighing of the features simply can’t keep the information we gained (which we see in the way the two matrices collapsed together). Luckily, there is an easy fix. It is to introduce any reasonable, non-linear function in between. 


## Going to ML
What comes next is the magic of machine learning. We've talked a lot about what good values for the weight matrices might be. Truth is, we don't need to decide on them. We simply choose them as the values that lead to the best result, using an algorithm that's astonishingly simple.

## Intro SGD
So, we found a criterion for model optimization: Minimizing the L2-loss of our model on the training dataset. We could, theoretically, just try out random parameters and select the combination with the lowest loss, but given the number of parameters, that's infeasible. The idea that's actually being used is this: We start with random weights. Then, for every image, we nudge the parameters by a very small amount into the direction that improves the prediction for this specific image. 


## Intro to Derivatives
We'll go through the calculation of the derivatives rather quickly. If you aren't super involved in math, this might be challenging for you, but it's not too important that you understand the individual steps anyway. Calculating the derivatives is something that Pytorch does for you automatically, and we'll just do it by hand this once in the introduction to Machine Learning. It's cool if you can follow the equations (and you might want to watch a slower-paced video for this), but it's more important to understand the principle of gradient descent itself. 

That was a lot of math in very little time. However, you'll see in the tutorial notebook that each step is actually not too difficult to compute and seldom involves writing more than one line of code. To prepare you for doing this yourself, let's go through an example: calculating the downstream gradient in the affine linear layer.

## Outro
In the tutorial, we'll put all of these steps together and implement a ready-to-use machine learning model for handwritten digit recognition. When you're ready, head over to the Notebook and start the actual implementation.

This was a really quick dive into Machine Learning. If you found the material to fast or think that you didn't understand important details, there are great alternative tutorials that will guide you through the topic in a more slow-paced, visually rich fashion. I really enjoyed watching [this](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=K9H1t44OgDMBidQG) series by 3Blue1Brown and [this](https://youtu.be/hfMk-kjRv4c?si=WbyaHb36XXPWXUaO) video by Sebastian Lague. Both are comprehensive, but the first one goes a little deeper into the mathematical details. 

All in all, remember that modern machine learning frameworks protect you from thinking about most of these details, and many guides to machine learning just don't walk you through this part (doing ML by hand) at all. Personally, I think it's really educative, but you don't have to worry about the rest of this series on Alphafold just because you didn't like the vector calculus for the derivatives. So, good luck with the tutorial notebook, and happy coding!

# Attention

## Intro Attention
Hi everyone, and welcome to our introduction to Attention. This is the third video in our series where we'll be implementing Alphafold from scratch.

Attention is a relatively new concept.
In 2017, Google released a paper on its Transformer architecture for natural language processing, giving the paper the name "Attention is All You Need". So far, it has lived up to its reputation. You'll have a hard time finding a modern breakthrough in machine learning that doesn't use Attention as its core mechanism. And, like most clever ideas in machine learning, it is actually a quite simple idea. 

Imagine trying to understand a complex sentence without knowing which words to focus on. This is where the game-changing concept of Attention in machine learning comes in. The attention mechanism addresses the problem of dealing with sequential data – which is most existing data: language is sequences of words, music is sequences of notes, and proteins are sequences of amino acids. The name attention comes from its idea to determine which tokens in the input are the most relevant for the task at hand, that is which tokens the model should pay attention to, before basing its prediction on these attention weights.

## Intro Mechanism
To the rescue comes the attention mechanism. It follows a simple idea: Every token in the input gives a quick summary of its content. The model creates a question and checks it against each of the summaries, to give scores on how much they should contribute to the result. Then, each of the inputs is weighted according to its attention score to compute the result. 

## Dynamic Queries
There are a bunch of small variations to this core mechanism, and we'll look at some of them next. Our first idea is to change up how we create the query vectors. For many tasks, the questions we want to ask are directly linked to the inputs, and static queries can't give this flexibility.

### Optional
Mechanisms like this that we can use to actually interpret how the network made its decisions are one of the key research areas of AI and AI Safety. But while in this example, explaining the attention weights works pretty well, it often falls short for large models. That is because to be able to interpret the reason why the model looked at specific neurons, we'd need to understand what these neurons encoded. And this often proves hard. Anthropic, the maker's of the LLM claude, state it like this:

### Optional
"Unfortunately, the most natural computational unit of the neural network – the neuron itself – turns out not to be a natural unit for human understanding. This is because many neurons are polysemantic: they respond to mixtures of seemingly unrelated inputs."

### Interpretability
For this reason, interpretation of attention in large models often falls short of its expectations. The results of interpretation in the AlphaFold paper mostly found out that in early layers, attention focused on residues that are next to each other in the sequence, and for late layers, attention focused on residues that are next to each other in the protein. Basically, this only restates that the network indeed found out the structure of the protein, but tells us little on the question of how.

## Cut and Motion
But we aren't helpless! The explosion of large language models lead to a huge increase in research on interpretability, driven by the big ethical questions that accompany their massive use. The paper from Anthropic we talked about earlier for example found out a way to actually detangle that polysemanticity and clearly label activation patterns. This process might be transferrable to AlphaFold, possibly making us able to tell in which stages AlphaFold starts coming up with hypotheses on commonly known structure motives. 

This is one of the reasons why I think it's so important to not be too specific when learning AI. Many of the big breakthroughs came from taking a method that's commonly known in a specific AI area, like Natural Language Processing, and taking it to a new problem, like protein folding. It's why we spend so much time talking about language in this chapter, and it's why I highly recommend you to also take a look at Computer Vision at one point or another, for example with the open material from Justin Johnson's course from the University of Michigan.


## Intro Small Problems
At this point, we're finished with our gated, multi-head attention module, which is what's most important for actually implementing AlphaFold. What I want to do next is to look at some problems that emerge from the structure of attention, and how we can solve them. 

The second problem of attention comes from the fact that each query can attend to each key. To see why this is a problem, let's look at the most prominent task for attention: Next token prediction. The way that large language models work is by training them to predict the next words in a text sequence, one at a time. 

The addition in the Add & Norm part is called a residual connection, or skip connection. It looks innocent enough, but its introduction had a huge impact on Machine Learning. The introduction of these residual connections strongly improved gradient flow in networks, and it is what made deep architectures possible. Before using residual connections, we were stuck on models of around 20-30 layers before training failed, limiting the capacity of the models. The first models using residual connections, ResNets, jumped up to 152 layers, with a direct boost in performance.

## Outro
    With that, we're done with our introduction to attention. It's a really cool mechanism: it's not too difficult to understand and implement, and it's just everywhere in Machine Learning. With our intro to attention, we are done with the introductory material. In the next video, we'll start with AlphaFold specific content. Concretely we'll implement feature extraction for AlphaFold, which in this case, means going from a text file to tensors. However, you still need to finish the notebook for this video, in which we build the Multi-Head Attention module that we'll use in the actual AlphaFold implementation. Aside of the attention module, a big part of the notebook is devoted to the implementation of a transformer encoder for sequence classification. I think you can learn a lot there, on the architecture itself but also on the training of models and transfer learning, which means fine-tuning a model that was trained on a different task. So, see you in the next video, and Happy Coding!
