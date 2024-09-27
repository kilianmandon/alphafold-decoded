Hi everyone, and welcome to the final video in our AlphaFold implementation series. I’m Kilian Mandon, and today, we’ll take everything we’ve built over the last eight videos and put it together to create a working AlphaFold model.

Bringing these modules together is a relatively straightforward task, so this video will be short. We’ll begin with a brief overview of each module, then walk through the code to integrate them, and finally discuss some common issues you might encounter.

The complete AlphaFold model consists of six modules:

	1.	Input Embedder: This module passes the extracted features through linear layers to create their embeddings.
	2.	Extra MSA Embedder: Similar to the input embedder, it processes the extra MSA feature.
	3.	Recycling Embedder: This integrates outputs from previous model iterations into the current prediction.
	4.	Evoformer Stack which is the core of the network, enriching both the MSA and pair representations with detailed information on the protein.
	5.	ExtraMSA Stack: A shallower version of the Evoformer, designed to handle the large number of extra sequences efficiently.
	6.	Structure Module: Converts the tensors from the Evoformer's output into precise atom positions.

While not directly part of the full model, the feature extraction module we built first plays a crucial role in the inference chain, converting the initial Multiple Sequence Alignment file into tensors.

Let's look at some of this with a little more depth. 
This is the main part of AlphaFold. The four features on the left are created during feature extraction. One core part of feature extraction was the idea that some of the sequences in the MSA file are selected as cluster centers and contribute directly, while others only contribute as averages over the clusters for the main features. These extra sequences are gathered in the extra MSA feat, over which they can also contribute individually to the feature creation, but through a shallower model, the Extra MSA Stack.

During Input Embedding, the initial features are fed through a number of linear layers to create the pair representation and the MSA representation. These two are the core tensors in the module, and most of the compute is spent to enrich them with information in the Evoformer. Before they reach the evoformer however, they are modified by the Recycling Embedder - a small module that performs layer normalization on the representations and modifies the pair representation by the pointwise distances of the pseudo-beta carbons, sorted into bins and one-hot encoded. The recycling embedder is part of a core concept of AlphaFold: Running the model multiple times, and improving the next output using the previous one. 

After the Recycling Embedder comes the Evoformer - a transformer-like module, that updates the MSA representation and pair representation using mostly row-wise and column-wise attention layers. The evoformer consists of two stacks, the MSA stack and the pair stack, with two information channels in-between: communication from the pair stack to the MSA stack by using the pair representation as bias in one of the Attention Modules in the MSA stack, and communication from the MSA stack to the pair stack with the module Outer Product Mean. In addition to the attention modules, the pair stack uses so called triangle multiplicative updates. These are basically sum-products of the rows or the columns, with a theoretical background of including information on the third edge in a triangle of three residues into the calculation. 

After the Evoformer, the model uses the Structure Module to convert the representations into actual atom positions, using an attention mechanism that bases attention on closeness in the currently assumed positions of the residues. 

The AlphaFold Inference Method puts all these modules together into the complete model. This is a slightly simplified version of the inference code. It leaves out the template stack - which we didn't implement - and it doesn't use ensembling, which is a technique where the model is run multiple times with slightly differing inputs, averaging the results. In AlphaFold, ensembling is an optional feature.

Nothing of this code is that surprising. Basically, it just stitches together the different modules. Two things of note are how Recycling is handled - the model uses zero initialization for all recycled tensors in the first round, then feeds them into the model in each following iteration - and how each recycling iteration uses newly created inputs from feature extraction. Because of the random elements during feature creation, this creates slightly different inputs in each cycle. 

We provide you with code for fitting the weights from openfold into our model architecture - there are some necessary changes, like switching some rows and columns in the weight matrices or renaming weights - because we didn't do an exact mirror of the OpenFold implementation. 

When writing the code yourself, you can expect that you'll need to chase down some errors, which were not detected by the checks in the individual notebooks. For example, the checks didn't cover for the datatypes of the tensors you returned, or which device they are on - cpu or gpu. If so, you can fix most of them by going into the method and setting the datatype or device setting by grabbing it from one of the input tensors. Note that for really running the network, you should use a machine with a compatible gpu - for example a free google colab machine. 

So, this was the last video of this series. Congratulations on making it through this complex journey! You’ve built a powerful tool, and I hope you’re as excited as I am about the potential of AlphaFold. As always, you can find the jupter notebook where you do the implementation yourself in the description. Thanks for coming with me on this journey, i really enjoyed making these videos and i hope that you learned something in this full-on deep dive into structural biology. AlphaFold hinted that they'll do a code release on AlphaFold 3 in the winter, so if everything works out, we might see each other again for that. Until then, Happy Coding!