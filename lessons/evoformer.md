# Evoformer

Hi everyone and welcome to this video on the Evoformer in AlphaFold! I'm Kilian Mandon and this video is the fifth in our series where we'll be implementing AlphaFold from scratch. 

The Evoformer is the largest component of the model in terms of parameters. Our full model will have approximately 93M parameters, with the Evoformer accounting for 88M of them. We can infer from the attention weights of the Evoformer's final layers that it already has a solid grasp of the protein structure: The heatmaps at the bottom reveal which residues where used for the attention update, and the attention pattern looks strikingly similar to the pairwise distances that the model predicts in the end. The Structure Module that follows the Evoformer then translates this concept into actual predicted atom coordinates.

The evoformer consists of several identical blocks with attention as its core mechanism, somewhat like the transformer architecture we explored in our video on attention. Still, the architecture is a little more elaborate and we'll examine it step-by-step.

In the last video on feature extraction, we already quickly glanced at this diagram. With feature extraction, we managed to create the four tensors on the left, the extra_msa_feat, the residue_index, the target_feat and the msa_feat. Now, we'll be skipping over everything in between and jump directly to the Evoformer. We'll cover the input embeddings in the next video. This is because the Extra MSA Stack - which is part of the input embedder - is actually just a smaller version of the Evoformer, so it's easier to learn it this way around. 

If you feel like you can't wait that long, here's a super quick rundown: The linear layers should be nothing new to you. Outer sum is just like the outer product, only with plus. relpos is the way that AlphaFold handles positional encodings: We don't directly replace the indices with learned vectors, but for each pair of residues, we take the difference of their indices and replace that by a learned vector. The tiling of target_feat just means broadcasting it to the bigger shape, and the two Rs next to the pair and MSA representation show where the Recycling Embedding takes place. AlphaFold runs its whole architecture several times, and the Recycling Embedder uses the outputs from the last iteration to update the pair and the MSA representation.

If you want to you can pause the video and check if all of that makes sense to you, that way you can watch the next video on double speed. But for now, all you need to know is that there's we end up with a pair representation of shape (r, r, c_z) and an MSA representation of shape (s_c, r, c_m), with r being the number of residues in the protein, s_c the number of sequences we selected as cluster centers during input embedding, and c_z and c_m the embedding dimensions of the representations.

The evoformer consists of a number of identical blocks, and the blocks look like this. Each block works on the MSA representation and the pair representation and transforms them to outputs of the same shape. It is split up into a MSA stack, the top row, and the pair stack, the bottom row, with two information channels between them. 

The basic idea for the Evoformer is this: We want to do attention on the inputs, but they are just too big to do a full attention mechanism. Imagine we have a protein that's 400 residues long. The pair representation would then have 160.000 entries. That is an incredibly big sequence length for attention, in comparison, the standard GPT-4 model has a context window of 8k tokens. With attention having squared memory demand, we'd need 102 Gigabytes of graphics memory just for one attention module. 
So, knowing we can't do full attention, AlphaFold uses either row-wise or column-wise attention. For 400 residues, that would mean 400^3 floats in the attention scores, which are around 256 megabytes for one attention module. 

We can see that in the diagram: The MSA stack consists of row-wise attention that uses the pair representation as bias in the attention mechanism, column-wise attention, and a transition, which is just a two-layer feed-forward net. Outer product mean is what it says: An outer product along the residue dimension, and averaging out the sequence dimension, to get the MSA representation to shape (r, r) so that it can be added to the pair representation. 
The modules in the pair stack look a little more funny. The two attention blocks, triangle attention around starting node and triangle attention around ending node are basically row-wise and column-wise attention as well, using the pair representation itself as bias. The name 'triangle attention' comes from a cool interpretation of the entries that explains the use of the bias as completing the third edge of a triangle in a graph - we'll look at that later. The two triangle updates before it are a row-wise and column-wise sum-product with the same triangle idea, and the transition module is a two-layer feed-forward net again. 

Lucky for us, we implemented our multi-head attention module in a way that already supports row-wise and column-wise attention: All we need to do is specifying the attention dimension for the module correctly, and constructing the biases where needed.

But before we start with that, let's do a brief recap on attention. For gated self-attention with bias, we compute for linear embeddings from our input sequence: The key, query and value embeddings, and the gate embedding. We compute the scaled dot-product of each query with each key, add the bias and use softmax to normalize the attention weights. Then, for each query, we scale the value vectors by the according attention weights and add them up to get the intermediate output. This is multiplied against the gate embeddings, normalized to range (0, 1) by sigmoid, and passed through a linear layer to get the final output. In this notation here, a linear layer going from ci to (h, c) means a linear layer going from ci to h*c, followed by reshaping the output. 

So, if we wanted to do attention on the MSA representation for example, we'd first need to select the input sequence. If it's rowwise attention, the input's just one row of the representation. As we saw, the row-wise attention in the MSA stack uses the pair representation as bias. That works pretty well, because the shapes almost match. We only need to use a linear embedding to transform the channel dimension to the attention head dimension of the multi-headed attention. If you're wondering, the reason why the bias looks like it has more dimensions than the pair representation is because I only used the staggered-map-visualization to represent the different heads. In the pair representation, each colored square represents a vector, while in the bias, dot-product affinities and attention weights, each square is one number. That means the pair representation and the bias both have 3 dimensions and everything works out nicely. 

So, let's see how this looks in code. This is pseudo-code from the AlphaFold paper for the row-wise gated self-attention with pair bias module. You can find it in the Supplementary Information of the paper.
The code starts with a layer normalization of the MSA representation m. Line 2 and 4 create the query, key, value and gate embeddings for the attention mechanism. In line 3, the pair representation is turned into the bias, using layer normalization followed by a linear layer to transform the channel dimension into the number-of-heads dimension. Note that b is the only letter here that's not drawn bold. That's because b the entries of b are single numbers, while all other symbols here are vectors.

In line 5 and 6, we've got a mathmatical formulation of a row-wise attention mechanism. You can see that it's row-wise because the row index, s, is just being carried along for the queries and keys in line 5 and for the gating, the attention scores and the value vectors in line 6. The column index on the other side is where the attention mechanism happens: It's the i-index of the queries and the j-index of the keys, showing how they act in an each-with-each fashion during multiplication, and it's also the index along which the softmax function does the normalization.
In line 7, the outputs from the attention heads are concatenated and passed through the output linear layer.

These steps should look like the attention flowchart we just went through, even though the notation might look a little odd. But, when you're doing the actual implementation of this, you won't need to go through most of them, given that we already implemented the attention mechanism. The actual python code will look more like this. The first two steps are identical to line 1 and 3. Moving the axis in the bias is more of a technical problem: In our attention implementation, the number of heads isn't the last dimension but the third from last. For the real attention mechanism, all we need to do is pass the m and b to a Multi-Head attention module, that had the attention dimension correctly specified during initialization. Note that for row-wise attention, we'd need to specify the column dimension as the attention dimension, as it is the index that is actually cycled through during row-wise attention. 

The column-wise attention function looks mostly like the row-wise attention, we don't even have to prepare a bias here. You can see that it's column-wise this time in line 4: The column index, i, is kept as is during the attention mechanism, while the each-with-each vector multiplication of the keys and queries happens along the row-dimension, denoted by s and t respectively. 
Line 5 is the only time that the pseudocode in this series will deviate from the code in the AlphaFold paper without it being a typo by me during typesetting: In the paper, the indices for v in line 5 are st instead of ti. I'm pretty sure I did it correctly, but I'm a little afraid of going against a paper by Deepmind, so I'd be glad to hear your opinions on that.

The last part of the MSA stack is the MSA transition, and as I said, it's just a two-layer feed-forward neural net, with layer normalization at the start. It goes to a higher-dimensional intermediate with 4 times the dimensions before going back. Such a feed-forward block is typical for transformer networks, including the number 4 as a scaling factor.

With that, we covered the three modules from the MSA stack. It's connected to the pair stack by Outer Product Mean, an operation that's designed to reshape the MSA representation to the pair representation. There are two things to note in this code: First, the outer product is not calculated between m and itself, but between two linear embeddings a and b of m. This increases the flexibility for the model to bring meaning into the outer product. The second thing to note is that we are actually calculating two outer products here: The first one comes from the way a and b are indexed, the column index of a is i and the one of b is j. This means we have an each-with-each calculation of the tensors, and its what creates the desired shape. The second outer product is along the channel dimension, and it's denoted by the cross operator in between them. The different channels of the tensors are multiplied each-with-each as well, expanding them to c by c, or c times c after flattening. The row dimension is contracted by taking the mean, and the c*c channel dimension is brought down to c_z by a linear layer. Line 3 might look a little convoluted, but it's just a linear operation again, and that makes it a perfect candidate for our trick of just designing einsum expressions that fit the dimensions. In python, this method looks like this: In the einsum expression, the row dimension is contracted, while the column dimensions and the channel dimensions are expanded to create the correct output. 
However, there's an actual difference between the python implementation and the pseudocode. If you want to, you can pause the video and try to spot it yourself.

Our einsum expression simply summed up along the row dimension, but the pseudocode tells us to take the mean. We divide by the number of sequences, but we do so after the linear_out layer, not before it. That's not a big difference but it is one. Concretely, it additionally scales down the bias by the number of sequences, compared to if you did it before the layer. This shouldn't have a big impact and the model would probably train just as well using the operation like its written in the pseudocode. We however don't train the model, which means we have to stick to the actual implementation so that the pretrained weights work for us. 

With OuterProductMean, we arrived at the pair stack. As we noted earlier, most of the modules in it start with the word triangle, and therefore we'll look at why that is before going into the individual algorithms. To understand the triangle mechanism, we need to think about what the representations might actually encode. For the MSA representation, this is pretty straightforward. We've got the different source organisms for one dimension and the different residue positions as the other, so each vector might carry information on a specific residue in a specific protein. If we have operations that are row-wise, we have cross-talk between the individual positions, but for each of the organisms in isolation. Column-wise operations allow looking at information at the same position but in other organisms. It's pretty clear why this type of information flow might be helpful for structure predicition.

However, if we go over to the pair stack, things are not that straightforward anymore. As its name suggests, we hope that the individual entries carry information on the interaction between pairs of residues. In fact, we have two entries per pair: The entry (i, j) and the entry (j, i). With that, we can imagine that one of them, ij, carries information going from i to j, while the entry ji carries signals from j to i. You can imagine a glutamate telling a nearby glycine it's negatively charged, and the glycine sending back that it doesn't care. Of course, this is all theoretical. The actual values of the entries just arise as learned features, and their semantic meaning is poorly analyzed. Still, AlphaFold was highly successful, so it might be good to pay closer attention to the theory behind its architecture. 

So, with the interpretation that the entries carry information on the directed interaction between two residues, let's look at what an update, like an attention update would look like. Let's say for example we wanted to update the pair ij from the pair ik. This is called an outgoing update or an update around the starting node. Since the first index, i, is fixed, this corresponds to row-wise operations, like row-wise attention or a row-wise multiplicative update. The authors from AlphaFold suggest that in this situation, it's important for the model to have information from interaction between j and k to complete the triangle. If the entries carry information on the distance for example, it's necessary to also take the distance between j and k into account, so that don't end up violating the three-dimensional structure in the distances, maybe by violating the triangle equation on the distances. The choice between jk and kj is a little arbitrary, but given that we're updating ij, AlphaFold uses jk in the spirit of outgoing edges. 

We can do everything just as well for incoming edges, or updates around the ending node. Here, we want to update ij based on kj. Since the column index is fixed, this corresponds to column-wise operations. Just like before, the authors suggest that it's beneficial to also take the values from ki into the equation, to complete the triangle with the last, incoming edge. 

With this theory, let's look at the pair stack. We can see that the first and third block are using outgoing edges or attention around starting node, which we identified as row-wise operations, while the other triangle updates use the incoming edges or attention around ending node, which are column-wise operations. 

The multiplicative update starts with a few steps to create two embeddings a and b of the pair representation. They are computed as a linear embedding times a gate-like linear-sigmoid embedding. The module also creates an output gate g that's multiplied against the layer's output. The interesting operation happens in line 4, where there's actual cross-talk between the positions. Here, using outgoing edges, we want to calculate an update of the entry ij with information from the outgoing edges ik for different k. Our thoughts on triangular attention suggest that to do so, we need to take jk into account as well. The way this is done here is by multiplying the vectors a_ik with b_jk pointwise, before summing them up. This summation is followed by layer normalization, a linear layer and multiplication against the gates.  
We do basically the same thing in the incoming updates. Highlighted are all the differences to the algorithm we just saw. Here, to compute the update for ij, we want to look at the other incoming edges kj. To complete the triangle, we also consider the entries ki, and the concrete operation is pointwise multiplication of the two columns followed by summation along the row dimension. Here you can see the two algorithms side by side and look at what makes them different. Using outgoing edges, we operate on two different rows aik and bjk, while for incoming edges, we operate on two columns aki and bkj. 

After the multiplicative updates, the next part of the pair stack are attention, around the starting node and around the ending node, which basically means "using outgoing edges" and "using incoming edges" again. Looking at the code for attention around the starting node, this is simply row-wise attention using the pair representation itself as the bias, with the same trick we used earlier to turn the channel dimension into the number-of-heads dimension. But we can see that this fits nicely into our theory on completing the triangles: The update for the entry ij corresponds to the query qij. It can get updated by all the different k_ik, which are the outgoing edges from i. But, the attention scores we get for the key ik is influenced by the bias b_jk, which is the third edge in the triangle. 

Attention around ending node is basically the same. The differences to the previous algorithms are highlighted in yellow. This time, we have column-wise attention, again using the pair representation itself as bias. But this time, we transpose the bias. You can tell from the indices in line 5: The column index j is fixed for q and k, as should be for column-wise attention. Using the query index as the first dimension and the key index as the second dimension of the attention scores, the attention scores have the indices in the order ik. b however is indexed as ki, which means that when going to PyTorch, we'd need to transpose it before adding it so that we implement the algorithm correctly. Like in the previous algorithm, the update to ij from kj is influenced by the third edge ki using the bias. 

The final part of the pair stack, the pair transition, is just a two-layer feed-forward module again, just like it was for the MSA representation. 

Putting all of it together, this is the full code for the EvoFormer. The methods in Yellow are the AlphaFold specific methods we went through in this video. As you can see, they are called in the order we've seen in the flow-chart: MSARowAttentionWithPairBias, MSAColumnAttention, MSATransition, OuterProductMean to go over to the PairStack, then in the pairstack outgoing triangle multiplication, incoming triangle multiplication, attention around starting node, attention around ending node, and the pair transition. 

Aside of the calls to the submethods, there are some other implementation details we can see here: As we discussed earlier, the evoformer consists of multiple identical blocks, 48 in fact, each with its own weights. The rowwise and columnwise dropout can be savely ignored. Dropout is a form of regularization that's used during training. It randomly sets some values to zero while scaling the other ones up so that they have the same expected value. AlphaFold uses dropout setting whole rows or columns in the representations to zero. But, we aren't training the model and Dropout isn't applied during inference, so we don't have to worry about it. 
One big detail to note is that we don't set the outputs to the results of the operations, but we add them to the inputs. These are the so called residual connections we talked about in the part on attention. Residual connections make training way more stable. If you think of an operation like the multiplicative updates, where we just took two rows in the full tensor, multiplied them and summed them up to generate an output, you can imagine that this would be a pretty transformative operation without residual connections. With residual connections, the model can just learn small weights for the embeddings and generate an output close to zero, that's more like an offset to the input then a complete substitution.

Finally, the Evoformer ends off with taking the first row of the MSA representation and passing it through a linear layer. This is the so called single representation and it's the only part of the MSA representation that goes on from the Evoformer to the structure module. The rest of the MSA representation is dropped and won't be used by the model anymore. It's somewhat like in our sentiment analysis implementation in the notebook on attention, where we only kept the output from the first token for classification, and the model needs to learn to aggregate the information into that. If we think back to how we constructed the MSA feature, the first row in it came from the actual target sequence, so it makes sense that this is what's living on. 

So, quickly going over the full architecture again:
The evoformer built up by a number of identical Evoformer blocks. Each block consists of an MSA stack and a pair stack, with two channels of information in between. The MSA stack consists of three modules: Row-wise attention using the pair representation as bias, column-wise attention, and a transition module that's just a two-layer feed-forward net. It's reshaped to the shape of the pair representation by outer product mean, taking the outer product along the residue dimension and averaging out the sequence dimension. This is added to the pair representation for information flow from the MSA representation to the pair representation. The pair stack itself consists of two multiplicative updates, two attention updates and another transition module. The multiplicative updates are computed as the sum-product of two of the rows or two of the columns respectively, and the attention updates are row-wise attention using the pair representation as bias and column-wise attention, using the pair representation as bias as well, but transposed.

With that, we're done with the Evoformer. Compared to an architecture like the transformer, the evoformer is a bit more involved, but I hope I could convince you that the individual steps aren't really that complicated. Using attention only column-wise or only row-wise isn't a too difficult solution to a tensor that's too large. The triangle attention mechanism is arguably clever, but maybe you could have come up with it by accident, given that it boils down to just using the pair representation itself as bias. 
In the description, I linked the Jupyter Notebook where you can do the full implementation of the Evoformer yourself, with automatic checks. In the next video, we'll close the gap we left between feature extraction and the evoformer. Until then, good luck with the notebook, and happy coding!