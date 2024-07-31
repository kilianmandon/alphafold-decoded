Hi everyone and welcome to this video on feature embedding in AlphaFold! I'm Kilian Mandon and this video is the sixth in our series where we'll be implementing AlphaFold from scratch.

With feature embedding, we're bridging the gap between feature extraction and the Evoformer, which we've built in the last videos. Feature embedding refers to the first few layers of a network, where we embed the initial features - which are often one-hot encoded - into the first, learned embedding vectors. Additionally, we are creating positional encodings for the model, to make up for the fact that the attention mechanism has no grasp of order without them, and we'll build the Recycling Embedder that uses the predicted output of the model to feed it into a new round, enhancing the prediction step-by-step.

So, let's look into how all of that works in a little more detail:

...

With that, we are done with feature embedding. It's a relatively small part of AlphaFold, and you'll probably need a little less time for actually writing the code as well. As always, you can find a jupyter notebook in the description where you can do the full implementation yourself. Basically, all that's left to do now is building the structure module and stitching all the parts together. But, building the structure module requires some non-standard knowledge on 3D geometry: We'll use rotation matrices, homogenous coordinates and quaternions to build the three dimensional structure of the protein. So, if all of these are words you never heard before, or if it's something that you heard but never truly understood, I'll be more than happy to explain them in the next video - I'll think it'll be my favorite in the series.
So, see you there and happy coding!