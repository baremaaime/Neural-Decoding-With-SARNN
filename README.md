# Neural-Decoding-With-SCRNN
 How does the brain transform bursts of neural activity into a sense of direction or location?
 How can one decode and uncover these two just from neural spike? This thesis explores that
 question by using patterns of recorded brain signals to predict either the direction a mouse is
 facing or its position in space. To do so, it combines geometric insights and with learning
 from time-based patterns. This approach captures not just individual signals, but how groups of
 neurons work together over time by converting the relationship into simplicial complexes to make
 it more conveniently interpretable.
## Files Summary:
* Simplicial Attention Recurrent Neural Networks (SARNNs), combine simplicial Attention with backend recurrently connected layers.
* Decoded Neurons: grid cells and head direction (HD) cells.
* The neural activity is first defined on a simplicial complex via a pre-processing procedure and then fed to the SARNN for decoding.
* Comparisons to a simplicial convolutional recurrent neural network (SCRNN).
* main.py: SARNN
* main_SCRNN.py: SCRNN or SCNN by setting the RNN variable to `True` of `False` respectively# Neural-Decoding-With-SARNN
