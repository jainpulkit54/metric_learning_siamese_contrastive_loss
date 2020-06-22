# metric_learning_siamese_contrastive_loss

MNIST datasets are used for training to get their respective embeddings. <br>
I have here used 2-dimensional embeddings which is not the best choice for other complicated datasets but works fine for easy datasets like "MNIST". <br>

The architecture details of the embedding network used is: <br>
### (32 conv 5x5 --> PReLU --> MaxPool 2x2 --> 64 Conv 5x5 --> PReLU --> MaxPool 2x2 --> Dense 256 --> PReLU --> Dense 256 --> PReLU --> Dense 2) <br>

The loss function used for training is the "contrastive loss" and the image pairs are selected randomnly (look for the code in the "datasets.py" file). The embeddings obtained for the dataset are in the "outputs" folder obtained after training for 20 epochs.
