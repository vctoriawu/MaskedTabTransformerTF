import numpy as np
import tensorflow as tf
from maskedtabtransformertf.models.tabtransformer import TransformerBlock
from tensorflow.keras.layers import (
    Dense,
    Flatten,
)
from tensorflow.keras.metrics import (
    Mean,
)

import math as m
from maskedtabtransformertf.models.embeddings import CEmbedding, NEmbedding
from keras import backend as K
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from positional_encodings.tf_encodings import TFPositionalEncoding1D

class FTTransformerEncoder(tf.keras.Model):
    def __init__(
        self,
        categorical_features: list,
        numerical_features: list,
        numerical_data: np.array,
        categorical_data: np.array,
        y: np.array = None,
        task: str = None,
        embedding_dim: int = 64,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        numerical_embedding_type: str = 'linear',
        numerical_bins: int = None,
        ple_tree_params: dict = {},
        explainable=False,
        positional = False
    ):
        """FTTransformer Encoder
        Args:
            categorical_features (list): names of categorical features
            numerical_features (list): names of numeric features
            categorical_lookup (dict): dictionary with categorical feature names as keys and adapted StringLookup layers as values
            embedding_dim (int, optional): embedding dimensions. Defaults to 32.
            depth (int, optional): number of transformer blocks. Defaults to 4.
            heads (int, optional): number of attention heads. Defaults to 8.
            attn_dropout (float, optional): dropout rate in transformer. Defaults to 0.1.
            ff_dropout (float, optional): dropout rate in mlps. Defaults to 0.1.
            mlp_hidden_factors (list[int], optional): numbers by which we divide dimensionality. Defaults to [2, 4].
            numerical_embeddings (dict, optional): dictionary with numerical feature names as keys and adapted numerical embedding layers as values. Defaults to None.
            numerical_embedding_type (str, optional): name of the numerical embedding procedure. Defaults to linear.
            use_column_embedding (bool, optional): flag to use fixed column positional embeddings. Defaults to True.
            explainable (bool, optional): flag to output importances inferred from attention weights. Defaults to False.
        """

        super(FTTransformerEncoder, self).__init__()
        self.numerical = numerical_features
        self.categorical = categorical_features
        self.numerical_embedding_type = numerical_embedding_type
        self.embedding_dim = embedding_dim
        self.explainable = explainable
        self.depth = depth
        self.heads = heads

        # Two main embedding modules
        if len(self.numerical) > 0:
            self.numerical_embeddings = NEmbedding(
                feature_names=self.numerical,
                X=numerical_data,
                y=y,
                task=task,
                emb_dim=embedding_dim,
                emb_type=numerical_embedding_type,
                n_bins=numerical_bins,
                tree_params=ple_tree_params
            )
        if len(self.categorical) > 0:
            self.categorical_embeddings = CEmbedding(
                feature_names=self.categorical,
                X=categorical_data,
                emb_dim=embedding_dim
            )
        
        # Add positional encoding
        if (positional):
            self.pos_emb = TFPositionalEncoding1D(embedding_dim)
        else:
            self.pos_emb = None

        # Transformers
        self.transformers = []
        for _ in range(depth):
            self.transformers.append(
                TransformerBlock(
                    embedding_dim,
                    heads,
                    embedding_dim,
                    att_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    explainable=self.explainable,
                    post_norm=False,  # FT-Transformer uses pre-norm
                )
            )
        self.flatten_transformer_output = Flatten()


    def call(self, inputs, training=None):
        
        transformer_inputs = []

        # If numerical features, add to list
        if len(self.numerical) > 0:
            num_input = []
            for n in self.numerical:
                num_input.append(inputs[n])
            num_input = tf.stack(num_input, axis=1)[:, :, 0]
            num_embs = self.numerical_embeddings(num_input, training)
            transformer_inputs += [num_embs]

        # If categorical features, add to list
        if len(self.categorical) > 0:
            cat_input = []
            for c in self.categorical:
                cat_input.append(inputs[c])

            cat_input = tf.stack(cat_input, axis=1)[:, :, 0]
            cat_embs = self.categorical_embeddings(cat_input)
            transformer_inputs += [cat_embs]

        # Prepare for Transformer
        transformer_inputs = tf.concat(transformer_inputs, axis=1)
        importances = []

        # Add positional embeddings
        if (self.pos_emb != None):
            transformer_inputs = transformer_inputs + self.pos_emb(transformer_inputs)

        # Pass through Transformer blocks
        for transformer in self.transformers:
            if self.explainable:
                transformer_inputs, att_weights = transformer(
                    transformer_inputs)
                importances.append(tf.reduce_sum(
                    att_weights[:, :, 0, :], axis=1))
            else:
                transformer_inputs = transformer(transformer_inputs)

        if self.explainable:
            # Sum across the layers
            importances = tf.reduce_sum(tf.stack(importances), axis=0) / (
                self.depth * self.heads
            )
            return transformer_inputs, importances
        else:
            return transformer_inputs


class FTTransformer(tf.keras.Model):
    def __init__(
        self,
        categorical_features: list = None,
        numerical_features: list = None,
        categorical_lookup: dict = None,
        embedding_dim: int = 64,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        numerical_embedding_type: str = None,
        numerical_embeddings: dict = None,
        explainable: bool = False,
        encoder=None,
        cat_ra_pressure: bool = False,
        ra_pressure_weights: list = None,
        loss_lambda: int = 1,
        dataloader_mask: bool = False
    ):
        super(FTTransformer, self).__init__()

        # Initialise encoder
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = FTTransformerEncoder(
                categorical_features=categorical_features,
                numerical_features=numerical_features,
                embedding_dim=embedding_dim,
                depth=depth,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                numerical_embedding_type=numerical_embedding_type,
                explainable=explainable,
            )

        #Reconstruction layers
        self.num_features = (len(self.encoder.numerical) if self.encoder.numerical is not None else 0) + \
               (len(self.encoder.categorical) if self.encoder.categorical is not None else 0)

        self.masked_predictions_layer = Dense(units=(self.num_features+2))

        self.loss_tracker = Mean(name="loss")
        self.sparse_ce_loss = SparseCategoricalCrossentropy(from_logits=True)
        self.cat_ra_pressure = cat_ra_pressure

        if ra_pressure_weights is not None and not ra_pressure_weights.empty:
            # Scale inverse frequencies by the highest frequency
            max_frequency = np.max(ra_pressure_weights)
            scaled_inverse_frequencies = 1 / (ra_pressure_weights / max_frequency)
            self.class_weights = scaled_inverse_frequencies

        self.loss_lambda = loss_lambda # how much the categorical loss should affect the total loss
        self.dataloader_mask = dataloader_mask

    def call(self, inputs, training=None):
        if self.encoder.explainable:
            x, expl = self.encoder(inputs, training)
        else:
            x = self.encoder(inputs, training)

        reshaped_x = tf.reshape(x, [-1, self.num_features*self.encoder.embedding_dim])
        masked_preds = self.masked_predictions_layer(reshaped_x)
        output_dict = {"masked_preds": masked_preds}

        if self.encoder.explainable:
            # Explainable models return three outputs
            output_dict["importances"] = expl

        return output_dict
    
    def calculate_sample_weights(self, y_true):
        # Create a tensor for class weights
        class_weights_tensor = tf.constant(self.class_weights, dtype=tf.float32)

        # Cast y_true to int32 before using as indices
        y_true_int = tf.cast(y_true, dtype=tf.int32)

        # Index the class weights using y_true
        sample_weights = tf.gather(class_weights_tensor, y_true_int)

        return sample_weights

    def masked_mse(self, y_true, y_pred):
            
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        nan_mask = K.cast(K.not_equal(y_true, -888), dtype=tf.float32)
        masked_true = y_true * nan_mask

        if self.cat_ra_pressure: # if the ra pressure is categorcal, use CE loss
            masked_pred = y_pred[:, :-3] * nan_mask[:, :-1]
            y_true_ra_pressure = masked_true[:, -1]  # Assuming the last column is "ra_pressure"
            y_pred_ra_pressure = tf.nn.softmax(y_pred[:, -3:], axis=-1)

            y_true_other = masked_true[:, :-1]  # Assuming the first columns are for numerical predictions
            y_pred_other = masked_pred
            
            #divide by the number of present values 
            mse_loss = K.mean(K.square(y_true_other - y_pred_other))

            if self.class_weights is not None and not self.class_weights.empty:
                loss_weights = self.calculate_sample_weights(y_true_ra_pressure)
                sparse_ce_loss = self.sparse_ce_loss(y_true_ra_pressure, y_pred_ra_pressure, sample_weight=loss_weights) * nan_mask[:, -1]
            else:
                sparse_ce_loss = self.sparse_ce_loss(y_true_ra_pressure, y_pred_ra_pressure) * nan_mask[:, -1]
            total_loss = mse_loss + self.loss_lambda * sparse_ce_loss 
        
        else:
            masked_pred = y_pred * nan_mask
            mse_loss = K.mean(K.square(masked_true - masked_pred))
            total_loss = mse_loss

        return total_loss

    def train_step(self, data):
        
        if self.dataloader_mask:
            x = data[0]
            y = data[1]
        else:
            x = data
            y = x #unmasked input is the output

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            # Compute the loss value.
            y_true_tensor = tf.cast(tf.concat(list(y.values()), axis=-1),dtype='float32')#tf.concat(y, axis=-1),dtype='float32')#
            loss = self.masked_mse(y_true_tensor, y_pred["masked_preds"])

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        if self.dataloader_mask:
            x = data[0]
            y = data[1]
        else:
            x = data
            y = x #unmasked input is the output
        
        y_pred = self(x, training=False)

        #Reshape y and calculate loss
        y_true_tensor = tf.cast(tf.concat(list(y.values()), axis=-1),dtype='float32')#y, axis=-1),dtype='float32') #
        loss = self.masked_mse(y_true_tensor, y_pred["masked_preds"])

        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

        
        @property
        def metrics(self):
            # We list our `Metric` objects here so that `reset_states()` can be
            # called automatically at the start of each epoch
            # or at the start of `evaluate()`.
            # Otherwise they would track average over entire training
            
            return[self.loss_tracker]
