import tensorflow as tf
from tensorflow.keras import layers, models, initializers, regularizers, constraints
import numpy as np
from layers.graph import GraphLayer,GraphConv

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import keras.backend as K
import pandas as pd
import numpy as np
import os

class DenseMoE(layers.Layer):
    def __init__(self, units,
                 n_experts,
                 n_gates,
                 expert_input_dims,
                 feature_dim,
                 expert_activation='LeakyReLU', #relu
                 gating_activation='ReLU', #LeakyReLU
                 use_expert_bias=True,
                 use_gating_bias=True,
                 expert_kernel_initializer_scale=1.0,
                 gating_kernel_initializer_scale=1.0,
                 expert_bias_initializer='zeros',
                 gating_bias_initializer='zeros',
                 expert_kernel_regularizer=None,
                 gating_kernel_regularizer=None,
                 expert_bias_regularizer=None,
                 gating_bias_regularizer=None,
                 expert_kernel_constraint=None,
                 gating_kernel_constraint=None,
                 expert_bias_constraint=None,
                 gating_bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super(DenseMoE, self).__init__(**kwargs)
        self.units = units
        self.n_experts = n_experts
        self.n_gates = n_gates
        self.feature_dim = feature_dim
        self.expert_input_dims = expert_input_dims

        self.expert_activation = tf.keras.activations.get(expert_activation)
        self.gating_activation = tf.keras.activations.get(gating_activation)
        self.use_expert_bias = use_expert_bias
        self.use_gating_bias = use_gating_bias

        self.expert_kernel_initializer_scale = expert_kernel_initializer_scale
        self.gating_kernel_initializer_scale = gating_kernel_initializer_scale

        self.expert_bias_initializer = tf.keras.initializers.get(expert_bias_initializer)
        self.gating_bias_initializer = tf.keras.initializers.get(gating_bias_initializer)

        self.expert_kernel_regularizer = tf.keras.regularizers.get(expert_kernel_regularizer)
        self.gating_kernel_regularizer = tf.keras.regularizers.get(gating_kernel_regularizer)

        self.expert_bias_regularizer = tf.keras.regularizers.get(expert_bias_regularizer)
        self.gating_bias_regularizer = tf.keras.regularizers.get(gating_bias_regularizer)

        self.expert_kernel_constraint = tf.keras.constraints.get(expert_kernel_constraint)
        self.gating_kernel_constraint = tf.keras.constraints.get(gating_kernel_constraint)

        self.expert_bias_constraint = tf.keras.constraints.get(expert_bias_constraint)
        self.gating_bias_constraint = tf.keras.constraints.get(gating_bias_constraint)

        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

    # one layer
    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.global_weights = tf.Variable(tf.ones(shape=(self.n_experts,)), trainable=True, name="global_weights")

        if isinstance(input_dim, tf.TensorShape):
            input_dim = input_dim.value

        self.expert_kernels = []
        self.expert_biases = []
        for i, expert_dim in enumerate(self.expert_input_dims):
            expert_init_lim = np.sqrt(3.0 * self.expert_kernel_initializer_scale / (max(1., float(expert_dim + self.units) / 2)))
            expert_kernel = self.add_weight(shape=(expert_dim, self.units),
                                            initializer=tf.keras.initializers.RandomUniform(minval=-expert_init_lim, maxval=expert_init_lim),
                                            name=f'expert_kernel_{i}',
                                            regularizer=self.expert_kernel_regularizer,
                                            constraint=self.expert_kernel_constraint)
            self.expert_kernels.append(expert_kernel)

            if self.use_expert_bias:
                expert_bias = self.add_weight(shape=(self.units,),
                                              initializer=self.expert_bias_initializer,
                                              name=f'expert_bias_{i}',
                                              regularizer=self.expert_bias_regularizer,
                                              constraint=self.expert_bias_constraint)
                self.expert_biases.append(expert_bias)
            else:
                self.expert_biases.append(None)

        self.gating_kernels = []
        self.gating_biases = []
        for i in range(self.n_gates):
            gating_init_lim = np.sqrt(3.0 * self.gating_kernel_initializer_scale / (max(1., float(self.feature_dim + 1) / 2)))
            gating_kernel = self.add_weight(shape=(self.feature_dim, self.n_experts),
                                            initializer=tf.keras.initializers.RandomUniform(minval=-gating_init_lim, maxval=gating_init_lim),
                                            name=f'gating_kernel_{i}',
                                            regularizer=self.gating_kernel_regularizer,
                                            constraint=self.gating_kernel_constraint)
            self.gating_kernels.append(gating_kernel)

            if self.use_gating_bias:
                gating_bias = self.add_weight(shape=(self.n_experts,),
                                              initializer=self.gating_bias_initializer,
                                              name=f'gating_bias_{i}',
                                              regularizer=self.gating_bias_regularizer,
                                              constraint=self.gating_bias_constraint)
                self.gating_biases.append(gating_bias)
            else:
                self.gating_biases.append(None)

    def call(self, inputs, feature_input, **kwargs):
        expert_inputs = tf.split(inputs, self.expert_input_dims, axis=-1)

        expert_outputs = []
        for i, expert_input in enumerate(expert_inputs):
            expert_output = tf.matmul(expert_input, self.expert_kernels[i])
            if self.use_expert_bias:
                expert_output = tf.keras.backend.bias_add(expert_output, self.expert_biases[i])
            if self.expert_activation is not None:
                expert_output = self.expert_activation(expert_output)
            expert_outputs.append(expert_output)

        expert_outputs = tf.stack(expert_outputs, axis=-1)
        print("expert_outputs", expert_outputs.shape)  # shape: (batch_size, units, n_experts)


        outputs_list = []
        for i in range(self.n_gates):
            gating_output = tf.keras.backend.dot(feature_input, self.gating_kernels[i])
            if self.use_gating_bias:
                gating_output = tf.keras.backend.bias_add(gating_output, self.gating_biases[i])
            if self.gating_activation is not None:
                gating_output = self.gating_activation(gating_output)
            
            weighted_gating_output = gating_output * self.global_weights
            gating_output_ = tf.keras.backend.expand_dims(weighted_gating_output, axis=1)

            gating_output_ = tf.nn.softmax(gating_output_/0.01)
            
            top_k_values, top_k_indices = tf.nn.top_k(gating_output_, k=6)
            top_k_indices = tf.squeeze(top_k_indices, axis=1)

            selected_expert_outputs = tf.gather(expert_outputs, top_k_indices, batch_dims=1,axis=-1)
            
            top_k_values = tf.squeeze(top_k_values, axis=1)

            softmax_weights = tf.nn.softmax(top_k_values/0.01)

            weighted_outputs = selected_expert_outputs * tf.expand_dims(softmax_weights, axis=1)
            output = tf.reduce_sum(weighted_outputs, axis=2)
            outputs_list.append(output)
            
        
        concat_output = tf.concat(outputs_list, axis=1)
        return concat_output, expert_outputs

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units * self.n_experts * self.n_gates
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'n_experts': self.n_experts,
            'n_gates': self.n_gates,
            'feature_dim': self.feature_dim,
            'expert_input_dims': self.expert_input_dims,
            'expert_activation': tf.keras.activations.serialize(self.expert_activation),
            'gating_activation': tf.keras.activations.serialize(self.gating_activation),
            'use_expert_bias': self.use_expert_bias,
            'use_gating_bias': self.use_gating_bias,
            'expert_kernel_initializer_scale': self.expert_kernel_initializer_scale,
            'gating_kernel_initializer_scale': self.gating_kernel_initializer_scale,
            'expert_bias_initializer': tf.keras.initializers.serialize(self.expert_bias_initializer),
            'gating_bias_initializer': tf.keras.initializers.serialize(self.gating_bias_initializer),
            'expert_kernel_regularizer': tf.keras.regularizers.serialize(self.expert_kernel_regularizer),
            'gating_kernel_regularizer': tf.keras.regularizers.serialize(self.gating_kernel_regularizer),
            'expert_bias_regularizer': tf.keras.regularizers.serialize(self.expert_bias_regularizer),
            'gating_bias_regularizer': tf.keras.regularizers.serialize(self.gating_bias_regularizer),
            'expert_kernel_constraint': tf.keras.constraints.serialize(self.expert_kernel_constraint),
            'gating_kernel_constraint': tf.keras.constraints.serialize(self.gating_kernel_constraint),
            'expert_bias_constraint': tf.keras.constraints.serialize(self.expert_bias_constraint),
            'gating_bias_constraint': tf.keras.constraints.serialize(self.gating_bias_constraint),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(DenseMoE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class KerasMultiSourceGCNModel:
    def __init__(self, use_mut, use_gexp, use_methy, regr=True):
        self.use_mut = use_mut
        self.use_gexp = use_gexp
        self.use_methy = use_methy
        self.regr = regr

    def createMaster(self, drug_dim, drug_emb_dim, gexpr_dim, units_list, use_relu=True, use_bn=True, use_GMP=True):
        drug_feat_input = layers.Input(shape=(None, drug_dim), name='drug_feat_input')
        drug_adj_input = layers.Input(shape=(None, None), name='drug_adj_input')

        drug_emb_input0 = layers.Input(shape=(drug_emb_dim[0],),name='drug_emb0')
        drug_emb_input1 = layers.Input(shape=(drug_emb_dim[1],),name='drug_emb1')
        drug_emb_input2 = layers.Input(shape=(drug_emb_dim[2],),name='drug_emb2')
        drug_emb_input3 = layers.Input(shape=(drug_emb_dim[3],),name='drug_emb3')
        drug_emb_input4 = layers.Input(shape=(drug_emb_dim[4],),name='drug_emb4')
        drug_emb_input5 = layers.Input(shape=(drug_emb_dim[5],),name='drug_emb5')
        drug_emb_input6 = layers.Input(shape=(drug_emb_dim[6],),name='drug_emb6')
        drug_emb_input7 = layers.Input(shape=(drug_emb_dim[7],),name='drug_emb7')

        gexpr_input0 = layers.Input(shape=(gexpr_dim[0],),name='gexpr_0')
        gexpr_input1 = layers.Input(shape=(gexpr_dim[1],),name='gexpr_1')
        gexpr_input2 = layers.Input(shape=(gexpr_dim[2],),name='gexpr_2')
        gexpr_input3 = layers.Input(shape=(gexpr_dim[3],),name='gexpr_3')
        gexpr_input4 = layers.Input(shape=(gexpr_dim[4],),name='gexpr_4')
        gexpr_input5 = layers.Input(shape=(gexpr_dim[5],),name='gexpr_5')
        gexpr_input6 = layers.Input(shape=(gexpr_dim[6],),name='gexpr_6')
        gexpr_input7 = layers.Input(shape=(gexpr_dim[7],),name='gexpr_7')
        gexpr_input8 = layers.Input(shape=(gexpr_dim[8],),name='gexpr_8')

        drug_emb_inputs = [drug_emb_input0, drug_emb_input1, drug_emb_input2, drug_emb_input3, drug_emb_input4, drug_emb_input5, drug_emb_input6, drug_emb_input7]
        gexpr_inputs = [gexpr_input0, gexpr_input1, gexpr_input2, gexpr_input3, gexpr_input4, gexpr_input5, gexpr_input6, gexpr_input7, gexpr_input8]

        GCN_layer = GraphConv(units=units_list[0], step_num=1)([drug_feat_input, drug_adj_input])
        if use_relu:
            GCN_layer = layers.Activation('relu')(GCN_layer)
        else:
            GCN_layer = layers.Activation('tanh')(GCN_layer)
        if use_bn:
            GCN_layer = layers.BatchNormalization()(GCN_layer)
        GCN_layer = layers.Dropout(0.1)(GCN_layer)

        for i in range(len(units_list) - 1):
            GCN_layer = GraphConv(units=units_list[i + 1], step_num=1)([GCN_layer, drug_adj_input])
            if use_relu:
                GCN_layer = layers.Activation('relu')(GCN_layer)
            else:
                GCN_layer = layers.Activation('tanh')(GCN_layer)
            if use_bn:
                GCN_layer = layers.BatchNormalization()(GCN_layer)
            GCN_layer = layers.Dropout(0.1)(GCN_layer)

        GCN_layer = GraphConv(units=512, step_num=1)([GCN_layer, drug_adj_input]) #256
        if use_relu:
            GCN_layer = layers.Activation('relu')(GCN_layer)
        else:
            GCN_layer = layers.Activation('tanh')(GCN_layer)
        if use_bn:
            GCN_layer = layers.BatchNormalization()(GCN_layer)
        GCN_layer = layers.Dropout(0.1)(GCN_layer)

        x_drug = layers.GlobalMaxPooling1D()(GCN_layer) if use_GMP else layers.GlobalAveragePooling1D()(GCN_layer)

        ori_gexpr_input = gexpr_inputs[0]
        ori_dim = gexpr_dim[0]
        remaining_dim = gexpr_dim[1:]
        remaining_gexpr_inputs = gexpr_inputs[1:]
        
        concatenated_drug_inputs = layers.Concatenate()(drug_emb_inputs)
        drug_expert_moe, expert_outputs_d = DenseMoE(units=512, n_experts=len(drug_emb_inputs), n_gates=8, expert_input_dims=drug_emb_dim, feature_dim = 512)(concatenated_drug_inputs, x_drug)        

        drug_expert_moe = layers.Dense(100, activation='relu', name='map_drug')(drug_expert_moe)
        concatenated_gexpr_inputs = layers.Concatenate()(remaining_gexpr_inputs)
        
        gexpr_expert_moe, expert_outputs_o = DenseMoE(units=512, n_experts=len(remaining_gexpr_inputs), n_gates=8, expert_input_dims=remaining_dim, feature_dim = ori_dim)(concatenated_gexpr_inputs, ori_gexpr_input)
        gexpr_expert_moe = layers.Dense(100, activation='relu', name='map_gexp')(gexpr_expert_moe)        
        combined_expert_outputs = layers.Concatenate()([x_drug, drug_expert_moe, ori_gexpr_input, gexpr_expert_moe])

        if self.regr:
            hidden_output1 = layers.Dense(1024, activation='relu', name='hidden_1')(combined_expert_outputs)
            bn_1 = layers.BatchNormalization()(hidden_output1)
            dropout_ = layers.Dropout(0.1)(bn_1)

            x = layers.Dense(300,activation = 'tanh')(dropout_)
            x = layers.Dropout(0.1)(x)
            x = layers.Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
            x = layers.Lambda(lambda x: K.expand_dims(x,axis=1))(x)
            x = layers.Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu', padding='valid')(x)
            x = layers.MaxPooling2D(pool_size=(1,2))(x)
            x = layers.Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu', padding='valid')(x)
            x = layers.MaxPooling2D(pool_size=(1,3))(x)
            x = layers.Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu', padding='valid')(x)
            x = layers.MaxPooling2D(pool_size=(1,3))(x)
            x = layers.Dropout(0.1)(x)
            x = layers.Flatten()(x)
            x = layers.Dropout(0.2)(x)
            final_output = layers.Dense(1,name='output')(x)

        else:
            final_output = layers.Dense(1, activation='sigmoid', name='output_final')(combined_expert_outputs)
        outputs_3= [final_output, expert_outputs_d, expert_outputs_o]
        model = models.Model(inputs=[drug_feat_input, drug_adj_input, drug_emb_input0, drug_emb_input1, drug_emb_input2, drug_emb_input3, drug_emb_input4, drug_emb_input5, drug_emb_input6, drug_emb_input7, gexpr_input0, gexpr_input1, gexpr_input2, gexpr_input3, gexpr_input4, gexpr_input5, gexpr_input6, gexpr_input7, gexpr_input8], outputs=outputs_3)
        return model
