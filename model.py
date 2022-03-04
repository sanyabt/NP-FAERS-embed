import tensorflow as tf
from utils import cosine_distance, loss

maxlen=400

def _build_model2(model_type, embedding_dim, num_rnn_node, num_dense_node, num_layer, activation_fn, learning_rate, optimizer, margin):
    input_x = tf.keras.layers.Input(maxlen)
    input_1 = tf.keras.layers.Input(maxlen)
    input_2 = tf.keras.layers.Input(maxlen)
    embedding = tf.keras.layers.Embedding(input_dim=28, output_dim=embedding_dim, mask_zero=True)
    x = embedding(input_x)
    
    if model_type == "lstm":
        x = tf.keras.layers.LSTM(num_rnn_node)(x)
    elif model_type=="gru":
        x = tf.keras.layers.GRU(num_rnn_node)(x)
 
    num = num_dense_node
    for _ in range(num_layer):
        x = tf.keras.layers.Dense(num, activation=activation_fn)(x)
        num /= 2
        
    embedding_network = tf.keras.Model(input_x, x)

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = tf.keras.layers.Lambda(cosine_distance)([tower_1, tower_2])
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(merge_layer)
    contr = tf.keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    
    if optimizer == "Adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer =="RMSprop":                
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    contr.compile(loss=loss(margin= margin), optimizer=opt, metrics=["accuracy"])
    return contr

def embed_model(path):
    model = _build_model2("lstm", 256, 512, 256, 1, "tanh", 2e-4, "Adam", 0.8)
    model.load_weights(path)
    return model.layers[2] 