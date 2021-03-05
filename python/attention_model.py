
# Model with attention
def create_model(shape=encoded.shape[1]):

    input_seq = keras.Input(shape=(shape,), dtype="int32")
    query_input = input_seq
    value_input = input_seq
    token_embedding = Embedding(1000, 20, input_length=shape)  # of shape [batch_size, Tq, dimension]
    query_embeddings = token_embedding(query_input)  # Value embeddings of shape [batch_size, Tv, dimension]
    value_embeddings = token_embedding(value_input)

    cnn_layer = Conv1D(filters=100, kernel_size=5, padding='same')
    query_seq_encoding = cnn_layer(query_embeddings)  # of shape [batch_size, Tq, filters]
    value_seq_encoding = cnn_layer(value_embeddings)  # of shape [batch_size, Tv, filters]
    query_value_attention_seq = Attention()([query_seq_encoding, value_seq_encoding])

    # Reduce over the sequence axis to produce encodings of shape [batch_size, filters]
    query_encoding = GlobalAveragePooling1D()(query_seq_encoding)
    query_value_attention = GlobalAveragePooling1D()(query_value_attention_seq)

    # Concatenate query and document encodings to produce a DNN input layer.
    input_layer = Concatenate()([query_encoding, query_value_attention])

    dense = layers.Dense(64, activation="relu")
    x = dense(input_layer)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)

    model = keras.Model(inputs=input_layer, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model

model_clf = create_model()
model_clf.fit(trainX, trainy, epochs=6)
model_clf.predict(testX)

cv = StratifiedKFold(n_splits=5)
for i, (train, test) in enumerate(cv.split(features, target)):
    model_clf.fit(features.iloc[train], target.iloc[train], batch_size=300, epochs=5, verbose=2)
    probs = model_clf.predict(features.iloc[test])[:, 1]


    if model == "NN":
        import keras
        from keras.layers import Dense, Dropout, Activation, Conv1D, Attention, Input, Embedding, GlobalAveragePooling1D, Concatenate, GlobalMaxPooling1D
        from keras.layers.convolutional import Conv1D, MaxPooling1D
        from keras import models, layers, Model
        from keras import regularizers, losses, optimizers
        from keras.models import Sequential
        from keras.layers import Dense
        from sklearn.metrics import auc

        # define the model
        def build_model():
            model = Sequential()
            model.add(Dense(120, input_dim=encoded.shape[1], activation='relu'))
            model.add(Dense(40, activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        keras_model = build_model()

        # ~~~~~~~ Train test split ~~~~~~~~~~~~~~~~~~
        trainX, testX, trainy, testy = train_test_split(encoded, target)

        keras_model.fit(trainX, trainy, epochs=20, batch_size=100, verbose=1)

        fpr_keras, tpr_keras, threshold = roc_curve(testy, keras_model.predict(testX))

        auc_keras = auc(fpr_keras, tpr_keras)

        else:
            plt.figure(1)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
            plt.savefig("AUC.png", dpi=110)


def create_model():

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    query_input = tf.keras.Input(shape=(None,), dtype='int32')
    value_input = tf.keras.Input(shape=(None,), dtype='int32')

    # Embedding lookup.
    token_embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=64)
    # Query embeddings of shape [batch_size, Tq, dimension].
    query_embeddings = token_embedding(query_input)
    # Value embeddings of shape [batch_size, Tv, dimension].
    value_embeddings = token_embedding(value_input)

    # CNN layer.
    cnn_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        # Use 'same' padding so outputs have the same shape as inputs.
        padding='same')
    # Query encoding of shape [batch_size, Tq, filters].
    query_seq_encoding = cnn_layer(query_embeddings)
    # Value encoding of shape [batch_size, Tv, filters].
    value_seq_encoding = cnn_layer(value_embeddings)

    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = tf.keras.layers.Attention()(
        [query_seq_encoding, value_seq_encoding])

    # Reduce over the sequence axis to produce encodings of shape
    # [batch_size, filters].
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
        query_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        query_value_attention_seq)

    # Concatenate query and document encodings to produce a DNN input layer.
    input_layer = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])
    inputs = keras.Input(shape=(784,))
    # Add DNN layers, and create Model.
    dense = layers.Dense(64, activation="relu")
    x = dense(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)

    model = keras.Model(inputs=input_layer, outputs=outputs)

    model.summary()

    outputs = layers.Dense(2)(x)
    model = keras.Model(inputs=input_layer, outputs=outputs)
    model.add(Dense(120, input_dim=encoded.shape[1], activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
