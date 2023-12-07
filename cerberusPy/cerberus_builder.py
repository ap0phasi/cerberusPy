from tensorflow.keras import layers, models
import tensorflow as tf

import matplotlib.pyplot as plt

def ksize(size):
    return max([2,round(size/9)])

def form_head(layer_input, size, feature_len, csize=64):
    head_out = layers.Conv2D(filters=csize, kernel_size=(ksize(size), ksize(feature_len)))(layer_input)
    head_out = layers.LeakyReLU()(head_out)
    head_out = layers.MaxPooling2D(pool_size=(2, 2))(head_out)
    head_out = layers.Flatten()(head_out)
    head_out = layers.Dense(units=csize)(head_out)
    head_out = layers.LeakyReLU()(head_out)

    return head_out


def build_cerberus(training_data, response_data, csize=64):
    train_call = training_data['call']
    train_contexts = [training_data[key] for key in training_data if 'context' in key]
    train_response= training_data['response']
    y_train = response_data
    
    # Extract dimensions from provided data
    context_len = len(train_contexts)
    context_dims = [cont.shape for cont in train_contexts]
    call_size, call_fl = train_call.shape[1], train_call.shape[2]
    res_size, res_fl = train_response.shape[1], train_response.shape[2]

    # Build call head
    in_call = layers.Input(shape=(call_size, call_fl, 1))
    call_head = form_head(in_call, call_size, call_fl, csize)

    # Include last known call
    last_known = in_call[:, call_size-1, :, 0]

    # Build context(s) head
    in_contexts = []
    context_heads = []
    for icl in context_dims:
        input_layer = layers.Input(shape=(icl[1], icl[2], 1))
        in_contexts.append(input_layer)
        context_heads.append(form_head(input_layer, icl[1], icl[2], csize))

    # Build response head
    in_response = layers.Input(shape=(res_size, res_fl, 1))
    response_head = form_head(in_response, res_size, res_fl, csize)

    # Combine heads with necks
    necks = layers.Concatenate(axis=1)([call_head] + context_heads + [response_head])
    necks = layers.Reshape((csize, context_len + 2, 1))(necks)
    necks = layers.Conv2D(filters=csize, kernel_size=(2, 2))(necks)
    necks = layers.LeakyReLU()(necks)
    necks = layers.MaxPooling2D(pool_size=(2, 2))(necks)
    necks = layers.Flatten()(necks)

    # Construct body
    body = layers.Concatenate(axis=1)([last_known, necks])
    body = layers.Dense(units=csize * 4)(body)
    body = layers.LeakyReLU()(body)
    body = layers.Dense(units=csize)(body)
    body = layers.LeakyReLU()(body)
    body = layers.Dense(units=csize // 2)(body)
    body = layers.LeakyReLU()(body)
    body = layers.Dense(units=y_train.shape[1], activation='linear')(body)

    # Assemble the model
    model = models.Model(inputs=[in_call] + in_contexts + [in_response], outputs=body)
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())

    return model

def train_cerberus(model, training_data, response_data, epochs, showplot = False):
    train_call = training_data['call']
    train_contexts = [training_data[key] for key in training_data if 'context' in key]
    train_response = training_data['response']
    y_train = response_data
    
    # Fit the model and save the history
    history = model.fit(
        [train_call] + train_contexts + [train_response],
        y_train,
        epochs=epochs,
        validation_split = 0.1,
        shuffle = True,
        verbose=1  # Change to 0 for no output, 1 for progress bar
    )
    
    if showplot:
        # Plot the training history
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'],
                label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # If there's validation loss, plot that as well
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.legend()

        # Additional metrics can be plotted in a similar fashion
        # ...

        plt.show()

    return model