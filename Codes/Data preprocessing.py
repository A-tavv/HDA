# Set the image size
img_size = 224

# Set the batch size for data generators
batch_size = 32

# Define the data generators
train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

 # Only rescale validation data
val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

# Generate train data from the DataFrame using flow_from_dataframe
train_generator = train_datagen.flow_from_dataframe(
    dataframe = df_train,
    directory='/content/drive/MyDrive/boneage Project/Bone Age Training Set/boneage-training-dataset',
    x_col='id',
    y_col='bone_age_z',
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode='other',
    color_mode='rgb',
    target_size=(img_size, img_size)
)

# Generate validation data from the DataFrame using flow_from_dataframe
val_generator = val_datagen.flow_from_dataframe(
    dataframe=df_valid,
    directory='/content/drive/MyDrive/boneage Project/Bone Age Training Set/boneage-training-dataset',
    x_col='id',
    y_col='bone_age_z',
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode='other',
    flip_vertical = True,
    color_mode='rgb',
    target_size=(img_size, img_size)
)

# Load and prepare test data
test_generator = val_datagen.flow_from_directory(
    directory='/content/drive/MyDrive/boneage Project/Bone Age Test Set',
    target_size=(img_size, img_size),
    shuffle = False,
    batch_size=1,
    color_mode='rgb',
    class_mode=None
)

val_X, val_Y = next(val_datagen.flow_from_dataframe(
                            df_valid,
                            directory = '/content/drive/MyDrive/boneage Project/Bone Age Training Set/boneage-training-dataset',
                            x_col = 'id',
                            y_col = 'bone_age_z',
                            target_size = (img_size, img_size),
                            batch_size = 1404,
                            class_mode = 'other'
                            ))

### **Hyper Parameter Tuning**

# Define the model building function
def build_model(hp):
    # Choose the hyperparameters
    architecture = hp.Choice('architecture', ['Xception', 'MobileNet', 'InceptionV3'])
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    activation = hp.Choice('activation', ['relu', 'tanh'])
    dropout = hp.Boolean('dropout')
    lr = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')

    # Set the input shape based on the selected architecture
    if architecture == 'Xception':
        with tf.device('/device:GPU:0'):
            base_model = Xception(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet')
    elif architecture == 'MobileNet':
        with tf.device('/device:GPU:0'):
            base_model = MobileNet(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet')
    elif architecture == 'InceptionV3':
        with tf.device('/device:GPU:0'):
            base_model = InceptionV3(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet')

    model = Sequential()
    model.add(base_model)
    model.add(Dense(units, activation=activation))
    if dropout:
        model.add(Dropout(0.25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer=Adam(lr), loss='mse', metrics=['mae'])

    return model

# Define the tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=10,
    factor=3,
    directory='/content/drive/MyDrive/boneage Project/tuner',
    project_name='bone_age'
)

# Perform the search with the tuner
tuner.search(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3), TensorBoard(log_dir='/content/drive/MyDrive/boneage Project/logs')]
)

# Get the best hyperparameters for each architecture
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=3)

# Create a DataFrame to store the results
results_df = pd.DataFrame(columns=['Architecture', 'Units', 'Activation', 'Dropout', 'Learning Rate'])

# Iterate over the best hyperparameters and add them to the DataFrame
for i, hyperparameters in enumerate(best_hyperparameters):
    results_df.loc[i] = [
        hyperparameters['architecture'],
        hyperparameters['units'],
        hyperparameters['activation'],
        hyperparameters['dropout'],
        hyperparameters['learning_rate']
    ]

# Show the results DataFrame
results_df.head()


###Tensorboard

# Function to calculate mean absolute error (MAE) in months

def mae_in_months(x_p, y_p):
    return mean_absolute_error((std_bone_age*x_p + mean_bone_age), (std_bone_age*y_p + mean_bone_age))

# Load the TensorBoard notebook extension
%load_ext tensorboard

# Specify the directory path for TensorBoard logs
logs_dir = './logs'

# Start TensorBoard server and visualize logs in the specified directory
%tensorboard --logdir {logs_dir}
