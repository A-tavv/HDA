# **InceptionV3_relu_64**

# Define the input shape
input_shape = (img_size, img_size, 3)

# Load the MobileNet model with pre-trained weights
base_model = InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet')

# Set the base model to be trainable
base_model.trainable = True

# Create a sequential model
model_InceptionV3 = Sequential()

# Add the MobileNet base model as the first layer in the model
model_InceptionV3.add(base_model)

# Add a global max pooling layer to reduce spatial dimensions
model_InceptionV3.add(GlobalMaxPooling2D())

# Flatten the tensor output from the previous layer
model_InceptionV3.add(Flatten())

# Add a dense layer with 64 units and tanh activation
model_InceptionV3.add(Dense(64, activation='relu'))

# Add a dense layer with 1 unit and linear activation for regression
model_InceptionV3.add(Dense(1, activation='linear'))

# Compile the model with mean squared error (MSE) loss and Adam optimizer
model_InceptionV3.compile(loss='mse', optimizer='adam', metrics=[mae_in_months])

# Print a summary of the model architecture
model_InceptionV3.summary()

###

# Early stopping callback to stop training if the validation loss does not improve
early_stopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0,
                              mode='auto')

# Model checkpoint callback to save the best model based on validation loss
mc = ModelCheckpoint('./content/drive/MyDrive/boneage Project/best_model-InceptionV3_relu.h5',
                     monitor='val_loss',
                     mode='min',
                     save_best_only=True)

# TensorBoard callback to log training progress for visualization
logdir = os.path.join(logs_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

# Learning rate value
lr = 0.004280

# List of callbacks to be used during model training
callbacks = [tensorboard_callback, early_stopping, mc]

# Fit the model to the training data
history = model_InceptionV3.fit(train_generator,
                             steps_per_epoch=395,
                             validation_data=val_generator,
                             validation_steps=1,
                             epochs=15,
                             callbacks=callbacks)
###

# Load the weights of the trained model
model_InceptionV3.load_weights('/content/drive/MyDrive/boneage Project/best_model-InceptionV3_relu.h5')

# Predict the bone age for test data using the loaded model
pred = mean_bone_age + std_bone_age*(model_InceptionV3.predict(val_X, batch_size = 32, verbose = True))

# Convert the true bone age values to months
test_months = mean_bone_age + std_bone_age * val_Y

# Sort the test data indices based on true bone age values
ord_ind = np.argsort(val_Y)

# Select 8 evenly spaced indices from the sorted list
ord_ind = ord_ind[np.linspace(0, len(ord_ind)-1, 8).astype(int)] # take 8 evenly spaced ones

# Create a figure with subplots to display the predicted and true bone ages for selected images
fig, axs = plt.subplots(4, 2, figsize=(15, 30))

# Iterate over the selected indices and corresponding subplots
for (ind, ax) in zip(ord_ind, axs.flatten()):
    ax.imshow(val_X[ind, :,:,0], cmap = 'bone')
    ax.set_title('Age: %fY\nPredicted Age: %fY' % (test_months[ind]/12.0, pred[ind]/12.0))
    ax.axis('off')
fig.savefig('/content/drive/MyDrive/boneage Project/trained_image_predictions_best_model-InceptionV3_relu.png', dpi = 300)

###

# Create a figure and axis object with a size of 7x7 inches
fig, ax = plt.subplots(figsize = (7,7))

# Plot the predicted bone ages against the actual bone ages
ax.plot(test_months, pred, 'r.', label = 'predictions')
ax.plot(test_months, test_months, 'b-', label = 'actual')

# Add a legend to the plot indicating the meaning of the plotted elements
ax.legend(loc = 'upper right')

# Set labels for the x-axis and y-axis
ax.set_xlabel('Actual Age (Months)')
ax.set_ylabel('Predicted Age (Months)')

#Show Plot
fig.show()

###
# Reset the test generator to its initial state
test_generator.reset()

# Print the information about the test generator
print(test_generator)

# Predict the bone ages using the model on the test generator
y_pred = model_InceptionV3.predict(test_generator)

# Flatten the predicted values into a 1D array
predicted = y_pred.flatten()

# Convert the predicted bone ages from normalized values to months
predicted_months = mean_bone_age + std_bone_age * (predicted)

# Get the filenames of the test samples from the test generator
filenames = test_generator.filenames

# Create a DataFrame to store the results with columns for filename and predictions
results = pd.DataFrame({"Filename": filenames, "Predictions": predicted_months})

# Save the results DataFrame to a CSV file named "results.csv"
results.to_csv("/content/drive/MyDrive/boneage Project/model_InceptionV3_relu_64_results.csv", index=False)


# **Xception_Tanh_512**
# Define the input shape
input_shape = (img_size, img_size, 3)

# Load the MobileNet model with pre-trained weights
base_model = Xception(input_shape=input_shape, include_top=False, weights='imagenet')

# Set the base model to be trainable
base_model.trainable = True

# Create a sequential model
model_Xception = Sequential()

# Add the MobileNet base model as the first layer in the model
model_Xception.add(base_model)

# Add a global max pooling layer to reduce spatial dimensions
model_Xception.add(GlobalMaxPooling2D())

# Flatten the tensor output from the previous layer
model_Xception.add(Flatten())

# Add a dense layer with 512 units and tanh activation
model_Xception.add(Dense(512, activation='tanh'))

# Add a dense layer with 1 unit and linear activation for regression
model_Xception.add(Dense(1, activation='linear'))

# Compile the model with mean squared error (MSE) loss and Adam optimizer
model_Xception.compile(loss='mse', optimizer='adam', metrics=[mae_in_months])

# Print a summary of the model architecture
model_Xception.summary()
##
# Early stopping callback to stop training if the validation loss does not improve
early_stopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0,
                              mode='auto')

# Model checkpoint callback to save the best model based on validation loss
mc = ModelCheckpoint('/content/drive/MyDrive/boneage Project/best_model-Xception_tanh_512.h5',
                     monitor='val_loss',
                     mode='min',
                     save_best_only=True)

# TensorBoard callback to log training progress for visualization
logdir = os.path.join(logs_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

# Learning rate value
lr = 0.000099

# List of callbacks to be used during model training
callbacks = [tensorboard_callback, early_stopping, mc]

# Fit the model to the training data
history = model_Xception.fit(train_generator,
                             steps_per_epoch=395,
                             validation_data=val_generator,
                             validation_steps=1,
                             epochs=5,
                             callbacks=callbacks)
##
# Load the weights of the trained model
model_Xception.load_weights('/content/drive/MyDrive/boneage Project/best_model-Xception_tanh_512.h5')

# Predict the bone age for test data using the loaded model
pred = mean_bone_age + std_bone_age*(model_Xception.predict(val_X, batch_size = 32, verbose = True))

# Convert the true bone age values to months
test_months = mean_bone_age + std_bone_age*(val_Y)

# Sort the test data indices based on true bone age values
ord_ind = np.argsort(val_Y)

# Select 2000 evenly spaced indices from the sorted list
ord_ind = ord_ind[np.linspace(0, len(ord_ind)-1, 8).astype(int)] # take 8 evenly spaced ones

# Create a figure with subplots to display the predicted and true bone ages for selected images
fig, axs = plt.subplots(4, 2, figsize = (15, 30))

# Iterate over the selected indices and corresponding subplots
for (ind, ax) in zip(ord_ind, axs.flatten()):
    ax.imshow(val_X[ind, :,:,0], cmap = 'bone')
    ax.set_title('Age: %fY\nPredicted Age: %fY' % (test_months[ind]/12.0, pred[ind]/12.0))
    ax.axis('off')

# Save the figure with predicted and true bone age values as an image
fig.savefig('trained_image_predictions_best_model_Xception_tanh_512.h5.png', dpi = 300)
##
# Create a figure and axis object with a size of 7x7 inches
fig, ax = plt.subplots(figsize = (7,7))

# Plot the predicted bone ages against the actual bone ages
ax.plot(test_months, pred, 'r.', label = 'predictions')
ax.plot(test_months, test_months, 'b-', label = 'actual')

# Add a legend to the plot indicating the meaning of the plotted elements
ax.legend(loc = 'upper right')

# Set labels for the x-axis and y-axis
ax.set_xlabel('Actual Age (Months)')
ax.set_ylabel('Predicted Age (Months)')

#Show Plot
fig.show()
##
# Reset the test generator to its initial state
test_generator.reset()

# Print the information about the test generator
print(test_generator)

# Predict the bone ages using the model on the test generator
y_pred = model_Xception.predict(test_generator)

# Flatten the predicted values into a 1D array
predicted = y_pred.flatten()

# Convert the predicted bone ages from normalized values to months
predicted_months = mean_bone_age + std_bone_age * (predicted)

# Get the filenames of the test samples from the test generator
filenames = test_generator.filenames

# Create a DataFrame to store the results with columns for filename and predictions
results = pd.DataFrame({"Filename": filenames, "Predictions": predicted_months})

# Save the results DataFrame to a CSV file named "results.csv"
results.to_csv("/content/drive/MyDrive/boneage Project/model_Xception_Tanh_512_results.csv", index=False)

# **Xception_Tanh_96**
# Define the input shape
input_shape = (img_size, img_size, 3)

# Load the MobileNet model with pre-trained weights
base_model = Xception(input_shape=input_shape, include_top=False, weights='imagenet')

# Set the base model to be trainable
base_model.trainable = True

# Create a sequential model
model_Xception = Sequential()

# Add the MobileNet base model as the first layer in the model
model_Xception.add(base_model)

# Add a global max pooling layer to reduce spatial dimensions
model_Xception.add(GlobalMaxPooling2D())

# Flatten the tensor output from the previous layer
model_Xception.add(Flatten())

# Add a dense layer with 64 units and tanh activation
model_Xception.add(Dense(96, activation='tanh'))

# Add a dense layer with 1 unit and linear activation for regression
model_Xception.add(Dense(1, activation='linear'))

# Compile the model with mean squared error (MSE) loss and Adam optimizer
model_Xception.compile(loss='mse', optimizer='adam', metrics=[mae_in_months])

# Print a summary of the model architecture
model_Xception.summary()

# Early stopping callback to stop training if the validation loss does not improve
early_stopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0,
                              mode='auto')

# Model checkpoint callback to save the best model based on validation loss
mc = ModelCheckpoint('./content/drive/MyDrive/boneage Project/best_model-Xception_Tanh_96.h5',
                     monitor='val_loss',
                     mode='min',
                     save_best_only=True)

# TensorBoard callback to log training progress for visualization
logdir = os.path.join(logs_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

# Learning rate value
lr = 0.000315

# List of callbacks to be used during model training
callbacks = [tensorboard_callback, early_stopping, mc]

# Fit the model to the training data
history = model_Xception.fit(train_generator,
                             steps_per_epoch=395,
                             validation_data=val_generator,
                             validation_steps=1,
                             epochs=10,
                             callbacks=callbacks)
##
# Load the weights of the trained model
model_Xception.load_weights('/content/drive/MyDrive/boneage Project/best_model-Xception_Tanh_96.h5')

# Predict the bone age for test data using the loaded model
pred = mean_bone_age + std_bone_age*(model_Xception.predict(val_X, batch_size = 32, verbose = True))

# Convert the true bone age values to months
test_months = mean_bone_age + std_bone_age*(val_Y)

# Sort the test data indices based on true bone age values
ord_ind = np.argsort(val_Y)

# Select 8 evenly spaced indices from the sorted list
ord_ind = ord_ind[np.linspace(0, len(ord_ind)-1, 8).astype(int)] # take 8 evenly spaced ones

# Create a figure with subplots to display the predicted and true bone ages for selected images
fig, axs = plt.subplots(4, 2, figsize = (15, 30))

# Iterate over the selected indices and corresponding subplots
for (ind, ax) in zip(ord_ind, axs.flatten()):
    ax.imshow(val_X[ind, :,:,0], cmap = 'bone')
    ax.set_title('Age: %fY\nPredicted Age: %fY' % (test_months[ind]/12.0, pred[ind]/12.0))
    ax.axis('off')

# Save the figure with predicted and true bone age values as an image
fig.savefig('trained_image_predictions_best_model-Xception_Tanh_96.png', dpi = 300)

##
# Create a figure and axis object with a size of 7x7 inches
fig, ax = plt.subplots(figsize = (7,7))

# Plot the predicted bone ages against the actual bone ages
ax.plot(test_months, pred, 'r.', label = 'predictions')
ax.plot(test_months, test_months, 'b-', label = 'actual')

# Add a legend to the plot indicating the meaning of the plotted elements
ax.legend(loc = 'upper right')

# Set labels for the x-axis and y-axis
ax.set_xlabel('Actual Age (Months)')
ax.set_ylabel('Predicted Age (Months)')

#Show Plot
fig.show()
##
# Reset the test generator to its initial state
test_generator.reset()

# Print the information about the test generator
print(test_generator)

# Predict the bone ages using the model on the test generator
y_pred = model_Xception.predict(test_generator)

# Flatten the predicted values into a 1D array
predicted = y_pred.flatten()

# Convert the predicted bone ages from normalized values to months
predicted_months = mean_bone_age + std_bone_age * (predicted)

# Get the filenames of the test samples from the test generator
filenames = test_generator.filenames

# Create a DataFrame to store the results with columns for filename and predictions
results = pd.DataFrame({"Filename": filenames, "Predictions": predicted_months})

# Save the results DataFrame to a CSV file named "results.csv"
results.to_csv("/content/drive/MyDrive/boneage Project/model-Xception_Tanh_96_results.csv", index=False)

### Evaluation
import pandas as pd

# Read the CSV files
Test_CSV = pd.read_csv("/content/drive/MyDrive/boneage Project/Bone Age Test Set/Bone age ground truth.csv")
Xception_Tanh_96_results = pd.read_csv("/content/drive/MyDrive/boneage Project/model-Xception_Tanh_96_results.csv")
InceptionV3_relu_64_results = pd.read_csv("/content/drive/MyDrive/boneage Project/model_InceptionV3_relu_64_results.csv")
Xception_Tanh_512_results = pd.read_csv("/content/drive/MyDrive/boneage Project/model_Xception_Tanh_512_results.csv")

# Modify the 'Filename' column in results dataframes
Xception_Tanh_96_results['Filename'] = Xception_Tanh_96_results['Filename'].str.replace('Test Set Images/', '', regex=False).str.replace('.png', '', regex=False)
InceptionV3_relu_64_results['Filename'] = InceptionV3_relu_64_results['Filename'].str.replace('Test Set Images/', '', regex=False).str.replace('.png', '', regex=False)
Xception_Tanh_512_results['Filename'] = Xception_Tanh_512_results['Filename'].str.replace('Test Set Images/', '', regex=False).str.replace('.png', '', regex=False)

# Modify the 'Case ID' column in Test_CSV dataframe
Test_CSV['Case ID'] = Test_CSV['Case ID'].astype(str)

# Merge the dataframes based on the 'Filename' column
merged_df = pd.merge(Xception_Tanh_96_results, Test_CSV, left_on='Filename', right_on='Case ID')
merged_df = pd.merge(merged_df, InceptionV3_relu_64_results, on='Filename')
merged_df = pd.merge(merged_df, Xception_Tanh_512_results, on='Filename')

# Rename the columns
merged_df.rename(columns={'Predictions_x': 'Xception_Tanh_96_Predictions',
                          'Predictions_y': 'InceptionV3_relu_64_Predictions',
                          'Predictions': 'Xception_Tanh_512_Predictions'},
                 inplace=True)

# Select the desired columns from the merged dataframe
desired_columns = ['Filename', 'Predictions_x', 'Predictions_y', 'Predictions', 'Ground truth bone age (months)']
merged_df = merged_df[desired_columns]

# Rename the columns
merged_df.rename(columns={'Predictions_x': 'Xception_Tanh_96_Predictions',
                          'Predictions_y': 'InceptionV3_relu_64_Predictions',
                          'Predictions': 'Xception_Tanh_512_Predictions'},
                 inplace=True)

merged_df.head()
###
# Calculate the absolute difference between each prediction and the ground truth bone age
merged_df['Xception_Tanh_96_Mae'] = abs(merged_df['Xception_Tanh_96_Predictions'] - merged_df['Ground truth bone age (months)'])
merged_df['InceptionV3_relu_64_Mae'] = abs(merged_df['InceptionV3_relu_64_Predictions'] - merged_df['Ground truth bone age (months)'])
merged_df['Xception_Tanh_512_Mae'] = abs(merged_df['Xception_Tanh_512_Predictions'] - merged_df['Ground truth bone age (months)'])

# Calculate the average accuracy for each model
xception_tanh_96_Mae = merged_df['Xception_Tanh_96_Mae'].mean()
inceptionv3_relu_64_Mae = merged_df['InceptionV3_relu_64_Mae'].mean()
xception_tanh_512_Mae = merged_df['Xception_Tanh_512_Mae'].mean()

# Create a DataFrame to display the results
result_df = pd.DataFrame({
    'Model': ['Xception_Tanh_96', 'InceptionV3_relu_64', 'Xception_Tanh_512'],
    'Mae': [xception_tanh_96_Mae, inceptionv3_relu_64_Mae, xception_tanh_512_Mae]
})

# Print the DataFrame
result_df.head()

###
import matplotlib.pyplot as plt

# Sort the result_df by Mae in descending order
result_df_sorted = result_df.sort_values(by='Mae', ascending=True)

# Create a bar plot with custom colors
colors = ['red', 'green', 'blue']
plt.bar(result_df_sorted['Model'], result_df_sorted['Mae'], color=colors)

# Add labels and title
plt.xlabel('Model')
plt.ylabel('Mae')
plt.title('Comparison of Mae')

# Set custom names for x-axis ticks
plt.xticks(result_df_sorted['Model'], ['Xception 1', 'InceptionV3', 'Xception 2'])

# Show the plot
plt.show()
