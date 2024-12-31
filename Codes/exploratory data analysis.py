# Read CSV files

# Read the training CSV file into a DataFrame
train_df = pd.read_csv('/content/drive/MyDrive/boneage Project/Bone Age Training Set/train.csv')

# Read the testing CSV file into a DataFrame
test_df = pd.read_csv('/content/drive/MyDrive/boneage Project/Bone Age Test Set/Bone age ground truth.csv')

# Appending file extension to 'id' column for both training and testing dataframes

# For the training dataframe
train_df['id'] = train_df['id'].apply(lambda x: str(x) + '.png')

# For the testing dataframe
test_df['Case ID'] = test_df['Case ID'].apply(lambda x: str(x) + '.png')
# Show 5 top row of train DataFrame
train_df.head()
# Show 5 top row of test DataFrame
test_df.head()

# Creating a new column called gender to keep the gender of the child as a string
train_df['gender'] = train_df['male'].apply(lambda x: 'male' if x else 'female')

# Plotting a countplot to visualize the distribution of genders
g = sns.countplot(x=train_df['gender'], palette='pastel')

# Add labels with the count of each category
for p in g.patches:
    g.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Set the plot title and axis labels
plt.title('Distribution of Genders')
plt.xlabel('Gender')
plt.ylabel('Count')

# Show the plot
plt.show()

###

# Create a list to store the rows
rows = []

# Oldest child in the dataset
max_age = train_df['boneage'].max()
rows.append({'Statistic': 'MAX age', 'Value': f'{max_age} months'})

# Youngest child in the dataset
min_age = train_df['boneage'].min()
rows.append({'Statistic': 'MIN age', 'Value': f'{min_age} months'})

# Mean age
mean_bone_age = train_df['boneage'].mean()
rows.append({'Statistic': 'Mean age', 'Value': f'{mean_bone_age}'})

# Median bone age
median_age = train_df['boneage'].median()
rows.append({'Statistic': 'Median age', 'Value': f'{median_age}'})

# Standard deviation of bone age
std_bone_age = train_df['boneage'].std()
rows.append({'Statistic': 'Standard Deviation', 'Value': f'{std_bone_age}'})

# Create the result DataFrame by concatenating the rows
result_df = pd.concat([pd.DataFrame(row, index=[0]) for row in rows], ignore_index=True)

# Calculate the z scores for the bone ages in the training DataFrame
train_df['bone_age_z'] = (train_df['boneage'] - mean_bone_age) / std_bone_age

# Show the result DataFrame
result_df.head()

# Display the updated DataFrame
train_df.head()

###

# Plotting a histogram for bone ages

# Plot a histogram of bone ages
plt.hist(train_df['boneage'], bins=20, color='blue', edgecolor='black')

# Set x-axis and y-axis labels
plt.xlabel('Age in months')
plt.ylabel('Number of children')

# Set the title of the plot
plt.title('Number of children in each age group')

# Add gridlines to the plot
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.show()

###

# Relationship between the number of children and bone age z score

# Plot a histogram of bone age z scores
plt.hist(train_df['bone_age_z'], bins=20, color='gray', edgecolor='black')

# Set x-axis and y-axis labels
plt.xlabel('Bone Age Z Score')
plt.ylabel('Number of Children')

# Set the title of the plot
plt.title('Relationship between Number of Children and Bone Age Z Score')

# Add gridlines to the plot
plt.grid(True, linestyle='--', alpha=0.4)

# Add vertical lines for reference
plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
plt.axvline(1, color='blue', linestyle='--', linewidth=1.5)

# Add text labels for the vertical lines
plt.text(-1.5, 200, 'Below Average', color='red')
plt.text(1.2, 200, 'Above Average', color='blue')

# Show the plot
plt.show()

###

# Distribution of age within each gender

# Filter the train_df DataFrame to separate male and female samples
male = train_df[train_df['gender'] == 'male']
female = train_df[train_df['gender'] == 'female']

# Create a figure with two subplots
fig, ax = plt.subplots(2, 1)

# Plot a histogram of bone age distribution for males
ax[0].hist(male['boneage'], color='blue')
ax[0].set_xlabel('Age in months')
ax[0].set_ylabel('Number of boys')

# Plot a histogram of bone age distribution for females
ax[1].hist(female['boneage'], color='red')
ax[1].set_xlabel('Age in months')
ax[1].set_ylabel('Number of girls')

# Set the figure size
fig.set_size_inches((10, 7))

# Display the figure with the histograms
plt.show()

####

# Distribution of age within each gender

# Filter the train_df DataFrame to separate male and female samples
male = train_df[train_df['gender'] == 'male']
female = train_df[train_df['gender'] == 'female']

# Create a figure with two subplots
fig, ax = plt.subplots(2, 1, figsize=(10, 7))

# Plot a histogram of bone age distribution for males
sns.histplot(data=male, x='boneage', color='blue', edgecolor='black', ax=ax[0])
ax[0].set_xlabel('Age in months')
ax[0].set_ylabel('Number of boys')

# Plot a histogram of bone age distribution for females
sns.histplot(data=female, x='boneage', color='red', edgecolor='black', ax=ax[1])
ax[1].set_xlabel('Age in months')
ax[1].set_ylabel('Number of girls')

# Adjust the layout of subplots to prevent overlap
fig.tight_layout()

# Display the figure with the histograms
plt.show()

# Splitting train DataFrame into training and validation DataFrames
df_train, df_valid = train_test_split(train_df,  test_size = 0.1, random_state = 0)

# Create a figure with subplots to display sample images
fig, axes = plt.subplots(2, 3, figsize=(10, 10))
fig.tight_layout()

# Iterate over a sample of image filenames, bone ages, and genders from the train_df DataFrame
for i, (filename, boneage, gender) in enumerate(train_df[['id', 'boneage', 'gender']].sample(6).values):
    # Read the image using mpimg.imread
    img = mpimg.imread('/content/drive/MyDrive/boneage Project/Bone Age Training Set/boneage-training-dataset/' + filename)

    # Determine the row and column for placing the image in the subplot grid
    row = i // 3
    col = i % 3

    # Display the image in the corresponding subplot
    axes[row, col].imshow(img)
    axes[row, col].set_title('Image Name: {}\nBone Age: {:.1f} years\nGender: {}'.format(filename, boneage / 12, gender))
    axes[row, col].axis('off')

# Show the figure with sample images
plt.show()
