import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from collections import OrderedDict

# Ignore FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set plot style
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18
plt.rcParams['patch.edgecolor'] = 'k'

# Increase the maximum number of displayed columns
pd.options.display.max_columns = 150

# Read in data
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# Explore the data
train.head()
test.info()
train.info()

# Visualize the count of unique values in integer columns
train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(
    color='blue', figsize=(8, 6), edgecolor='k', linewidth=2)
plt.xlabel('Number of Unique Values')
plt.ylabel('Count')
plt.title('Count of Unique Values in Integer Columns')

# Color mapping and poverty levels
colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})

# Explore float columns using KDE plots
plt.figure(figsize=(20, 16))
for i, col in enumerate(train.select_dtypes('float')):
    ax = plt.subplot(4, 2, i + 1)
    for poverty_level, color in colors.items():
        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(),
                    ax=ax, color=color, label=poverty_mapping[poverty_level])
    plt.title(f'{col.capitalize()} Distribution')
    plt.xlabel(f'{col}')
    plt.ylabel('Density')
plt.subplots_adjust(top=2)

# Mapping for 'yes' and 'no' values to 1 and 0 in certain columns
mapping = {"yes": 1, "no": 0}

# Apply the mapping to both train and test data
for df in [train, test]:
    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)
    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)
    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)

# Visualize the distribution of 'dependency', 'edjefa', and 'edjefe' columns
plt.figure(figsize=(16, 12))
for i, col in enumerate(['dependency', 'edjefa', 'edjefe']):
    ax = plt.subplot(3, 1, i + 1)
    for poverty_level, color in colors.items():
        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(),
                    ax=ax, color=color, label=poverty_mapping[poverty_level])
    plt.title(f'{col.capitalize()} Distribution')
    plt.xlabel(f'{col}')
    plt.ylabel('Density')
plt.subplots_adjust(top=2)

# Add null 'Target' column to the test dataset
test['Target'] = np.nan
data = pd.concat([train, test], ignore_index=True)

# Identify heads of households
heads = data.loc[data['parentesco1'] == 1].copy()

# Labels for training
train_labels = data.loc[(data['Target'].notnull()) & (data['parentesco1'] == 1), ['Target', 'idhogar']]

# Visualize the distribution of poverty levels
label_counts = train_labels['Target'].value_counts().sort_index()
label_counts.plot.bar(figsize=(8, 6), color=colors.values(), edgecolor='k', linewidth=2)
plt.xlabel('Poverty Level')
plt.ylabel('Count')
plt.xticks([x - 1 for x in poverty_mapping.keys()], list(poverty_mapping.values()), rotation=60)
plt.title('Poverty Level Breakdown')

# Identify households where family members do not have the same target
all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))

# Handle households without a head
households_leader = train.groupby('idhogar')['parentesco1'].sum()
households_no_head = train.loc[train['idhogar'].isin(households_leader[households_leader == 0].index), :]
print('There are {} households without a head.'.format(households_no_head['idhogar'].nunique()))

# Check households without a head and different labels
households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
print('{} Households with no head have different labels.'.format(sum(households_no_head_equal == False)))

# Set the correct target label for each household
for household in not_equal.index:
    true_target = int(train[(train['idhogar'] == household) & (train['parentesco1'] == 1.0)]['Target'].iloc[0])
    train.loc[train['idhogar'] == household, 'Target'] = true_target

# Check again for households where family members do not have the same target
all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))

# Calculate and visualize missing values
missing = pd.DataFrame(data.isnull().sum()).rename(columns={0: 'total'})
missing['percent'] = missing['total'] / len(data)
missing.sort_values('percent', ascending=False).head(10).drop('Target')

# Define a function to plot counts of two categoricals
def plot_categoricals(x, y, data, annotate=True):
    raw_counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize=False))
    raw_counts = raw_counts.rename(columns={x: 'raw_count'})
    counts = pd.DataFrame(data.groupby(y)[x].value_counts(normalize=True))
    counts = counts.rename(columns={x: 'normalized_count'}).reset_index()
    counts['percent'] = 100 * counts['normalized_count']
    counts['raw_count'] = list(raw_counts['raw_count'])

    plt.figure(figsize=(14, 10))
    plt.scatter(counts[x], counts[y], edgecolor='k', color='lightgreen',
                s=100 * np.sqrt(counts['raw_count']), marker='o',
                alpha=0.6, linewidth=1.5)

    if annotate:
        for i, row in counts.iterrows():
            plt.annotate(xy=(row[x] - (1 / counts[x].nunique()), row[y] - (0.15 / counts[y].nunique())),
                         color='navy', s=f"{round(row['percent'], 1)}%")

    plt.yticks(counts[y].unique())
    plt.xticks(counts[x].unique())

    sqr_min = int(np.sqrt(raw_counts['raw_count'].min()))
    sqr_max = int(np.sqrt(raw_counts['raw_count'].max()))
    msizes = list(range(sqr_min, sqr_max, int((sqr_max - sqr_min) / 5)))
    markers = []

    for size in msizes:
        markers.append(plt.scatter([], [], s=100 * size,
                                   label=f'{int(round(np.square(size) / 100) * 100)}',
                                   color='lightgreen', alpha=0.6, edgecolor='k', linewidth=1.5))

    plt.legend(handles=markers, title='Counts', labelspacing=3, handletextpad=2,
               fontsize=16, loc=(1.10, 0.19))

    plt.annotate(f'* Size represents the raw count while % is for a given y value.',
                 xy=(0, 1), xycoords='figure points', size=10)

    plt.xlim((counts[x].min() - (6 / counts[x].nunique()), counts[x].max() + (6 / counts[x].nunique())))
    plt.ylim((counts[y].min() - (4 / counts[y].nunique()), counts[y].max() + (4 / counts[y].nunique())))
    plt.grid(None)
    plt.xlabel(f"{x}")
    plt.ylabel(f"{y}")
    plt.title(f"{y} vs {x}")

# Example: Displaying the categorical strip plot
category1 = 'rez_esc'
category2 = 'Target'

# Create a categorical strip plot
sns.catplot(x=category1, y=category2, data=train, kind='strip')
plt.show()
