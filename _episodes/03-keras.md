---
title: "Introduction to Keras"
teaching: 0
exercises: 0
questions:
- "How do I compose a Deep Neural Network using Keras?"
- "How do I train this network on a dataset"
- "How do I get insight into learning process"
- "How do I measure the performance of the network"
- "What happens to the number of parameters if you add another layer?"

objectives:
- "Explain what a classification task is"
- "Use one-hot-encoding to prepare data for classification in Keras"
- "Describe a fully connected layer"
- "Implement a fully connected layer with Keras"
- "Use Keras to train a small fully connected network on prepared data"
- "Plot the loss curve of the training process"
---

## Iris flower classification
In this first episode we will learn how to create and train a Neural Network using Keras to solve a simple classification task. In this episode we will be using the iris dataset, this is a dataset that was published in 1936 by Ronald Fisher and contains data on three different variants of the iris flower.

Let us take a look at this dataset using a jupyter notebook.

> ## Prerequisite: Start Jupyter Notebook
> Start a jupyter notebook by issueing the following command on a command line. It is probably a
> good idea to first create an empty directory for this course first.
>
> ~~~
> $ mkdir deep-learning-intro
> $ cd deep-learning-intro
> $ jupyter notebook
> ~~~
> {:.language-bash}
{:.prereq}

## Create a notebook
Create a new notebook and call it assignment1.ipynb. We will start by importing a few libraries that will help us get the dataset and visualize it.
~~~
import seaborn as sns # Seaborn is a library for creating nice graphs
import pandas as pd   # Pandas is a data analysis library
from sklearn.datasets import load_iris # sklearn is a machine learning library
                                       # that has many built in sets
~~~
{:.language-python}

### Loading the dataset
We can now load the iris dataset using
~~~
iris = load_iris()
~~~
{:.language-python}

This will give you an object which contains the iris data in the `data` attribute, as well as some metadata information in other attributes such as `target`, `feature_names` and `target_names`

> ## sklearn dataset
>
> Use sklearn to load the dataset and inspect the mentioned attributes.
> * What are the different features called?
> * Are the target classes of the dataset stored as numbers or strings?
> * How many samples does this dataset have?
>
> > ## Solution
> > Printing the `iris.feature_names` attribute should show you an array with the name of the features:
> > ~~~
> > print(iris.feature_names)
> > ~~~
> > {:.language-python}
> > ~~~
> > ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
> > ~~~
> > {:.output}
> > Printing the `iris.target` attribute shows that the targets are stored as a number of 0, 1 or 2
> > ~~~
> > print(iris.target)
> > ~~~
> > {:.language-python}
> > ~~~
> > [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
> >  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
> >  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
> >  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
> >  2 2]
> > ~~~
> > {:.output}
> > To know the number of samples we can look at the shape of the `iris.data` matrix.
> > This shows us that we have 150 samples, and each sample has 4 attributes
> > ~~~
> > print(iris.data.shape)
> > ~~~
> > {:.language-python}
> > ~~~
> > (150, 4)
> > ~~~
> > {:.output}
> {:.solution}
{:.challenge}

### Visualization
Looking at numbers like this usually does not give a very good intuition about the data we are
working with, so let us create a visualization.

The visualization library we will be using is called [Seaborn](https://seaborn.pydata.org/).
Seaborn is a powerful library with many visualizations, but it requires the data to be in a
pandas dataframe, so let us do that now.

> ## Load the dataset into a dataframe
> Load the data into a pandas dataframe including a column with the target names and
> print out the top 5 rows of the pandas dataframe
> > ## Solution
> > ~~~
> > # Create a dataframe from the data
> > df = pd.DataFrame(iris.data, columns=iris.feature_names)
> > # Create an array with the human readable names of the target classes
> > classes = [iris.target_names[x] for x in iris.target]
> > # Add the array as the class
> > df['class'] = classes
> > # df.head() prints the top 5 rows
> > df.head()
> > ~~~
> > {:.language-python}
> >
> > The result should look something like this:
> >
> > |   | sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | class |
> > |---| --- | --- | --- | --- | --- |
> > | 0	| 5.1 | 3.5 | 1.4 | 0.2 | setosa |
> > | 1	| 4.9 | 3.0 | 1.4 | 0.2 | setosa |
> > | 2	| 4.7 | 3.2 | 1.3 | 0.2 | setosa |
> > | 3	| 4.6 | 3.1 | 1.5 | 0.2 | setosa |
> > | 4 | 5.0 | 3.6 | 1.4 | 0.2 | setosa |
> {:.solution}
{:.challenge}

### Pair Plot
One nice visualization for datasets with relatively few attributes is the Pair Plot.
This can be created using `sns.pairplot(...)`. It shows a scatterplot of each attribute plotted against each of the other attributes.
By using the `hue='class'` setting for the pairplot the graphs on the diagonal are layered kernel density estimate plots.

> ## Create the pairplot using Seaborn
> Use the seaborn pairplot function to create a pairplot
> * Is there any class that is easily distinguishable from the others?
> * Which combination of attributes shows the best separation?
>
> > ## Solution
> > ~~~
> > sns.pairplot(df, hue="class")
> > ~~~
> > ![pair plot][pairplot]
> >
> > The plots show that the blue class, setosa is very easily distinguishable from the other two.
> > The other two seem to be most easily separated by a combination of petal length and sepal
> > length.
> {:.solution}
{:.challenge}


[pairplot]: ../fig/pairplot.png "Pair Plot"
{: width="66%"}

{% include links.md %}

