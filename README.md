# Documentation

## Note: After Having installed the package ,you may need to additionally run this command in your command prompt,in the same enviornment you previously installed the package 

```
python -m spacy download en
```
## **Import the Module**


```python
from YouGlance import spy
```

    [nltk_data] Downloading package vader_lexicon to
    [nltk_data]     C:\Users\Jash\AppData\Roaming\nltk_data...
    [nltk_data]   Package vader_lexicon is already up-to-date!
    

## Create An Instance of the spy object Note:'HGXBsRYGsYw' is the video Id which we are going to anaylze


```python
obj=spy('HGXBsRYGsYw')
```

## There are a Certain List of Entity Labels that are by default ignored,when Generating the DataFrame.Here we'll grab that List of Entity and then later tweak that List to include and drop Entitiy Labels  according to your Liking


```python
obj.get_unwanted()
```




    ['DATE', 'TIME', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'LANGUAGE', 'LOC']



## Generate DataFrame by calling .generate_df() method which returns the consolidated DataFrame of The Video Transcript


```python
y=obj.generate_df()
```

    Creating DataFrame.....
    Cleaning Text....
    Identifying Entities....
    

## Having a Look at the DataFrame Returned

## Note : The Label Column is the Row wise Identified Entities in cleaned_text column


```python
y.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Start</th>
      <th>cleaned_text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hi everyone I'm Carrie Daugherty filling</td>
      <td>0.530</td>
      <td>hi everyone I'm Carrie Daugherty filling</td>
      <td>[Carrie Daugherty]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>in for Tim cash and this is the IMDB</td>
      <td>3.210</td>
      <td>in for Tim cash and this is the IMDB</td>
      <td>[Tim, IMDB]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>show today's guest is my friend Martin</td>
      <td>5.370</td>
      <td>show today's guest is my friend Martin</td>
      <td>[Martin]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Starr welcome Martin it's weird when you</td>
      <td>7.740</td>
      <td>Starr welcome Martin it's weird when you</td>
      <td>[Starr, Martin]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>clap it sounds like multiple people</td>
      <td>12.059</td>
      <td>clap it sounds like multiple people</td>
      <td>[]</td>
    </tr>
  </tbody>
</table>
</div>



## Get a List of All Unique Entities mentioned in the Video


```python
obj.get_unique_ents()
```




    ['Carrie Daugherty',
     'Tim',
     'IMDB',
     'Martin Starr',
     'Martin',
     'Gilfoyle',
     'Kumail',
     'Farsi',
     'Dinesh',
     'Apple',
     'John',
     'Freaks',
     'Geeks',
     'Model UN Battle Royale']




```python
len(obj.get_unique_ents())
```




    14



## View How Many Times Recognised Entity Label is Referenced 

### This method .show_label_stats() returns a Counter Object 


```python
obj.show_label_stats()
```




    Counter({'PERSON': 8, 'ORG': 11, 'NORP': 1})



## Make a Wildcard Search,which returns all Instances of transcripts closest to your Search(Calculated using Cosine Similarity).Returns a Dictionary of all those related Transcripts


```python
obj.wildcard_search('Silicon valley')
```




    [{'text': 'favorite bit on valley will always be', 'start': 170.94, 'ent': []},
     {'text': 'on Silicon Valley was it nice to switch',
      'start': 223.32,
      'ent': []},
     {'text': 'finale of Silicon Valley yes the final', 'start': 20.88, 'ent': []},
     {'text': 'purposes between Silicon Valley party', 'start': 267.09, 'ent': []}]



## Search By N number of Entities Recognised in the Video,For Filtering pass it as a list of Entities you want to search by


```python
obj.search_by_ents(['IMDB','Tim','Apple'])
```




    [{'text': 'in for Tim cash and this is the IMDB',
      'start': 3.21,
      'ent': ['Tim', 'IMDB']},
     {'text': 'about you when you go to the Apple',
      'start': 128.039,
      'ent': ['Apple']},
     {'text': "don't go to the Apple store for exactly",
      'start': 141.93,
      'ent': ['Apple']},
     {'text': "haven't taken them in to the Apple store",
      'start': 148.05,
      'ent': ['Apple']},
     {'text': 'fridge into the Apple store you know',
      'start': 152.19,
      'ent': ['Apple']},
     {'text': "didn't check my IMDB page there's only",
      'start': 296.82,
      'ent': ['IMDB']}]



## Perform  Auto Topic Modeling to segregate the transcripts into segments ,call the .segregate_topic() method . A tuple of Dataframe and dictionary will be returned


```python
data,d=obj.segregate_topic()
```

## Having a Look At the returned DataFrame


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Start</th>
      <th>cleaned_text</th>
      <th>label</th>
      <th>topic_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hi everyone I'm Carrie Daugherty filling</td>
      <td>0.530</td>
      <td>hi everyone I'm Carrie Daugherty filling</td>
      <td>[Carrie Daugherty]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>in for Tim cash and this is the IMDB</td>
      <td>3.210</td>
      <td>in for Tim cash and this is the IMDB</td>
      <td>[Tim, IMDB]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>show today's guest is my friend Martin</td>
      <td>5.370</td>
      <td>show today's guest is my friend Martin</td>
      <td>[Martin]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Starr welcome Martin it's weird when you</td>
      <td>7.740</td>
      <td>Starr welcome Martin it's weird when you</td>
      <td>[Starr, Martin]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>clap it sounds like multiple people</td>
      <td>12.059</td>
      <td>clap it sounds like multiple people</td>
      <td>[]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Having a Look At the Dictionary Returned,Index is the topic Label and the value is the list of top occuring words belonging to that Label


```python
d
```




    {0: ['did',
      'doing',
      'way',
      'silicon',
      'store',
      'sorry',
      'ensemble',
      'valley',
      'moment',
      'nice',
      'yeah',
      'funny',
      'people',
      'cuz',
      'know',
      'really',
      'don',
      'metal',
      'favorite',
      'think'],
     1: ['lot',
      'quit',
      'say',
      'number',
      'apple',
      'great',
      'saying',
      'yeah',
      'gonna',
      'way',
      'don',
      'nice',
      'know',
      'feel',
      'oh',
      'people',
      've',
      'just',
      'like',
      'good']}



## Unsatisfied with the Number of topic segments,Pass your choice and we will segment the transcript accordingly 


```python
data,d=obj.segregate_topic(3)
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Start</th>
      <th>cleaned_text</th>
      <th>label</th>
      <th>topic_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hi everyone I'm Carrie Daugherty filling</td>
      <td>0.530</td>
      <td>hi everyone I'm Carrie Daugherty filling</td>
      <td>[Carrie Daugherty]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>in for Tim cash and this is the IMDB</td>
      <td>3.210</td>
      <td>in for Tim cash and this is the IMDB</td>
      <td>[Tim, IMDB]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>show today's guest is my friend Martin</td>
      <td>5.370</td>
      <td>show today's guest is my friend Martin</td>
      <td>[Martin]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Starr welcome Martin it's weird when you</td>
      <td>7.740</td>
      <td>Starr welcome Martin it's weird when you</td>
      <td>[Starr, Martin]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>clap it sounds like multiple people</td>
      <td>12.059</td>
      <td>clap it sounds like multiple people</td>
      <td>[]</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
d
```




    {0: ['way',
      'silicon',
      'did',
      'apple',
      'store',
      'sorry',
      'ensemble',
      'valley',
      'moment',
      'nice',
      'yeah',
      'people',
      'funny',
      'cuz',
      'know',
      'really',
      'don',
      'metal',
      'favorite',
      'think'],
     1: ['throw',
      'funny',
      'doing',
      'favorite',
      'right',
      'series',
      'sorry',
      'kind',
      'season',
      'know',
      'store',
      'apple',
      'gonna',
      'quit',
      'yeah',
      'way',
      'just',
      'don',
      'oh',
      'good'],
     2: ['community',
      'dying',
      'gonna',
      'end',
      'set',
      'definitely',
      'thing',
      'fun',
      'number',
      'say',
      'lot',
      'know',
      'great',
      'saying',
      'just',
      'nice',
      'feel',
      'people',
      've',
      'like']}



## Perform Sentiment Analysis,on the Transcript,by calling the sentiment_analysis(),which returns a Dictionary consisting of DataFrame,number of Instances for postivie sentiments and the same for negative and neutral sentiments


```python
k=obj.sentiment_analysis()
```


```python
dataf=k['DataFrame']
```

## Having a look at the DataFrame


```python
dataf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Start</th>
      <th>cleaned_text</th>
      <th>label</th>
      <th>topic_label</th>
      <th>sentiment_dict</th>
      <th>sentiment_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hi everyone I'm Carrie Daugherty filling</td>
      <td>0.530</td>
      <td>hi everyone I'm Carrie Daugherty filling</td>
      <td>[Carrie Daugherty]</td>
      <td>0</td>
      <td>{'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>in for Tim cash and this is the IMDB</td>
      <td>3.210</td>
      <td>in for Tim cash and this is the IMDB</td>
      <td>[Tim, IMDB]</td>
      <td>1</td>
      <td>{'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>2</th>
      <td>show today's guest is my friend Martin</td>
      <td>5.370</td>
      <td>show today's guest is my friend Martin</td>
      <td>[Martin]</td>
      <td>0</td>
      <td>{'neg': 0.0, 'neu': 0.652, 'pos': 0.348, 'comp...</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Starr welcome Martin it's weird when you</td>
      <td>7.740</td>
      <td>Starr welcome Martin it's weird when you</td>
      <td>[Starr, Martin]</td>
      <td>0</td>
      <td>{'neg': 0.175, 'neu': 0.515, 'pos': 0.309, 'co...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>4</th>
      <td>clap it sounds like multiple people</td>
      <td>12.059</td>
      <td>clap it sounds like multiple people</td>
      <td>[]</td>
      <td>2</td>
      <td>{'neg': 0.0, 'neu': 0.667, 'pos': 0.333, 'comp...</td>
      <td>Neutral</td>
    </tr>
  </tbody>
</table>
</div>




```python
positive=k['Positive']
```


```python
positive
```




    38




```python
negative=k['Negative']
```


```python
negative
```




    8




```python
neutral=k['Neutral']
```


```python
neutral
```




    81



## We calculate sentiment course with a Specific Criteria being that if the Compound Score Lies in the range (-0.3,0.4),this however can be changed based on your intution by passing the range you want for neutral Sentiment


```python
d=obj.sentiment_analysis(thresh=(0.6,-0.4))
```


```python
d['DataFrame'].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Start</th>
      <th>cleaned_text</th>
      <th>label</th>
      <th>topic_label</th>
      <th>sentiment_dict</th>
      <th>sentiment_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hi everyone I'm Carrie Daugherty filling</td>
      <td>0.530</td>
      <td>hi everyone I'm Carrie Daugherty filling</td>
      <td>[Carrie Daugherty]</td>
      <td>0</td>
      <td>{'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>in for Tim cash and this is the IMDB</td>
      <td>3.210</td>
      <td>in for Tim cash and this is the IMDB</td>
      <td>[Tim, IMDB]</td>
      <td>1</td>
      <td>{'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>2</th>
      <td>show today's guest is my friend Martin</td>
      <td>5.370</td>
      <td>show today's guest is my friend Martin</td>
      <td>[Martin]</td>
      <td>0</td>
      <td>{'neg': 0.0, 'neu': 0.652, 'pos': 0.348, 'comp...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Starr welcome Martin it's weird when you</td>
      <td>7.740</td>
      <td>Starr welcome Martin it's weird when you</td>
      <td>[Starr, Martin]</td>
      <td>0</td>
      <td>{'neg': 0.175, 'neu': 0.515, 'pos': 0.309, 'co...</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>4</th>
      <td>clap it sounds like multiple people</td>
      <td>12.059</td>
      <td>clap it sounds like multiple people</td>
      <td>[]</td>
      <td>2</td>
      <td>{'neg': 0.0, 'neu': 0.667, 'pos': 0.333, 'comp...</td>
      <td>Neutral</td>
    </tr>
  </tbody>
</table>
</div>




```python
d['Neutral']
```




    109




```python
d['Positive']
```




    11




```python
d['Negative']
```




    7



## If Suppose you want to Include some sentiment labels that were excluded 

### Create new instance of the spy class


```python
p=spy('HGXBsRYGsYw')
```


```python
p.get_unwanted()
```




    ['DATE', 'TIME', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'LANGUAGE', 'LOC']



## Generate DataFrame based on your choice on unwanted labels

### In this Case we will be passing an empty list to be considered for unwanted sentiment labels


```python
p.tweak_unwanted([])
```

## Now the Process remains the Same


```python
l=p.generate_df()
```

    Creating DataFrame.....
    Cleaning Text....
    Identifying Entities....
    


```python
l.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Start</th>
      <th>cleaned_text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hi everyone I'm Carrie Daugherty filling</td>
      <td>0.530</td>
      <td>hi everyone I'm Carrie Daugherty filling</td>
      <td>[Carrie Daugherty]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>in for Tim cash and this is the IMDB</td>
      <td>3.210</td>
      <td>in for Tim cash and this is the IMDB</td>
      <td>[Tim, IMDB]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>show today's guest is my friend Martin</td>
      <td>5.370</td>
      <td>show today's guest is my friend Martin</td>
      <td>[today, Martin]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Starr welcome Martin it's weird when you</td>
      <td>7.740</td>
      <td>Starr welcome Martin it's weird when you</td>
      <td>[Starr, Martin]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>clap it sounds like multiple people</td>
      <td>12.059</td>
      <td>clap it sounds like multiple people</td>
      <td>[]</td>
    </tr>
  </tbody>
</table>
</div>




```python
p.get_unique_ents()
```




    ['Carrie Daugherty',
     'Tim',
     'IMDB',
     'today',
     'Martin Starr',
     'Martin',
     'Silicon Valley',
     'this week',
     'every day',
     'Gilfoyle',
     'Kumail',
     'first',
     'Farsi',
     'two',
     'Dinesh',
     'second',
     'Apple',
     '1',
     'one',
     'John',
     '10 years',
     'a number of years',
     'Freaks',
     'Geeks',
     'Model UN Battle Royale']




```python
len(p.get_unique_ents())
```




    25




```python

```
