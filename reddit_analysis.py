
# coding: utf-8

# In[2]:


import urllib2
import json
hdr = {'User-Agent': 'osx:r/relationships.single.result:v1.0 (by /u/shubham_kc)'}
url = 'https://www.reddit.com/r/relationships/top/.json?sort=top&t=all&limit=1'
req = urllib2.Request(url, headers=hdr)
text_data = urllib2.urlopen(req).read()

data = json.loads(text_data)

data.values()


# In[3]:


data.values()[1]['children'][0]['data']['title']


# In[7]:


data.values()[1]['children'][0]['data']['score']


# In[8]:


data.values()[1]['children'][0]['data']['name']


# In[13]:


import time
import urllib2
import json

hdr = {'User-Agent': 'osx:r/relationships.multiple.results:v1.0 (by /u/shubham_kc)'}
url = 'https://www.reddit.com/r/relationships/top/.json?sort=top&t=all&limit=100'
req = urllib2.Request(url, headers=hdr)
text_data = urllib2.urlopen(req).read()
data = json.loads(text_data)
data_all = data.values()[1]['children']

while (len(data_all) <= 2000):
    time.sleep(2)
    last = data_all[-1]['data']['name']
    url = 'https://www.reddit.com/r/relationships/top/.json?sort=top&t=all&limit=100&after=%s' % last
    req = urllib2.Request(url, headers=hdr)
    text_data = urllib2.urlopen(req).read()
    data = json.loads(text_data)
    data_all += data.values()[1]['children']


# In[15]:


len(data_all)


# In[16]:


article_title = []
article_flairs = []
article_date = []
article_comments = []
article_score = []

for i in range(0, len(data_all)):
    article_title.append(data_all[i]['data']['title'])
    article_flairs.append(data_all[i]['data']['link_flair_text'])
    article_date.append(data_all[i]['data']['created_utc'])
    article_comments.append(data_all[i]['data']['num_comments'])
    article_score.append(data_all[i]['data']['score'])


# In[17]:


import numpy as np
from pandas import Series, DataFrame
import pandas as pd

rel_df = DataFrame({'Date': article_date,
                    'Title': article_title,
                    'Flair': article_flairs,
                    'Comments': article_comments,
                    'Score': article_score})
rel_df = rel_df[['Date', 'Title', 'Flair', 'Comments', 'Score']]


# In[18]:


rel_df[:5]


# In[19]:


rel_df['Date'] = pd.to_datetime((rel_df['Date'].values*1e9).astype(int))
rel_df[:5]


# In[20]:


import re

replace_value = rel_df['Flair'][1]
rel_df['Flair'] = rel_df['Flair'].replace(replace_value, np.nan)

rel_df['Flair'].isnull().sum()


# In[21]:


cond1 = rel_df['Title'].str.contains(
    '^\[?[a-z!?A-Z ]*UPDATE\]?:?', flags = re.IGNORECASE)
cond2 = rel_df['Flair'].isnull()

rel_df.loc[(cond1 & cond2), 'Flair'] = rel_df.loc[(cond1 & cond2), 'Flair'].replace(np.nan, 'Updates')
rel_df[:5]


# In[22]:


rel_df['Flair'].isnull().sum()


# In[23]:


poster_age_sex = rel_df['Title'].str.extract(
    "((i\'m|i|my|me)\s?(\[|\()(m|f)?(\s|/)?[0-9]{1,2}(\s|/)?([m,f]|male|female)?(\]|\)))", 
        flags = re.IGNORECASE)[0]
poster_age_sex[:5]


# In[24]:


poster_age_sex = poster_age_sex.str.replace("((i\'m|i|my|me))\s?", "", flags = re.IGNORECASE)
poster_age = poster_age_sex.str.extract('([0-9]{1,2})')
poster_sex = poster_age_sex.str.extract('([m,f])', flags = re.IGNORECASE)

rel_df['PosterAge'] = pd.to_numeric(poster_age)
rel_df['PosterSex'] = poster_sex.str.upper()

rel_df[:5]


# In[25]:


rel_df['PosterAge'].isnull().sum()


# In[26]:


rel_df['PosterSex'].isnull().sum()


# In[27]:


rel_df['DayOfWeek'] = rel_df['Date'].dt.dayofweek
days = {0: 'Mon', 1: 'Tues', 2: 'Weds', 3: 'Thurs', 4: 'Fri',
        5: 'Sat', 6: 'Sun'}
rel_df['DayOfWeek'] = rel_df['DayOfWeek'].apply(lambda x: days[x])

rel_df[:5]


# In[28]:


rel_df['DayOfWeek'].isnull().sum()


# In[29]:


rel_df['PosterAge'].describe()


# In[30]:


rel_df['PosterSex'].notnull().sum()


# In[31]:


rel_df['PosterSex'].value_counts()


# In[32]:


100 * rel_df['PosterSex'].value_counts() / rel_df['PosterSex'].notnull().sum()


# In[33]:


rel_df['Flair'].notnull().sum()


# In[34]:


rel_df['Flair'].value_counts()


# In[35]:


100 * rel_df['Flair'].value_counts() / rel_df['Flair'].notnull().sum()


# In[36]:


rel_df['Score'].describe()


# In[37]:


rel_df['Comments'].describe()


# In[38]:


rel_df['DayOfWeek'].value_counts()


# In[39]:


100 * rel_df['DayOfWeek'].value_counts() / rel_df['DayOfWeek'].notnull().sum()


# In[40]:


import numpy as np
from pandas import Series, DataFrame
import pandas as pd

rel_df['PosterAge'].groupby([rel_df['PosterSex']]).median()


# In[41]:


from scipy import stats  

d = {}
for key, value in rel_df.groupby('PosterSex'):
        d['%s' % key] = value['PosterAge']

u_stat, p_val = stats.ranksums(d['M'], d['F'])
print "The test statistic is %1.3f, and the significance level is %1.3f." % (u_stat, p_val / 2)


# In[42]:


sex_flair = pd.crosstab(rel_df['Flair'], rel_df['PosterSex'], 
            rownames = ['Flair'], colnames=['Poster Sex']).ix[
    ['Infidelity', 'Non-Romantic', 'Relationships', 'Updates']]
sex_flair


# In[43]:


sex_flair = pd.crosstab(rel_df['Flair'], rel_df['PosterSex'], 
            rownames = ['Flair'], colnames=['Poster Sex']).loc[
    ['Infidelity', 'Non-Romantic', 'Relationships', 'Updates']]
sex_flair


# In[44]:


sex_flair.apply(lambda c: c/c.sum() * 100, axis = 0)


# In[45]:


chi2, p_val, df, exp_vals = stats.chi2_contingency(sex_flair)
print "The test statistic is %1.3f, and the significance level is %1.3f." % (chi2, p_val)


# In[46]:


score_com_day = rel_df.groupby('DayOfWeek')[
    ['Score', 'Comments']].median()
score_com_day = score_com_day.reindex(['Mon', 'Tues', 'Weds', 'Thurs', 
                                       'Fri', 'Sat', 'Sun'])
score_com_day


# In[47]:


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
get_ipython().magic(u'matplotlib inline')

# Create plot template with two graph boxes
fig = plt.figure(figsize = (720/120, 420/120), dpi = 120)

# Create subplot 1 for median scores
ax1 = fig.add_subplot(1, 2, 1)
score_com_day['Score'].plot(linestyle='solid', color='#FF3721')
ax1.set_title('Score')
plt.xlabel('Day of Week', fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=8)

# Create subplot 2 for median comments
ax2 = fig.add_subplot(1, 2, 2)
score_com_day['Comments'].plot(linestyle='dashed', color='#4271AE')
ax2.set_title('Comments')
plt.xlabel('Day of Week', fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=8)

