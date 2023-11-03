# Scrape Job data and try to predict Max Salary

I scraped the Hiring Cafe website for remote jobs in Poland (where I live) to get job descriptions with salaries. Then I fed that data into Chat GPT and had it spit out some data points about each job description. Then I used Pandas to massage the data into something that could be fed into a Machine Learning estimator (RandomForestClassifier) and was able to predict with around 72% accuracy what the salary should be if given just a job description.

### Improvements I could make to this

I could:
<ul>
    <li>Gather a lot more data from different sites. </li>
    <li>Play around with different estimators from sklearn.</li>
    <li>Gather more data points from the job description using Chat GPT.</li>
    <li>Gather data about the size of the company, startups might pay differently than larger companies.</li>
</ul>


### Notes

I am doing a Machine Learning / AI course on my own time and this is the first thing I decided to try out while going through the course. As I progress through the course I'll do more little projects and maybe come back to this one to get the score higher or use different estimat


```python
%%capture
%pip install algoliasearch
%pip install langchain
%pip install openai

import os
import time
import requests
import json
import html
import re
from typing import Optional


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from pydantic import BaseModel, Field
from algoliasearch.search_client import SearchClient

os.environ['OPENAI_API_KEY'] = '<Open AI Key goes here>'
```


```python
# Some helper functions and pydantic models to store the data in
def unescape_html(text):
    return html.unescape(text)

def strip_html_tags(text: str) -> str:
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text)

class CompanyInfo(BaseModel):
    name: Optional[str]
    description: Optional[str]

class HighlightResult(BaseModel):
    value: Optional[str]
    matchLevel: Optional[str]
    matchedWords: list[str]

class JobData(BaseModel):
    path: Optional[str]
    board_token: Optional[str]
    job_id: Optional[str]
    job_title: Optional[str]
    job_location: Optional[str]
    gptApplied: bool
    source: Optional[str]
    job_description: Optional[str]
    companyInfo: CompanyInfo
    dateFetched: int
    about_company: Optional[str]
    responsibilities: Optional[str]
    requirements: Optional[str]
    tech_stack: Optional[str]
    salary_range: Optional[str]
    job_type: Optional[str]
    applyUrl: Optional[str]
    isSalaryAvailable: bool
    industry: Optional[str]
    role: Optional[str]
    min_years_experience: int
    country: Optional[str]
    isRemote: bool
    addressLocality: Optional[str]
    _geoloc: dict
    objectID: Optional[str]
    _highlightResult: dict[str, HighlightResult]

    @classmethod
    def from_json(cls, json_data):
        # Here you can perform any data transformations you need
        job_description_html = html.unescape(json_data['job_description'])
        stripped_job_description = strip_html_tags(job_description_html)
        json_data['job_description'] = stripped_job_description
        return cls(**json_data)
```

## This is how you can query Hiring Cafe Data


```python
# These aren't my keys, these are just freely available on Hiring Cafe if you look at the network traffic on the site
client = SearchClient.create('8HEMFGMPST', '360c8026d33e372e6b37d18b177f7df5')
index = client.init_index('HiringCafe-V2')

query_list = ["Python", "TypeScript", "Java", "C#"]
filters = ("isRemote:true AND gptApplied:true AND country:'PL' AND job_type:'Full Time' AND "
           "role:'Engineering' AND isSalaryAvailable:true")

hits_per_page = 20
all_hits = []

for query in query_list:   
    page = 0
    total_pages = 20 # Don't do more than 20 pages
  
    while page < total_pages:
        response = index.search(query, {
            'filters': filters,
            'page': page,
            'hitsPerPage': hits_per_page
        })
        
        total_pages = min(response['nbPages'], total_pages)
        hits = response['hits']
        if not hits:
            break
        all_hits.extend(hits)
        page += 1
    
        # Sleep so we don't overload the server with requests
        time.sleep(2)
    
print(f"found {len(all_hits)} hits")
```

    found 129 hits


## Store the job data

We need to store the job data somewhere and also we need to store the data in a way that we can re-run the cell with the chat gpt calls without re-doing all of them if needed.


```python
all_job_data = []
job_ids_done = []
```

## Use Chat GPT to extract data points about the job description


```python
system_prompt = """
Given a job description extract information about the job and put it into a JSON object.

The keys are as follows:
"work_style" - "remote", "hybrid" if the job allows working from home a few days per week, or "onsite" if you're unsure just use "onsite"
"salary_currency" - "PLN", "EUR", "USD"
"salary_minimum_hourly" - if it exists as an hourly salary
"salary_maximum_hourly" - if it exists as an hourly salary
"salary_minimum_monthly" - if it exists as a monthly salary
"salary_maximum_monthly" - if it exists as a monthly salary
"salary_minimum_yearly" - if it exists as a yearly salary
"salary_maximum_yearly" - if it exists as a yearly salary
"title" - the job title
"company_name" - the company name
"level_of_experiene" - the total number of years someone should be experienced
"primary_language" - the primary language for the job
“level_of_experience_in_language” - Given the primary language put the value 1 if asking for 0-1 years of experience with a language or a ‘willingness to learn’ or ‘general understanding of’, 2 if they are asking for 2-4 years in a language or “a proven ability” in the language, 5 if they are asking for 5+ years if they are asking for “mastery”of the language of “proven track record of success in” the language
"level_of_experience_in_platform" - the total number of years someone should be experienced in a particular platform, use the maxmium you can find

For the following keys use true if the requirement is mentioned and false if it is not:
"is_freelance" - true if the job mentions hourly pay and is a freelance job, 0 if the job is a full time long term position
"on_call" - true if it mentions being on-call
"system_architecture" if the job requires architecting large systems
"aws_experience" - true if the job mentions it requires experience with AWS
"gcp_experience" - true if the job mentions it requires experience with GCP or Google Cloud Platform
"azure_experience" - true if the job mentions it requires experience with Azure
"dev_ops" - true if the job mentions you'll have to do devops work
"front_end_work" - true if the job mentions you'll have to do front end work
"backend_work" - true if the job mentions you'll have to do backend work
"full_stack_work" - true if the job mentions you'll have to do full stack work including front-end, backend and devops.
"tech_lead" - true if the job mentions you'll have to lead a team of developers
"project_leadership" - true if the job mentions you'll have to lead a project
"unlimited_time_off" true if it mentions anything about unlimited time off
"has_benefits" true if it mentions you get benefits like budget for self development or conferences, or private medical insurance, or a gym membership
"education" 'NoCollege' if no college is required, 'MS/BS' if a MS/BS in Comp science or Engineering or equivalent experience, 'SpecificHigherEd' if MS/BS/PhD in something other than Computer Science - its a specialized job like something to do with 


Example:
{'work_style': 'remote',
 'salary_currency': 'PLN',
 'salary_minimum_yearly': 184000,
 'salary_maximum_yearly': 249000,
 'title': 'Senior .NET Software Engineer - Matter Admin',
 'company_name': 'Legal technology company',
 'level_of_experience': 5,
 'primary_language': '.NET',
 'level_of_experience_in_language': 5,
 'level_of_experience_in_platform': 5,
 'is_freelance': 0,
 'on_call': 0,
 'system_architecture': 0,
 'aws_experience': 0,
 'gcp_experience': 0,
 'azure_experience': 1,
 'dev_ops': 0,
 'front_end_work': 0,
 'backend_work': 1,
 'full_stack_work': 1,
 'tech_lead': 0,
 'project_leadership': 0,
 'unlimited_time_off': 1,
 'has_benefits': 1,
 'education': 'NoCollege'
}
"""

system_message = SystemMessage(content=system_prompt)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

for i in range(len(all_hits[:2])):
    hit = all_hits[i]
    job_data = JobData.from_json(hit)
    print(f"Doing {i} of {len(all_hits)}")
    
    if job_data.job_id in job_ids_done:
        print("Already done")
        continue

    job_query = f"""
    Title: {job_data.job_title}
    Location: {job_data.job_location}
    Salary: {job_data.salary_range}
    Type: {job_data.job_type}
    Minimum Years of Experience: {job_data.min_years_experience}
    Responsibilities: {job_data.responsibilities}
    Description: {job_data.job_description}
    Requirements: {job_data.requirements}
    Tech Stack: {job_data.tech_stack}
    About company: {job_data.about_company}
    """
    human_message = HumanMessage(content=job_query)
    response = llm.predict_messages(messages=[system_message, human_message])
    raw_job_data = json.loads(response.content)
    
    raw_job_data["job_id"] = job_data.job_id
    all_job_data.append(raw_job_data)

    job_ids_done.append(job_data.job_id)
    
print(f"Finished with {len(all_job_data)} job descriptions parsed")
```

## Load all the data into Pandas


```python
# Use the data we just got from chat gpt
df = pd.DataFrame(all_job_data)

# Or use the data that was saved from a previous run
#df = pd.read_json('all_job_data.json', orient='records', lines=True)
```

## Fill in the Workstyles that might be missing


```python
df["work_style"].fillna("onsite", inplace=True)
```

## Normalize the salaries
We only care about the max salary for this.


```python
allowed_currencies = ["USD", "PLN", "EUR"]
df.loc[~df["salary_currency"].isin(allowed_currencies), "salaryCurrency"] = "PLN"

# df["salaryCurrency"].fillna("PLN", inplace=True)
currency_to_pln = {"PLN": 1, "USD": 4, "EUR": 4.5}
for currency, conversion_rate in currency_to_pln.items():
    conversion_rate = currency_to_pln[currency]
    mask = df["salary_currency"] == currency
    # Max
    df.loc[mask, "salary_maximum_hourly"] *= conversion_rate
    df.loc[mask, "salary_maximum_monthly"] *= conversion_rate
    df.loc[mask, "salary_maximum_yearly"] *= conversion_rate
    # Min
    df.loc[mask, "salary_minimum_hourly"] *= conversion_rate
    df.loc[mask, "salary_minimum_monthly"] *= conversion_rate
    df.loc[mask, "salary_minimum_yearly"] *= conversion_rate

# Max Salary
df["salary_maximum_monthly"].fillna(df["salary_maximum_hourly"] * 40 * 4, inplace=True)
df["salary_maximum_monthly"].fillna(df["salary_maximum_yearly"] / 12, inplace=True)

df.drop(["salary_maximum_hourly", "salary_maximum_yearly"], axis=1, inplace=True)

# Min Salary
df["salary_minimum_monthly"].fillna(df["salary_minimum_hourly"] * 40 * 4, inplace=True)
df["salary_minimum_monthly"].fillna(df["salary_minimum_yearly"] / 12, inplace=True)

df.drop(["salary_minimum_hourly", "salary_minimum_yearly"], axis=1, inplace=True)

df.drop(
    ["salary_currency", "title", "company_name", "job_id", "salaryCurrency" ],
    axis=1,
    inplace=True,
)
```

## Drop anything that might have been a hallucination


```python
columns_to_keep = ['work_style', 'salary_maximum_monthly', 'level_of_experience',
       'primary_language', 'level_of_experience_in_language',
       'level_of_experience_in_platform', 'is_freelance', 'on_call',
       'system_architecture', 'aws_experience', 'gcp_experience',
       'azure_experience', 'dev_ops', 'front_end_work', 'backend_work',
       'full_stack_work', 'tech_lead', 'project_leadership',
       'unlimited_time_off', 'has_benefits', 'education']

columns_to_drop = [col for col in df.columns if col not in columns_to_keep]

df.drop(columns=columns_to_drop, inplace=True)

```

## Drop rows without some of the most important data


```python
df = df.dropna(subset=["salary_maximum_monthly", "primary_language", "level_of_experience_in_language", "level_of_experience_in_platform"])
```

## Convert any columns with floats into int's, sometimes Chat GPT spits out floats instead of ints


```python
# Get columns with float dtype, but not int dtype
float_columns = df.select_dtypes(include=['float64'], exclude=['int64']).columns.tolist()

# Convert those columns to int
for col in float_columns:
    df.loc[:, col] = df[col].astype('int')
```

## One Hot Encode the categorical columns


```python
df = pd.get_dummies(df, columns=['work_style', 'primary_language', 'education'])
```

## Split it into Training and Test Data


```python
X = df.drop(["salary_maximum_monthly"], axis=1)
y = df["salary_maximum_monthly"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Using RandomForestClassifier


```python
n_estimators = 100

clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"n_estimators={n_estimators}, Accuracy:{accuracy_score(y_test, y_pred)}")
```

    n_estimators=100, Accuracy:0.7142857142857143

