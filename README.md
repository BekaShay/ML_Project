# Simple linear regression

<div style="display:none" id="MSE">W3sicXVlc3Rpb24iOiAiSW4gdGhlIGxpbmVhciByZWdyZXNzaW9uIG1vZGVsICAkeT1heCtiJCwgd2hhdCBkb2VzIHRoZSBwYXJ0aWFsIGRlcml2YXRpdmUgJHtcXHBhcnRpYWwgTVNFfS97XFxwYXJ0aWFsIGF9JCByZXByZXNlbnQ/IiwgInR5cGUiOiAibXVsdGlwbGVfY2hvaWNlIiwgImFuc3dlcnMiOiBbeyJhbnN3ZXIiOiAiVGhlIHNsb3BlIG9mIHRoZSBsaW5lIiwgImNvcnJlY3QiOiBmYWxzZSwgImZlZWRiYWNrIjogIkluY29ycmVjdC4uIFRyeSBhZ2Fpbi4ifSwgeyJhbnN3ZXIiOiAiVGhlIHktaW50ZXJjZXB0IiwgImNvcnJlY3QiOiBmYWxzZSwgImZlZWRiYWNrIjogIkluY29ycmVjdC4uIFRyeSBhZ2Fpbi4ifSwgeyJhbnN3ZXIiOiAiVGhlIHJhdGUgb2YgY2hhbmdlIG9mIE1TRSB3aXRoIHJlc3BlY3QgdG8gYSIsICJjb3JyZWN0IjogdHJ1ZSwgImZlZWRiYWNrIjogIkl0IGlzIGNvcnJlY3QhIn0sIHsiYW5zd2VyIjogIlRoZSBvcHRpbWFsIHZhbHVlIG9mIGEiLCAiY29ycmVjdCI6IGZhbHNlLCAiZmVlZGJhY2siOiAiSW5jb3JyZWN0Li4gVHJ5IGFnYWluLiJ9XX1d</div>

## Team Members

| Name          | Role                                          | Day    |
|---------------| ----------------------------------------------|--------|
| Sanzhar       | Technical writer                              |Friday  |
| Maksat        | Author of executable content                  |Thursday|
| Zhaisan       | Designer of interactive plots                 |Friday  |
| Nazym         | Project Manager                               |Thursday|
| Tair          | Designer of quizzes                           |Friday  |

## Cool book about ML

Read [here](https://drive.google.com/file/d/11qrBFxWdK171PB110P_PNu3jQL0Rtsl9/view?usp=sharing)


<p align="center">
  <img src="../_static/images/mai-sakurajima-holding-hands-on-machine-learning.jpg" />
</p>

<div style="display:none" id="Intro">W3siZm9yU2FuemgiOiAiZW5kIEludHJvZHVjdGlvbiIsICJxdWVzdGlvbiI6ICJXaGF0IGlzIGxpbmVhciByZWdyZXNzaW9uJ3MgdGFyZ2V0IGlmIGJpYXMgaXMgMTQ5LCBzbG9wZSBpcyA1MywgYW5kIHByZWRpY3RvciBpcyA1PyIsICJ0eXBlIjogIm51bWVyaWMiLCAicHJlY2lzaW9uIjogMiwgImFuc3dlcnMiOiBbeyJ0eXBlIjogInZhbHVlIiwgInZhbHVlIjogNDE0LCAiY29ycmVjdCI6IHRydWUsICJmZWVkYmFjayI6ICJJdCBpcyBjb3JyZWN0ISJ9LCB7InR5cGUiOiAicmFuZ2UiLCAicmFuZ2UiOiBbLTEwMDAwMDAwMCwgMF0sICJjb3JyZWN0IjogZmFsc2UsICJmZWVkYmFjayI6ICJDb3JyZWN0IGFuc3dlciBpcyA0MTQuIEJ5IGZvcm11bGEsIDE0OSArIDUzKjUgPSA0MTQifV19XQ==</div>

<div style="display:none" id="Dummy">W3sicXVlc3Rpb24iOiAiV2hhdCBpcyB0aGUgbWFpbiBkaWZmZXJlbmNlIGJldHdlZW4gYSBkdW1teSBtb2RlbCBhbmQgc2ltcGxlIGxpbmVhciByZWdyZXNzaW9uPyIsICJ0eXBlIjogIm11bHRpcGxlX2Nob2ljZSIsICJhbnN3ZXJzIjogW3siYW5zd2VyIjogIkFjY3VyYWN5IG9mIHByZWRpY3Rpb25zLiIsICJjb3JyZWN0IjogZmFsc2UsICJmZWVkYmFjayI6ICJJbmNvcnJlY3QsIGFzIHRoZSBkdW1teSBtb2RlbCBhbHdheXMgdXNlcyB0aGUgc2FtZSBzdGF0aWMgdmFsdWUgZm9yIGFsbCBwcmVkaWN0aW9ucy4ifSwgeyJhbnN3ZXIiOiAiVXNlIG9mIHByZWRpY3RvcnMuIiwgImNvcnJlY3QiOiB0cnVlLCAiZmVlZGJhY2siOiAiQ29ycmVjdCwgYXMgdGhlIGR1bW15IG1vZGVsIGRvZXMgbm90IHVzZSBwcmVkaWN0b3JzIGZvciBmb3JlY2FzdGluZywgd2hpbGUgc2ltcGxlIGxpbmVhciByZWdyZXNzaW9uIGRvZXMuIn0sIHsiYW5zd2VyIjogIk1hdGhlbWF0aWNhbCBhbGdvcml0aG1zLiIsICJjb3JyZWN0IjogZmFsc2UsICJmZWVkYmFjayI6ICJJbmNvcnJlY3QsIGJvdGggbW9kZWxzIGRpZmZlciBidXQgbm90IGZ1bmRhbWVudGFsbHkgaW4gdGVybXMgb2YgYWxnb3JpdGhtcy4ifSwgeyJhbnN3ZXIiOiAiTmFtZSBvZiB0aGUgbWV0aG9kLiIsICJjb3JyZWN0IjogZmFsc2UsICJmZWVkYmFjayI6ICJJbmNvcnJlY3QsIHRoZSBuYW1lcyBkbyBub3QgZGV0ZXJtaW5lIHRoZSBtYWluIGZ1bmN0aW9uYWwgZGlmZmVyZW5jZSBiZXR3ZWVuIHRoZSBtb2RlbHMuIn1dfV0=</div>

## Introduction

Simple linear regression is a specific case of linear regression where there is only one independent variable influencing the dependent variable. In simple linear regression, the goal is to find the best-fitting line (linear relationship) that minimizes the sum of squared errors (differences between the observed values and the values predicted by the line).
    
    
In case of one feature ($d = 1$) linear regression is written as

```{math}
    :label: simple-lin-reg
    y = a x + b.
```
    
The parameters of this model $θ = (a, b)$, where $b$ is **intercept** (or **bias**), $a$ **slope**.

<center>
<video width="640" height="360" 
       src="../_static/videos/LinReg.mp4"  
       controls>
</video>
</center>

The [feature matrix](https://fedmug.github.io/kbtu-ml-book/intro/data.html#feature-matrix) here has only one column, denote it $x$ and let $y$ be the vector of corresponding labels.
    
Also denote
    
* $\overline {\boldsymbol x} = \frac 1n \sum\limits_{i=1}^n x_i$ — sample mean of predictors;
* $\overline {\boldsymbol y} = \frac 1n \sum\limits_{i=1}^n y_i$ — sample mean of targets.


```python

```


```python
import matplotlib.pyplot as plt
font = {'family' : 'serif',
        'size'   : 17,
        'weight' : 'normal'
       }

plt.rc('font', **font)

plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('axes', titlesize=18)
# plt.rc('legend', fontsize=18)
plt.rc('text.latex', preamble=r'\usepackage[utf8]{inputenc}')
plt.rc('text.latex', preamble=r'\usepackage[T2A]{fontenc}')
plt.rc('text.latex', preamble=r'\usepackage[russian]{babel}')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
```


```python
pip install jupyterquiz
```

    Requirement already satisfied: jupyterquiz in c:\programdata\anaconda3\lib\site-packages (2.6.3)
    Note: you may need to restart the kernel to use updated packages.
    

<div style="display:none" id="Boston">W3sicXVlc3Rpb24iOiAiSWYgdGhlIGNvZWZmaWNpZW50IG9mIGRldGVybWluYXRpb24gJFJeMiQgZm9yIGEgc2ltcGxlIGxpbmVhciByZWdyZXNzaW9uIG1vZGVsIGlzICQwLjg1JCwgd2hhdCBwZXJjZW50YWdlIG9mIHRoZSB2YXJpYW5jZSBpbiB0aGUgbWVkaWFuIGhvdXNlIHZhbHVlcyBpcyBub3QgZXhwbGFpbmVkIGJ5IHRoZSBtb2RlbD8iLCAidHlwZSI6ICJudW1lcmljIiwgInByZWNpc2lvbiI6IDIsICJhbnN3ZXJzIjogW3sidHlwZSI6ICJ2YWx1ZSIsICJ2YWx1ZSI6IDE1LCAiY29ycmVjdCI6IHRydWUsICJmZWVkYmFjayI6ICJJdCBpcyBjb3JyZWN0ISJ9LCB7InR5cGUiOiAidmFsdWUiLCAidmFsdWUiOiAwLjE1LCAiY29ycmVjdCI6IHRydWUsICJmZWVkYmFjayI6ICJJdCBpcyBjb3JyZWN0ISJ9LCB7InR5cGUiOiAicmFuZ2UiLCAicmFuZ2UiOiBbLTEwMDAwMDAwMCwgMF0sICJjb3JyZWN0IjogZmFsc2UsICJmZWVkYmFjayI6ICJJbmNvcnJlY3QuLiBUcnkgYWdhaW4uIn1dfV0=</div>


```python
from jupyterquiz import display_quiz
display_quiz("#Intro")
```


<div id="rWhwxcwmmhhT" data-shufflequestions="False"
               data-shuffleanswers="True"
               data-preserveresponses="false"
               data-numquestions="1000000"
               data-maxwidth="600"
               style="border-radius: 10px; text-align: left"> <style>
#rWhwxcwmmhhT {
   --jq-multiple-choice-bg: #6f78ffff;
   --jq-mc-button-bg: #fafafa;
   --jq-mc-button-border: #e0e0e0e0;
   --jq-mc-button-inset-shadow: #555555;
   --jq-many-choice-bg: #f75c03ff;
   --jq-numeric-bg: #392061ff;
   --jq-numeric-input-bg: #c0c0c0;
   --jq-numeric-input-label: #101010;
   --jq-numeric-input-shadow: #999999;
   --jq-incorrect-color: #c80202;
   --jq-correct-color: #009113;
   --jq-text-color: #fafafa;
}

.Quiz {
    max-width: 600px;
    margin-top: 15px;
    margin-left: auto;
    margin-right: auto;
    margin-bottom: 15px;
    padding-bottom: 4px;
    padding-top: 4px;
    line-height: 1.1;
    font-size: 16pt;
    border-radius: inherit;
}

.QuizCode {
    font-size: 14pt;
    margin-top: 10px;
    margin-left: 20px;
    margin-right: 20px;
}

.QuizCode>pre {
    padding: 4px;
}

.Answer {
    margin: 10px 0;
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-gap: 10px;
    border-radius: inherit;
}

.Feedback {
    font-size: 16pt;
    text-align: center;
    min-height: 2em;
}

.Input {
    align: left;
    font-size: 20pt;
}

.Input-text {
    display: block;
    margin: 10px;
    color: inherit;
    width: 140px;
    background-color: var(--jq-numeric-input-bg);
    color: var(--jq-text-color);
    padding: 5px;
    padding-left: 10px;
    font-family: inherit;
    font-size: 20px;
    font-weight: inherit;
    line-height: 20pt;
    border: none;
    border-radius: 0.2rem;
    transition: box-shadow 0.1s);
}

.Input-text:focus {
    outline: none;
    background-color: var(--jq-numeric-input-bg);
    box-shadow: 0.6rem 0.8rem 1.4rem -0.5rem var(--jq-numeric-input-shadow);
}

.MCButton {
    background: var(--jq-mc-button-bg);
    border: 1px solid var(--jq-mc-button-border);
    border-radius: inherit;
    padding: 10px;
    font-size: 16px;
    cursor: pointer;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
}

.MCButton p {
    color: inherit;
}

.MultipleChoiceQn {
    padding: 10px;
    background: var(--jq-multiple-choice-bg);
    color: var(--jq-text-color);
    border-radius: inherit;
}

.ManyChoiceQn {
    padding: 10px;
    background: var(--jq-many-choice-bg);
    color: var(--jq-text-color);
    border-radius: inherit;
}

.NumericQn {
    padding: 10px;
    background: var(--jq-numeric-bg);
    color: var(--jq-text-color);
    border-radius: inherit;
}

.NumericQn p {
    color: inherit;
}

.InpLabel {
    line-height: 34px;
    float: left;
    margin-right: 10px;
    color: var(--jq-numeric-input-label);
    font-size: 15pt;
}

.incorrect {
    color: var(--jq-incorrect-color);
}

.correct {
    color: var(--jq-correct-color);
}

.correctButton {
    /*
    background: var(--jq-correct-color);
   */
    animation: correct-anim 0.6s ease;
    animation-fill-mode: forwards;
    color: var(--jq-text-color);
    box-shadow: inset 0px 0px 5px var(--jq-mc-button-inset-shadow);
    outline: none;
}

.incorrectButton {
    animation: incorrect-anim 0.8s ease;
    animation-fill-mode: forwards;
    color: var(--jq-text-color);
    box-shadow: inset 0px 0px 5px var(--jq-mc-button-inset-shadow);
    outline: none;
}

@keyframes incorrect-anim {
    100% {
        background-color: var(--jq-incorrect-color);
    }
}

@keyframes correct-anim {
    100% {
        background-color: var(--jq-correct-color);
    }
}
</style>



    <IPython.core.display.Javascript object>



```python
# !pip install pandas
# !pip install numpy
# !pip install plotly
!pip install scikit-learn
```

    Requirement already satisfied: scikit-learn in c:\programdata\anaconda3\lib\site-packages (1.3.2)
    Requirement already satisfied: numpy<2.0,>=1.17.3 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.26.2)
    Requirement already satisfied: scipy>=1.5.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.11.4)
    Requirement already satisfied: joblib>=1.1.1 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.3.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (3.2.0)
    

Below you can see the application of linear regression for prediction and analysis in different areas.


```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression


df = pd.read_csv('assets/Real-estate.csv')
salary_data = pd.read_csv('assets/Salary_dataset.csv')

insurance_data = pd.read_csv('assets/insurance.csv')
emissions_data = pd.read_csv('assets/CO2_Emissions_Canada.csv')

X_real_estate = df[['X2 house age']]
y_real_estate = df['Y house price of unit area']

# Real estate model training
model_real_estate = LinearRegression()
model_real_estate.fit(X_real_estate, y_real_estate)

# Predictions for real estate
predictions_real_estate = model_real_estate.predict(X_real_estate)

# For salary
X_salary = salary_data[['YearsExperience']]
y_salary = salary_data['Salary']

# Training the Salary Model
model_salary = LinearRegression()
model_salary.fit(X_salary, y_salary)

# Salary Predictions
predictions_salary = model_salary.predict(X_salary)

X_insurance_bmi = insurance_data[['bmi']].values  # Ensure 'bmi' is the correct column name
y_insurance_charges = insurance_data['charges'].values  # Ensure 'charges' is the correct column name

model_insurance_bmi = LinearRegression()
model_insurance_bmi.fit(X_insurance_bmi, y_insurance_charges)

# Make predictions
predictions_insurance_bmi = model_insurance_bmi.predict(X_insurance_bmi)


# Function to create the linear equation text
def linear_equation_text(model):
    a = model.coef_[0]
    b = model.intercept_
    return f"y = {a:.2f}x + {b:.2f}"

def create_annotation(model, x_data, y_data, line_color, feature_name):
    equation_text = linear_equation_text(model)

    return {
        "x": 1,
        "y": 1,
        "xref": "paper",
        "yref": "paper",
        "text": equation_text,
        "showarrow": False,
        "font": {
            "size": 16,
            "color": "white",
            "family": "Arial, bold"
        },
        "align": "right",
        "bgcolor": line_color,  # Match the line color
        "bordercolor": "black",
        "borderwidth": 2,
        "borderpad": 4,
        "xanchor": "right", 
        "yanchor": "top",
        "xshift": -35,
        "yshift": -10,
    }

# Define your line colors for each dataset
line_colors = {
    'real_estate': 'red',
    'salary': 'orange',
    'insurance': 'green',
}

# Generate annotations with matched colors
real_estate_annotation = create_annotation(model_real_estate, df['X2 house age'], df['Y house price of unit area'], line_colors['real_estate'], 'X2 house age')
salary_annotation = create_annotation(model_salary, salary_data['YearsExperience'], salary_data['Salary'], line_colors['salary'], 'YearsExperience')
insurance_annotation = create_annotation(model_insurance_bmi, insurance_data['bmi'], insurance_data['charges'], line_colors['insurance'], 'bmi')

# Create your Plotly figure and add traces for each dataset
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Real-estate Data Traces
fig.add_trace(
    go.Scatter(x=df['X2 house age'], y=df['Y house price of unit area'], mode='markers', name='Data (Real-estate)'),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=df['X2 house age'], y=predictions_real_estate, mode='lines', name='Linear Regression (Real-estate)', line=dict(color=line_colors['real_estate'])),
    secondary_y=False,
)

# Salary Data Traces
fig.add_trace(
    go.Scatter(x=salary_data['YearsExperience'], y=salary_data['Salary'], mode='markers', name='Data (Salary)', visible=False),
    secondary_y=True,
)
fig.add_trace(
    go.Scatter(x=salary_data['YearsExperience'], y=predictions_salary, mode='lines', name='Linear Regression (Salary)', visible=False, line=dict(color=line_colors['salary'])),
    secondary_y=True,
)

# Insurance Data Traces
fig.add_trace(
    go.Scatter(x=insurance_data['bmi'], y=insurance_data['charges'], mode='markers', name='Data (Insurance)', visible=False),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=insurance_data['bmi'], y=predictions_insurance_bmi, mode='lines', name='Linear Regression (Insurance)', visible=False, line=dict(color=line_colors['insurance'])),
    secondary_y=False,
)

# Update the buttons to include the new axis titles and annotations
buttons = [
    {
        "label": "Real-estate Full",
        "method": "update",
        "args": [
            {"visible": [True, True, False, False, False, False]},
            {"title": "House Price vs. House Age",
             "xaxis.title": "House Age (years)",
             "yaxis.title": "House Price (unit area price)",
             "annotations": [real_estate_annotation]}
        ],
    },
    {
        "label": "Salary Full",
        "method": "update",
        "args": [
            {"visible": [False, False, True, True, False, False]},
            {"title": "Salary vs. Years of Experience",
             "xaxis.title": "Years of Experience",
             "yaxis.title": "Salary (USD)",
             "annotations": [salary_annotation]}
        ],
    },
    {
        "label": "Insurance Full",
        "method": "update",
        "args": [
            {"visible": [False, False, False, False, True, True]},
            {"title": "Insurance Charges vs. BMI",
             "xaxis.title": "BMI",
             "yaxis.title": "Insurance Charges (USD)",
             "annotations": [insurance_annotation]}
        ],
    },
]

# Update the layout with the initial plot (Real-estate) axis titles
fig.update_layout(
    updatemenus=[
        {
            "buttons": buttons,
            "direction": "down",
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "x": 0.35,
            "xanchor": "left",
            "y": 1.2,
            "yanchor": "top"
        },
    ],
    title="Data Visualization",
    xaxis_title="X-axis",
    yaxis_title="Y-axis",
    width=900,  
    height=600,  
    margin=dict(l=100, r=100, t=100, b=100),
)

# Initially, set the linear equation annotation for the Real-estate model
fig.add_annotation(**real_estate_annotation)

fig.show()

```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[5], line 8
          4 from plotly.subplots import make_subplots
          5 from sklearn.linear_model import LinearRegression
    ----> 8 df = pd.read_csv('assets/Real-estate.csv')
          9 salary_data = pd.read_csv('assets/Salary_dataset.csv')
         11 insurance_data = pd.read_csv('assets/insurance.csv')
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\io\parsers\readers.py:948, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
        935 kwds_defaults = _refine_defaults_read(
        936     dialect,
        937     delimiter,
       (...)
        944     dtype_backend=dtype_backend,
        945 )
        946 kwds.update(kwds_defaults)
    --> 948 return _read(filepath_or_buffer, kwds)
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\io\parsers\readers.py:611, in _read(filepath_or_buffer, kwds)
        608 _validate_names(kwds.get("names", None))
        610 # Create the parser.
    --> 611 parser = TextFileReader(filepath_or_buffer, **kwds)
        613 if chunksize or iterator:
        614     return parser
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\io\parsers\readers.py:1448, in TextFileReader.__init__(self, f, engine, **kwds)
       1445     self.options["has_index_names"] = kwds["has_index_names"]
       1447 self.handles: IOHandles | None = None
    -> 1448 self._engine = self._make_engine(f, self.engine)
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\io\parsers\readers.py:1705, in TextFileReader._make_engine(self, f, engine)
       1703     if "b" not in mode:
       1704         mode += "b"
    -> 1705 self.handles = get_handle(
       1706     f,
       1707     mode,
       1708     encoding=self.options.get("encoding", None),
       1709     compression=self.options.get("compression", None),
       1710     memory_map=self.options.get("memory_map", False),
       1711     is_text=is_text,
       1712     errors=self.options.get("encoding_errors", "strict"),
       1713     storage_options=self.options.get("storage_options", None),
       1714 )
       1715 assert self.handles is not None
       1716 f = self.handles.handle
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\io\common.py:863, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        858 elif isinstance(handle, str):
        859     # Check whether the filename is to be opened in binary mode.
        860     # Binary mode does not support 'encoding' and 'newline'.
        861     if ioargs.encoding and "b" not in ioargs.mode:
        862         # Encoding
    --> 863         handle = open(
        864             handle,
        865             ioargs.mode,
        866             encoding=ioargs.encoding,
        867             errors=errors,
        868             newline="",
        869         )
        870     else:
        871         # Binary mode
        872         handle = open(handle, ioargs.mode)
    

    FileNotFoundError: [Errno 2] No such file or directory: 'assets/Real-estate.csv'


## Cases using simple linear regression


```python
!pip install bokeh
```


```python
from bokeh.io import output_notebook

output_notebook()
```

### Example 1

**Scenario: Predicting Salary based on Years of Experience.**

Here, we have one independent variable (years of experience) and one dependent variable (salary)

```python
# Example Data
Years of Experience: [2, 4, 6, 8, 10]
Salary:              [50000, 70000, 90000, 110000, 130000]
```

Applying Simple Linear Regression finding $b$ (intercept) and $a$ (slope). We discuss these coefficients in more detail in subtopic [MSE fit](content:references:MSE):

**Calculation steps:**

1. **Calculate Means:**
    - Calculate the mean (average) of years of experience $\bar{x}$ and salary $\bar{y}$.

    $\bar{\boldsymbol x} = \frac{2 + 4 + 6 + 8 + 10}{5} = 6$

    $\bar{\boldsymbol y} = \frac{50000 + 70000 + 90000 + 110000 + 130000}{5} = 90000$

2. **Calculate Slope $a$ using formula {eq}`1-d-weights`:**

    $a = \frac{(2-6)(50000-90000) + (4-6)(70000-90000) + \ldots + (10-6)(130000-90000)}{(2-6)^2 + (4-6)^2 + \ldots + (10-6)^2} = 10000$

    After calculations, we find $a = 10000$.

3. **Calculate Intercept $b$:**

    Using formula {eq}`1-d-weights`: $b = \bar{x} - a \times \bar{x}$

    $b = 90000 - 10000 \times 6 = 30000$


After applying simple linear regression, we might find the equation:

$\text{Salary} = 30000 + 10000 \times \text{Years of Experience}$

**This equation suggests that, on average, each additional year of experience is associated with a $10,000 increase in salary.** 



```python
import plotly.graph_objects as go
import numpy as np

# Example data
years_of_experience = np.array([2, 4, 6, 8, 10])
salary = np.array([50000, 70000, 90000, 110000, 130000])

# Fit the linear regression model
model = np.polyfit(years_of_experience, salary, 1)
predict = np.poly1d(model)

# Create the base figure
fig = go.Figure()

# Function to update the figure
def update_figure(new_x):
    new_y = predict(new_x)
    updated_years = np.append(years_of_experience, new_x)
    updated_salary = np.append(salary, new_y)
    updated_model = np.polyfit(updated_years, updated_salary, 1)
    updated_predict = np.poly1d(updated_model)
    
    # Update the existing traces
    fig.data = []
    fig.add_trace(go.Scatter(x=updated_years, y=updated_salary, mode='markers', name='Data'))
    fig.add_trace(go.Scatter(x=np.linspace(min(years_of_experience), max(updated_years), 100),
                             y=updated_predict(np.linspace(min(years_of_experience), max(updated_years), 100)),
                             mode='lines', name='Updated Regression'))

# Add the initial traces
fig.add_trace(go.Scatter(x=years_of_experience, y=salary, mode='markers', name='Data'))
fig.add_trace(go.Scatter(x=np.linspace(min(years_of_experience), max(years_of_experience), 100),
                         y=predict(np.linspace(min(years_of_experience), max(years_of_experience), 100)),
                         mode='lines', name='Regression'))

# Layout configuration
fig.update_layout(title='Interactive Salary Prediction',
                  xaxis_title='Years of Experience',
                  yaxis_title='Salary ($)')

# Add a slider for selecting new data point
fig.update_layout(sliders=[
    {
        'steps': [
            {
                'method': 'animate',
                'label': str(year),
                'args': [
                    [str(year)], 
                    {
                        'mode': 'immediate',
                        'frame': {'duration': 500, 'redraw': True},
                        'transition': {'duration': 300}
                    }
                ]
            }
            for year in range(11, 21)
        ],
        'currentvalue': {'prefix': 'Years of Experience: '}
    }
])

# Add frames for each possible new data point
fig.frames = [
    go.Frame(
        data=[
            go.Scatter(x=np.append(years_of_experience, year), 
                       y=np.append(salary, predict(year)), 
                       mode='markers', name='Data'),
            go.Scatter(x=np.linspace(min(years_of_experience), year, 100),
                       y=np.poly1d(np.polyfit(np.append(years_of_experience, year),
                                              np.append(salary, predict(year)), 1))
                         (np.linspace(min(years_of_experience), year, 100)),
                       mode='lines', name='Regression')
        ],
        name=str(year)
    )
    for year in range(11, 21)
]

# Show the figure
fig.show()
```

### Example 2

**Scenario: Predicting Exam Scores**

Suppose we want to predict a student's exam score based on the number of hours they studied.

Independent Variable: Hours Studied
Dependent Variable: Exam Score

```python
# Example Data
Hours Studied: [3, 4, 5, 6, 7]
Exam Score:    [60, 70, 75, 85, 90]
```

After applying simple linear regression, we may find the equation: 

$\text{Exam Score} = 50 + 5 \times \text{Hours Studied}$

This equation suggests that, on average, each additional hour studied is associated with an increase of 5 points in the exam score.


```python
import numpy as np
from bokeh.io import output_notebook, push_notebook, show
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, TextInput, Button
from bokeh.models.callbacks import CustomJS
from IPython.display import display

# Enable inline display of Bokeh plots in Jupyter Notebook
output_notebook()

# New example data for Hours Studied vs Exam Score
hours_studied = [3, 4, 5, 6, 7]
exam_score = [60, 70, 75, 85, 90]

# Convert lists to numpy arrays for linear regression
hours_array = np.array(hours_studied)
score_array = np.array(exam_score)

# Fit a linear regression model for the new data
model_exam = np.polyfit(hours_array, score_array, 1)
predict_exam = np.poly1d(model_exam)

# Create a Bokeh figure
p = figure(title='Exam Score vs Hours Studied', x_axis_label='Hours Studied', y_axis_label='Exam Score', height=400, width=600)
source = ColumnDataSource(data={'hours': hours_studied, 'scores': exam_score})
p.scatter('hours', 'scores', source=source, color='purple', size=8, legend_label='Original Data')
p.line([min(hours_studied)-1, max(hours_studied)+2], [predict_exam(min(hours_studied)-1), predict_exam(max(hours_studied)+2)], color='gold', line_width=2, legend_label='Linear Regression')
p.legend.location = 'top_left'

# Create input widgets
hours_input = TextInput(value='', placeholder='Enter hours studied', title='Hours:')
add_data_button = Button(label='Add Data Point')

# Custom JavaScript callback for button click event
add_data_button_callback = CustomJS(args=dict(source=source, hours_input=hours_input, model_exam=model_exam), code="""
    var new_hours = parseInt(hours_input.value);
    if (!isNaN(new_hours)) {
        var new_score = model_exam[0] * new_hours + model_exam[1];
        new_score = Math.min(new_score, 100);
        source.data['hours'].push(new_hours);
        source.data['scores'].push(new_score);
        source.change.emit();
        // Redraw the plot
        Bokeh.index[0].model.document.interactive_obj.model.plot.invalidate_layout();
        Bokeh.index[0].model.document.interactive_obj.model.plot.reset();
    } else {
        console.log("Please enter a valid integer for hours studied.");
    }
""")

add_data_button.js_on_click(add_data_button_callback)

# Create a layout for widgets
inputs = column(hours_input, add_data_button)

# Display the plot and widgets
layout = column(p, inputs)
handle = show(layout, notebook_handle=True)

# Function to update the plot
def update():
    p.title.text = 'Exam Score vs Hours Studied'
    p.x_range.start = min(hours_studied) - 1
    p.x_range.end = max(hours_studied) + 2
    p.y_range.start = min(exam_score) - 10
    p.y_range.end = 110
    p.line([min(hours_studied)-1, max(hours_studied)+2], [predict_exam(min(hours_studied)-1), predict_exam(max(hours_studied)+2)], color='gold', line_width=2, legend_label='Linear Regression')
    p.legend.location = 'top_left'

# Initial plot
push_notebook(handle=handle)
update()
```


```python
pip install bokeh
```

### Example 3

**Scenario: Predicting Ice Cream Sales based on Temperature**

Independent Variable: Temperature (in degrees Celsius)
Dependent Variable: Ice Cream Sales (in units)

```python
# Example Data 
Temperature: [20, 25, 30, 35, 40] 
Ice Cream Sales: [50, 70, 90, 110, 130] 
```


```python
import numpy as np
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
from bokeh.models import Slider, CustomJS

# New example data
temperature = [20, 25, 30, 35, 40]
ice_cream_sales = [50, 70, 90, 110, 130]

# Initial regression model
model_ice_cream = np.polyfit(temperature, ice_cream_sales, 1)

# Create a Bokeh figure
p = figure(title="Ice Cream Sales vs Temperature", x_axis_label="Temperature (°C)", y_axis_label="Ice Cream Sales (units)",
           height=400, width=600)

# Plot the original data points
p.circle(temperature, ice_cream_sales, size=8, color='green', legend_label='Data')

# Define sliders for slope and intercept
slope_slider = Slider(title="Slope", value=model_ice_cream[0], start=-10, end=10, step=0.1)
intercept_slider = Slider(title="Intercept", value=model_ice_cream[1], start=-100, end=100, step=1)

# Define a data source
source = ColumnDataSource(data=dict(x=temperature, y=[model_ice_cream[0] * x + model_ice_cream[1] for x in temperature]))

# Function to update the plot based on slider values
callback = CustomJS(args=dict(source=source, slope=slope_slider, intercept=intercept_slider), code="""
    const data = source.data;
    const x = data['x'];
    const y = data['y'];
    const slope_value = slope.value;
    const intercept_value = intercept.value;
    for (let i = 0; i < x.length; i++) {
        y[i] = slope_value * x[i] + intercept_value;
    }
    source.change.emit();
""")

# Attach the callback to the sliders
slope_slider.js_on_change('value', callback)
intercept_slider.js_on_change('value', callback)

# Plot the regression line
p.line(x='x', y='y', source=source, line_width=2, line_color='orange', legend_label='Regression')

# Create a layout with the plot and sliders
layout = column(p, slope_slider, intercept_slider)

# Display the plot
output_notebook()
show(layout)
```

(content:references:MSE)=
## MSE fit

We use [Mean Squared Error (MSE)](https://fedmug.github.io/kbtu-ml-book/eval_metrics/regression.html#mean-squared-error-mse) in linear regression to quantify how well the model's predictions match the actual data.

Use MSE to fit parameters $\boldsymbol \theta = (a, b)$:

```{math}
:label: 1-d-mse
\mathcal L(a, b) =  \frac 1n\sum\limits_{i=1}^n (y_i - ax_i - b)^2 \to \min\limits_{a, b}.
```

````{admonition} What about some calculus?
:class: dropdown
We have

$$
    \frac{\partial \mathcal L}{\partial a} = -\frac 2n\sum\limits_{i=1}^n x_i(y_i - ax_i - b) = 0,
$$

$$
    \frac{\partial \mathcal L}{\partial b} = -\frac 2n\sum\limits_{i=1}^n (y_i - ax_i - b) = 0.
$$

From the last equality it follows that

$$
    b = \overline {\boldsymbol y} - a \overline {\boldsymbol x} 
$$

```{admonition} Proof
:class: tip, dropdown
To complete the proof, let's look at the first partial derivative:
        
$\frac{d\mathcal{L}}{da}=-\frac{2}{n}\sum\limits_{i=1}^nx_i(y_i-ax_i-b)=0$

Let’s solve the equation:

$-\frac{2}{n}\sum\limits_{i=1}^nx_i(y_i-ax_i-b)=0$

Substituting $b = \overline{\boldsymbol y}-a\overline{\boldsymbol x}$ into this equation, we get:

$-\frac{2}{n}\sum\limits_{i=1}^nx_i(y_i - a x_i - \overline{\boldsymbol y} + a\overline{\boldsymbol x}) = 0$

Multiplying both sides by $-\frac{2}{n}$:

$\sum\limits_{i=1}^nx_i(y_i - \overline{\boldsymbol y} - a (x_i -\overline{\boldsymbol x})) = 0$

$\sum\limits_{i=1}^nx_i(y_i-\overline{\boldsymbol y})  - a\sum\limits_{i=1}^nx_i(x_i-\overline{\boldsymbol x}) = 0$

Solve for $a$ is:

$a\sum\limits_{i=1}^nx_i(x_i-\overline{\boldsymbol x}) = \sum\limits_{i=1}^nx_i(y_i-\overline{\boldsymbol y})$ 

$a = \frac{\sum\limits_{i=1}^nx_i(y_i-\overline{\boldsymbol y})}{\sum\limits_{i=1}^nx_i(x_i-\overline{\boldsymbol x})} = \frac{\sum\limits_{i=1}^n(y_i-\overline{\boldsymbol y})}{\sum\limits_{i=1}^n(x_i-\overline{\boldsymbol x})}$

The values of $a$ and $b$ obtained from these equations ensure that the sum of the squared differences (the loss function $\mathcal{L}$) between the observed values $y_i$ and the predicted values $ax_i + b$ is minimized.
```
````

The optimal parameters are

```{math}
:label: 1-d-weights

\hat{a}= \frac{\sum\limits_{i=1}^n(x_i-\overline{\boldsymbol x})(y_i-\overline{\boldsymbol y})}{\sum\limits_{i=1}^n(x_i-\overline{\boldsymbol x})^2}

\hat{a}= \frac{\sum\limits_{i=1}^n(x_i-\overline{\boldsymbol x})(y_i-\overline{\boldsymbol y})}{\sum\limits_{i=1}^n(x_i-\overline{\boldsymbol x})(x_i-\overline{\boldsymbol x})}

\hat{a}= \frac{\sum\limits_{i=1}^n(y_i-\overline{\boldsymbol y})}{\sum\limits_{i=1}^n(x_i-\overline{\boldsymbol x})}

\hat{b}=\overline{\boldsymbol y}-\hat{a}\overline{\boldsymbol x}
```

Note that the slope is equal to the ratio of sample correlation between $\boldsymbol x$ and $\boldsymbol y$ to the sample variance of $\boldsymbol x$.


````{admonition} Question
:class: important
Does {eq}`1-d-weights` work for all possible values of $\boldsymbol x$ and $\boldsymbol y$?
```{admonition} Answer
:class: tip, dropdown
Equation {eq}`1-d-weights` provides the best values for $a$ and $b$ to minimize the $MSE$ in linear regression. However, it may encounter issues when the sample variance of $x$ is very small, potentially leading to numerical instability. In such cases, it's important to be cautious and consider alternative methods or adjustments to avoid potential problems arising from division by nearly zero.
```
````

````{admonition} What about code  for MSE?
:class: dropdown
```python
# MSE calculation
mse_real_estate = np.mean((y_real_estate - (a_real_estate * x_real_estate + b_real_estate))**2)
mse_salary = np.mean((y_salary - (a_salary * x_salary + b_salary))**2)

# Output of results
print("Real Estate Linear Regression MSE: ", mse_real_estate)
print("Salary Linear Regression MSE: ", mse_salary)
```

```
Real Estate Linear Regression MSE:  176.50047403131393
Salary Linear Regression MSE:  31270951.722280946
```
````

A lower MSE indicates better model performance. It reflects how well the model's predictions align with the actual data. However, be cautious about overfitting, as an overly complex model might perform well on training data but poorly on new, unseen data.

````{admonition} What about code for slope and bias?
:class: tip, dropdown
Let predictors be `house age`, target — `house price`. Let’s calculate the slope and bias using {eq}`1-d-weights`

```python
# For Real Estate Data
x_real_estate = df['X2 house age']
y_real_estate = df['Y house price of unit area']
a_real_estate = np.sum((x_real_estate - x_real_estate.mean()) * (y_real_estate - y_real_estate.mean())) / np.sum((x_real_estate - x_real_estate.mean()) ** 2)
b_real_estate = y_real_estate.mean() - a_real_estate * x_real_estate.mean()
print("a_real_estate = ", a_real_estate)
print("b_real_estate = ", b_real_estate)
```

```
a_real_estate = -0.2514884190853454
b_real_estate = 42.43469704626289
```

And in this example about salaries, let predictors be `YearsExperience`, target — `salary`. Let’s calculate: 

```python
# For Salary Data
x_salary = salary_data['YearsExperience']
y_salary = salary_data['Salary']
a_salary = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
b_salary = y.mean() - a * x.mean()
print("a_salary = ", a_salary)
print("b_salary = ", b_salary)
```

```
a_salary = 9449.962321455076
b_salary = 24848.2039665232
```
````


```python
display_quiz("#MSE")
```

## Dummy model

The model {eq}`simple-lin-reg` is simple but we can do even simpler! Let $a=0$ and predict labels by a **constant** (or **dummy**) model $\widehat y = b$. In fact, for constant predictions you don't need any features, only labels do the job.

````{admonition} Question
:class: important
Which value of $b$ minimizes the MSE {eq}`1-d-mse`?
```{admonition} Answer
:class: tip, dropdown
The optimal value of $b$ that minimizes the $MSE$ for the constant (dummy) model is obtained by setting the derivative of the $MSE$ with respect to $b$, to zero:
        
$\sum\limits_{i=1}^n (y_i - b)=0$
        
Solving for $b$, we get:
        
$\sum\limits_{i=1}^n y_i = nb$ 
        
$b = \frac{1}{n} \sum\limits_{i=1}^n y_i$ 
        
Therefore, the optimal value of $b$ that minimizes the MSE for the constant model is the mean of the observed values $y$:
        
$b_{\text{optimal}} = \frac{1}{n} \sum\limits_{i=1}^n y_i$
````

Linear regression can be used with different loss functions. For example, we can choose mean absolute error (MAE) instead of MSE:

```{math}
:label: 1-d-mae
\mathcal L(a, b) =  \frac 1n\sum\limits_{i=1}^n \vert y_i - ax_i - b\vert \to \min\limits_{a, b}.
```

This time it is unlikely that we can find the analytical solution. But maybe it can be done for the dummy model?

````{admonition} Question
:class: important
For which value of $b$ the value of MAE

$$
     \frac 1n\sum\limits_{i=1}^n \vert y_i - b\vert
$$

is minimal?

```{admonition} Answer
:class: tip, dropdown
$\widehat b = \mathrm{med}(\boldsymbol y)$ (see [this discussion](https://math.stackexchange.com/questions/113270/the-median-minimizes-the-sum-of-absolute-deviations-the-ell-1-norm) for details)
```
````

A real-life example of a dummy model is predicting, for instance, the average daily sales for a new store in retail, the mean exam score for students when lacking specific predictors, or the constant click-through rate in online advertising. These simple models serve as baseline references for more sophisticated analyses, helping assess the performance of advanced models in comparison.


```python
# For real estate data
y_real_estate = df['Y house price of unit area']
# For salary data
y_salary = salary_data['Salary']

# Calculating average values for dummy models
mean_y_real_estate = y_real_estate.mean()
mean_y_salary = y_salary.mean()

# Calculating MSE for dummy models
mse_dummy_real_estate = np.mean((y_real_estate - mean_y_real_estate)**2)
mse_dummy_salary = np.mean((y_salary - mean_y_salary)**2)

print("Dummy Model MSE (Real Estate): ", mse_dummy_real_estate)
print("Dummy Model MSE (Salary): ", mse_dummy_salary)
```


```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# For real state data
x_real_estate = df['X2 house age']
y_real_estate = df['Y house price of unit area']

a_real_estate = np.sum((x_real_estate - x_real_estate.mean()) * (y_real_estate - y_real_estate.mean())) / np.sum((x_real_estate - x_real_estate.mean())**2)
b_real_estate = y_real_estate.mean() - a_real_estate * x_real_estate.mean()

# For salary data
x_salary = salary_data['YearsExperience']
y_salary = salary_data['Salary']

a_salary = np.sum((x_salary - x_salary.mean()) * (y_salary - y_salary.mean())) / np.sum((x_salary - x_salary.mean())**2)
b_salary = y_salary.mean() - a_salary * x_salary.mean()

# Calculate mean values for the dummy models
mean_y_real_estate = y_real_estate.mean()
mean_y_salary = y_salary.mean()


# Create a single panel plot
fig = go.Figure()

# Add Real Estate data traces
fig.add_trace(go.Scatter(x=x_real_estate.squeeze(), y=y_real_estate, mode='markers', name='Real Data (Real Estate)', marker=dict(color='blue')))
fig.add_trace(go.Scatter(x=x_real_estate.squeeze(), y=predictions_real_estate, mode='lines', name='Linear Regression (Real Estate)', line=dict(color='red')))
fig.add_trace(go.Scatter(x=[x_real_estate.min(), x_real_estate.max()], y=[mean_y_real_estate, mean_y_real_estate], mode='lines', name='Dummy Model (Real Estate)', line=dict(color='green', dash='dash')))

# Add Salary data traces
fig.add_trace(go.Scatter(x=x_salary.squeeze(), y=y_salary, mode='markers', name='Real Data (Salary)', marker=dict(color='blue'), visible=False))
fig.add_trace(go.Scatter(x=x_salary.squeeze(), y=predictions_salary, mode='lines', name='Linear Regression (Salary)', line=dict(color='red'), visible=False))
fig.add_trace(go.Scatter(x=[x_salary.min(), x_salary.max()], y=[mean_y_salary, mean_y_salary], mode='lines', name='Dummy Model (Salary)', line=dict(color='green', dash='dash'), visible=False))

# Define interactive buttons for controlling the displayed data
buttons = [
    dict(label='Real Estate',
         method='update',
         args=[{'visible': [True, True, True, False, False, False]},
               {'title': 'Real Estate Data'}]),
    dict(label='Salary',
         method='update',
         args=[{'visible': [False, False, False, True, True, True]},
               {'title': 'Salary Data'}]),
]

# Add buttons to the layout
fig.update_layout(updatemenus=[{'buttons': buttons, 'direction': 'down', 'showactive': True, 'x': 1, 'xanchor': 'left', 'y': 1.21, 'yanchor': 'top'}])

# Update the layout
fig.update_layout(title="Comparison of Two Datasets<br>with Linear Regression and Dummy Model")

# Display the plot
fig.show()
```


```python
display_quiz("#Dummy")
```

## RSS and $R^2$-score

$RSS$ (Residual Sum of Squares) measures how well a linear regression model fits the data by calculating the sum of the squared errors. A lower RSS indicates a better fit. 

Putting the optimal weights $\widehat a$ and $\widehat b$ into the loss function {eq}`1-d-mse`, we obtain **residual square error** (RSE). Multiplying by $n$ we get **residual sum of squares**

$$
    RSS = \sum\limits_{i=1}^n(y_i - \widehat a x_i - \widehat b)^2.
$$

Also, **total sum of squares** is defined as

$$
TSS = \sum\limits_{i=1}^n(y_i - \overline {\boldsymbol y})^2.
$$

A popular metric called **coefficient of determination** (or **$R^2$-score**) is defined as

```{math}
:label: R2-score
R^2 = 1 - \frac{RSS}{TSS} = 1 - \frac{\sum\limits_{i=1}^n(y_i - \widehat y_i)^2}{\sum\limits_{i=1}^n(y_i - \overline {\boldsymbol y})^2}.
```

The coefficient of determination shows proportion of variance explained. $R^2$-score does not exceed $1$ (the greater —  the better).

<span style="display:none" id="R2_dummy">W3sicXVlc3Rpb24iOiAiV2hhdCBpcyB0aGUgJFJeMiQtc2NvcmUgb2YgdGhlIGR1bW15IG1vZGVsICRcXHdpZGVoYXQgeSA9IFxcb3ZlcmxpbmUge1xcYm9sZHN5bWJvbCB5fSQ/IiwgInR5cGUiOiAibnVtZXJpYyIsICJhbnN3ZXJzIjogW3sidHlwZSI6ICJ2YWx1ZSIsICJ2YWx1ZSI6IDAsICJjb3JyZWN0IjogdHJ1ZSwgImZlZWRiYWNrIjogIllvdSBuYWlsZWQgaXQhICRSXjIgPSAxIC0gMSA9IDAkIn0sIHsidHlwZSI6ICJ2YWx1ZSIsICJ2YWx1ZSI6IDEsICJjb3JyZWN0IjogZmFsc2UsICJmZWVkYmFjayI6ICJObywgaXQgc2hvdWxkIGJlIHN1YnRyYWN0ZWQgZnJvbSAkMSQifSwgeyJ0eXBlIjogInJhbmdlIiwgInJhbmdlIjogWy0xMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMCwgMF0sICJmZWVkYmFjayI6ICIkUl4yJC1zY29yZSBtdXN0IGJlIGJldHdlZW4gMCBhbmQgMSJ9LCB7InR5cGUiOiAicmFuZ2UiLCAicmFuZ2UiOiBbMS4wMDAwMDAwMDAxLCAxMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMF0sICJmZWVkYmFjayI6ICIkUl4yJC1zY29yZSBtdXN0IGJlIGJldHdlZW4gMCBhbmQgMSJ9LCB7InR5cGUiOiAiZGVmYXVsdCIsICJmZWVkYmFjayI6ICJUaGlzIHZhbHVlIGxvb2tzIGluY29ycmVjdCJ9XX1d</span>


```python
from jupyterquiz import display_quiz
display_quiz("#R2_dummy")
```

$R^2$-score measures the amount of variability that is left unexplained after performing the regression. It shows how better the model works in comparison with dummy prediction.


```python
from sklearn.metrics import mean_squared_error, r2_score

# For real estate model
rss_real_estate = np.sum((y_real_estate - predictions_real_estate) ** 2)
r2_real_estate = r2_score(y_real_estate, predictions_real_estate)

# For the salary model
rss_salary = np.sum((y_salary - predictions_salary) ** 2)
r2_salary = r2_score(y_salary, predictions_salary)

print("Real Estate Model: RSS =", rss_real_estate, "R\u00b2-Score =", r2_real_estate)
print("Salary Model: RSS =", rss_salary, "R\u00b2-Score =", r2_salary)

```

```{admonition} What do these values mean?
:class: tip, dropdown
In our case, the real estate model has significantly less importance in terms of RSS compared to the salary model, indicating that the real estate model is better at minimizing errors in data than the salary model. Additionally, the salary model has a very high $R^2$, suggesting that a large portion of variability in salary data is explained by this model. In contrast, the real estate model has a very low $R^2$, indicating its low ability to explain variability in real estate data.
```

<div style="display:none" id="Example2">W3siZm9yU2FuemgiOiAiQ2FzZXMgdXNpbmcgLi4tPiBlbmQgRXhhbXBsZSAyIiwgInF1ZXN0aW9uIjogIklmIGNhbGN1bGF0ZWQgdXNpbmcgdGhpcyBmb3JtdWxhLCB3aGF0IGdyYWRlIHdpbGwgYSBzdHVkZW50IHJlY2VpdmUgd2hvIHN0dWRpZWQgZm9yIDYgaG91cnM/IiwgInR5cGUiOiAibnVtZXJpYyIsICJwcmVjaXNpb24iOiAyLCAiYW5zd2VycyI6IFt7InR5cGUiOiAidmFsdWUiLCAidmFsdWUiOiA4MCwgImNvcnJlY3QiOiB0cnVlLCAiZmVlZGJhY2siOiAiSXQgaXMgY29ycmVjdCEifSwgeyJ0eXBlIjogInJhbmdlIiwgInJhbmdlIjogWy0xMDAwMDAwMDAsIDBdLCAiY29ycmVjdCI6IGZhbHNlLCAiZmVlZGJhY2siOiAiQ29ycmVjdCBhbnN3ZXIgaXMgODAuIDUwICs1ICogNiA9IDgwIn1dfV0=</div>

## Example: Boston Dataset


```python
import pandas as pd
boston = pd.read_csv("../ISLP_datsets/Boston.csv").drop("Unnamed: 0", axis=1)
boston.head()
```

<span style="display:none" id="R2_dummy">W3sicXVlc3Rpb24iOiAiV2hhdCBpcyB0aGUgJFJeMiQtc2NvcmUgb2YgdGhlIGR1bW15IG1vZGVsICRcXHdpZGVoYXQgeSA9IFxcb3ZlcmxpbmUge1xcYm9sZHN5bWJvbCB5fSQ/IiwgInR5cGUiOiAibnVtZXJpYyIsICJhbnN3ZXJzIjogW3sidHlwZSI6ICJ2YWx1ZSIsICJ2YWx1ZSI6IDAsICJjb3JyZWN0IjogdHJ1ZSwgImZlZWRiYWNrIjogIllvdSBuYWlsZWQgaXQhICRSXjIgPSAxIC0gMSA9IDAkIn0sIHsidHlwZSI6ICJ2YWx1ZSIsICJ2YWx1ZSI6IDEsICJjb3JyZWN0IjogZmFsc2UsICJmZWVkYmFjayI6ICJObywgaXQgc2hvdWxkIGJlIHN1YnRyYWN0ZWQgZnJvbSAkMSQifSwgeyJ0eXBlIjogInJhbmdlIiwgInJhbmdlIjogWy0xMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMCwgMF0sICJmZWVkYmFjayI6ICIkUl4yJC1zY29yZSBtdXN0IGJlIGJldHdlZW4gMCBhbmQgMSJ9LCB7InR5cGUiOiAicmFuZ2UiLCAicmFuZ2UiOiBbMS4wMDAwMDAwMDAxLCAxMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMF0sICJmZWVkYmFjayI6ICIkUl4yJC1zY29yZSBtdXN0IGJlIGJldHdlZW4gMCBhbmQgMSJ9LCB7InR5cGUiOiAiZGVmYXVsdCIsICJmZWVkYmFjayI6ICJUaGlzIHZhbHVlIGxvb2tzIGluY29ycmVjdCJ9XX1d</span>

Let predictors be `lstat`, target — `medv`. Let's calculate the regression coefficients using {eq}`1-d-weights`:


```python
import numpy as np

x = boston['lstat']
y = boston['medv']
a = np.sum((x -x.mean()) * (y - y.mean())) /  np.sum((x -x.mean()) ** 2)
b = y.mean() - a*x.mean()
print("a = ", a)
print("b = ", b)
```

Now plot the data and the regression line:


```python
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'svg'
plt.scatter(x, y, s=10, c='b', alpha=0.7)
xs = np.linspace(x.min(), x.max(), num=10)
plt.plot(xs, a*xs + b, c='r', lw=2, label=r"$y = \widehat a x + \widehat b$")
plt.plot([xs.min(), xs.max()], [y.mean(), y.mean()], c='orange', lw=2, label=r"$y = \overline y$")
plt.xlabel("lstat")
plt.ylabel("medv")
plt.title("Simple linear regression vs dummy model")
plt.legend()
plt.grid(ls=":");
```

Calculate MSE:


```python
mse_lin_reg = np.mean((y - a*x - b)**2)
mse_dummy = np.mean((y - y.mean())**2)
print("Linear regression MSE:", mse_lin_reg)
print("Dummy model MSE:", mse_dummy)
```

Coefficient of determination:


```python
print("R\u00b2-score:", 1 - mse_lin_reg / np.mean((y - y.mean())**2))
print("Dummy R\u00b2-score:", 1 - mse_dummy / np.mean((y - y.mean())**2))
```

Of course, the linear regression line can be found automatically:


```python
import seaborn as sb

sb.regplot(boston, x="lstat", y="medv",
           scatter_kws={"color": "blue", "s": 10}, line_kws={"color": "red"}, 
           ).set_title('Boston linear regression');
```

Linear regression from `sklearn`:


```python
from sklearn.linear_model import LinearRegression

LR = LinearRegression()
x_reshaped = x.values.reshape(-1, 1)
LR.fit(x_reshaped, y)
print("intercept:", LR.intercept_)
print("slope:", LR.coef_[0])
print("r-score:", LR.score(x_reshaped, y))
print("MSE:", np.mean((LR.predict(x_reshaped) - y) ** 2))
```

Compare this with dummy model:


```python
dummy_mse = np.mean((y - y.mean())**2)
print(dummy_mse)
```


```python
display_quiz("#Boston")
```

## Exercises

````{admonition} Exercises
:class: sphinx_exercise
Prove that $\frac{1}{n}\sum\limits_{i=1}^n(x_i-\overline{x})^2=\overline{x^2}-(\overline{x})^2$     where       $\overline{x^2}=\frac{1}{n}\sum\limits_{i=1}^nx_i^2$

```{admonition} Solution
:class: hint, dropdown
Prove that $\frac{1}{n}\sum\limits_{i=1}^n(x_i-\overline{x})^2=\frac{1}{n}\sum\limits_{i=1}^nx_i^2-(\overline{x})^2$

$\frac{1}{n}\sum\limits_{i=1}^n(x_i-\overline{x})^2 = $

$ \frac{1}{n}\sum\limits_{i=1}^n(x_i^2 - 2x_i\bar{x} + \bar{x}^2)= $

$ \frac{1}{n}\left(\sum\limits_{i=1}^nx_i^2 - 2\bar{x}\sum\limits_{i=1}^nx_i + \sum\limits_{i=1}^n\bar{x}^2\right)$

where $\sum\limits_{i=1}^nx_i = n\bar{x}$ AND $\sum\limits_{i=1}^n\bar{x}^2 = n\bar{x}^2$ 

$\frac{1}{n}\left(\sum\limits_{i=1}^nx_i^2 - 2n\bar{x}^2 + n\bar{x}^2\right) = 
\frac{1}{n}\left(\sum\limits_{i=1}^nx_i^2 - n\bar{x}^2\right) = \frac{1}{n}\sum\limits_{i=1}^nx_i^2 - \bar{x}^2$
```
````
