{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Importing necessary libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 362,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import math\n",
    "import pickle as pk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 363,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv('train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 364,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(223, 16)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>WBC</th>\n",
       "      <th>Platelets</th>\n",
       "      <th>Neutrophils</th>\n",
       "      <th>Lymphocytes</th>\n",
       "      <th>Monocytes</th>\n",
       "      <th>Eosinophils</th>\n",
       "      <th>Basophils</th>\n",
       "      <th>CRP</th>\n",
       "      <th>AST</th>\n",
       "      <th>ALT</th>\n",
       "      <th>ALP</th>\n",
       "      <th>GGT</th>\n",
       "      <th>LDH</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>F</td>\n",
       "      <td>70</td>\n",
       "      <td>4.9</td>\n",
       "      <td>145.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.7</td>\n",
       "      <td>0.3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>87.9</td>\n",
       "      <td>68.0</td>\n",
       "      <td>41.0</td>\n",
       "      <td>89.0</td>\n",
       "      <td>63.0</td>\n",
       "      <td>606.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>M</td>\n",
       "      <td>61</td>\n",
       "      <td>11.3</td>\n",
       "      <td>166.0</td>\n",
       "      <td>10.7</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>174.3</td>\n",
       "      <td>118.0</td>\n",
       "      <td>95.0</td>\n",
       "      <td>59.0</td>\n",
       "      <td>77.0</td>\n",
       "      <td>808.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>F</td>\n",
       "      <td>70</td>\n",
       "      <td>15.8</td>\n",
       "      <td>202.0</td>\n",
       "      <td>13.9</td>\n",
       "      <td>1</td>\n",
       "      <td>0.8</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>43.8</td>\n",
       "      <td>20.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>80.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>235.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>M</td>\n",
       "      <td>85</td>\n",
       "      <td>10.6</td>\n",
       "      <td>227.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.3</td>\n",
       "      <td>31.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>F</td>\n",
       "      <td>89</td>\n",
       "      <td>4.5</td>\n",
       "      <td>170.0</td>\n",
       "      <td>3.8</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>133.6</td>\n",
       "      <td>57.0</td>\n",
       "      <td>17.0</td>\n",
       "      <td>81.0</td>\n",
       "      <td>19.0</td>\n",
       "      <td>626.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  GENDER  AGE   WBC  Platelets  Neutrophils Lymphocytes  Monocytes  \\\n",
       "0      F   70   4.9      145.0          4.0         0.7        0.3   \n",
       "1      M   61  11.3      166.0         10.7         0.5        0.1   \n",
       "2      F   70  15.8      202.0         13.9           1        0.8   \n",
       "3      M   85  10.6      227.0          NaN         NaN        NaN   \n",
       "4      F   89   4.5      170.0          3.8         0.5        0.2   \n",
       "\n",
       "   Eosinophils  Basophils    CRP    AST   ALT   ALP   GGT    LDH  Class  \n",
       "0          0.0        0.0   87.9   68.0  41.0  89.0  63.0  606.0      1  \n",
       "1          0.0        0.0  174.3  118.0  95.0  59.0  77.0  808.0      1  \n",
       "2          0.0        0.0   43.8   20.0  26.0  80.0  16.0  235.0      0  \n",
       "3          NaN        NaN    6.3   31.0  16.0   NaN   NaN    NaN      0  \n",
       "4          0.0        0.0  133.6   57.0  17.0  81.0  19.0  626.0      1  "
      ]
     },
     "execution_count": 364,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(data.shape)\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 365,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GENDER          0.000000\n",
       "AGE             0.000000\n",
       "WBC             0.896861\n",
       "Platelets       0.896861\n",
       "Neutrophils    26.008969\n",
       "Lymphocytes    26.008969\n",
       "Monocytes      26.008969\n",
       "Eosinophils    26.008969\n",
       "Basophils      26.457399\n",
       "CRP             1.793722\n",
       "AST             0.448430\n",
       "ALT             3.587444\n",
       "ALP            52.466368\n",
       "GGT            49.775785\n",
       "LDH            29.596413\n",
       "Class           0.000000\n",
       "dtype: float64"
      ]
     },
     "execution_count": 365,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(data.isnull().sum()/data.shape[0])*100"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Around 50% of the observations are null in the columns ALP and GGT\n",
    "- So we can drop them"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 366,
   "metadata": {},
   "outputs": [],
   "source": [
    "data.drop(['ALP','GGT'],inplace=True,axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- for each row we are setting a threshold of minimum 6 values out of 14 are required inorder to predict the outcome"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 367,
   "metadata": {},
   "outputs": [],
   "source": [
    "data.dropna(thresh=6,axis=0,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling WBC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 368,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>WBC</th>\n",
       "      <th>Platelets</th>\n",
       "      <th>Neutrophils</th>\n",
       "      <th>Lymphocytes</th>\n",
       "      <th>Monocytes</th>\n",
       "      <th>Eosinophils</th>\n",
       "      <th>Basophils</th>\n",
       "      <th>CRP</th>\n",
       "      <th>AST</th>\n",
       "      <th>ALT</th>\n",
       "      <th>LDH</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>188</th>\n",
       "      <td>M</td>\n",
       "      <td>37</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>108.8</td>\n",
       "      <td>27.0</td>\n",
       "      <td>39.0</td>\n",
       "      <td>321.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    GENDER  AGE  WBC  Platelets  Neutrophils Lymphocytes  Monocytes  \\\n",
       "188      M   37  NaN        NaN          NaN         NaN        NaN   \n",
       "\n",
       "     Eosinophils  Basophils    CRP   AST   ALT    LDH  Class  \n",
       "188          NaN        NaN  108.8  27.0  39.0  321.0      1  "
      ]
     },
     "execution_count": 368,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data['WBC'].isnull()]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- we are taking median of WBC of class 1 observations to replace null value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 369,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6.8"
      ]
     },
     "execution_count": 369,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data['Class']==1]['WBC'].median()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 370,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['WBC'].fillna(6.8,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "scrolled": true
   },
   "source": [
    "## handling Neutrophils"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 371,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 4. , 10.7, 13.9,  nan,  3.8,  7.6,  5.7,  3.4,  3.3,  2.8,  4.3,\n",
       "        5.9,  5.2,  3.5,  2.7,  8.9,  5.6,  3.7, 14.1, 12.2,  9. , 10. ,\n",
       "       16. ,  2.1,  6.5,  9.7,  3.1,  7.9,  4.2, 10.5,  4.8,  1.2,  0.8,\n",
       "        8. ,  3.6,  2.4, 16.5,  7.1,  2.6,  8.1,  3.2,  4.5, 24.3,  9.6,\n",
       "        9.5, 12. ,  7.4,  6.7,  0.9,  5.1,  6.2,  2.2,  2.9,  5.3, 15.9,\n",
       "        6.8,  1.4,  9.2, 13.5, 20.2, 16.1, 14. ,  3.9,  1.9,  2.5,  4.7,\n",
       "        5.8,  5.4,  6.3,  2. ,  4.1,  1.5,  4.4,  3. ,  0.5, 18.9, 12.9,\n",
       "       17.6,  8.8, 15.7,  7.3,  9.4,  1.8,  1.1,  5. ,  9.3,  9.1])"
      ]
     },
     "execution_count": 371,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Neutrophils'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 372,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>AGE</th>\n",
       "      <th>WBC</th>\n",
       "      <th>Platelets</th>\n",
       "      <th>Neutrophils</th>\n",
       "      <th>Monocytes</th>\n",
       "      <th>Eosinophils</th>\n",
       "      <th>Basophils</th>\n",
       "      <th>CRP</th>\n",
       "      <th>AST</th>\n",
       "      <th>ALT</th>\n",
       "      <th>LDH</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>AGE</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.048690</td>\n",
       "      <td>-0.168887</td>\n",
       "      <td>0.078683</td>\n",
       "      <td>0.016221</td>\n",
       "      <td>-0.079684</td>\n",
       "      <td>-0.042996</td>\n",
       "      <td>0.126386</td>\n",
       "      <td>0.013693</td>\n",
       "      <td>-0.093119</td>\n",
       "      <td>0.209956</td>\n",
       "      <td>0.102342</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>WBC</th>\n",
       "      <td>0.048690</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.438809</td>\n",
       "      <td>0.947954</td>\n",
       "      <td>0.567929</td>\n",
       "      <td>0.202245</td>\n",
       "      <td>0.524366</td>\n",
       "      <td>0.292700</td>\n",
       "      <td>0.134747</td>\n",
       "      <td>0.058097</td>\n",
       "      <td>0.342937</td>\n",
       "      <td>-0.167186</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Platelets</th>\n",
       "      <td>-0.168887</td>\n",
       "      <td>0.438809</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.342153</td>\n",
       "      <td>0.316795</td>\n",
       "      <td>0.219614</td>\n",
       "      <td>0.316627</td>\n",
       "      <td>0.113938</td>\n",
       "      <td>0.031140</td>\n",
       "      <td>0.143753</td>\n",
       "      <td>0.107239</td>\n",
       "      <td>-0.066778</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Neutrophils</th>\n",
       "      <td>0.078683</td>\n",
       "      <td>0.947954</td>\n",
       "      <td>0.342153</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.430662</td>\n",
       "      <td>0.090837</td>\n",
       "      <td>0.440451</td>\n",
       "      <td>0.392323</td>\n",
       "      <td>0.192125</td>\n",
       "      <td>0.101442</td>\n",
       "      <td>0.334340</td>\n",
       "      <td>-0.103483</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Monocytes</th>\n",
       "      <td>0.016221</td>\n",
       "      <td>0.567929</td>\n",
       "      <td>0.316795</td>\n",
       "      <td>0.430662</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.264536</td>\n",
       "      <td>0.444978</td>\n",
       "      <td>-0.034554</td>\n",
       "      <td>-0.099062</td>\n",
       "      <td>-0.060665</td>\n",
       "      <td>-0.168622</td>\n",
       "      <td>-0.262518</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Eosinophils</th>\n",
       "      <td>-0.079684</td>\n",
       "      <td>0.202245</td>\n",
       "      <td>0.219614</td>\n",
       "      <td>0.090837</td>\n",
       "      <td>0.264536</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.423101</td>\n",
       "      <td>-0.078713</td>\n",
       "      <td>-0.044971</td>\n",
       "      <td>-0.032244</td>\n",
       "      <td>-0.131654</td>\n",
       "      <td>-0.228690</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Basophils</th>\n",
       "      <td>-0.042996</td>\n",
       "      <td>0.524366</td>\n",
       "      <td>0.316627</td>\n",
       "      <td>0.440451</td>\n",
       "      <td>0.444978</td>\n",
       "      <td>0.423101</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.031367</td>\n",
       "      <td>0.055703</td>\n",
       "      <td>0.057552</td>\n",
       "      <td>0.005043</td>\n",
       "      <td>-0.150111</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>CRP</th>\n",
       "      <td>0.126386</td>\n",
       "      <td>0.292700</td>\n",
       "      <td>0.113938</td>\n",
       "      <td>0.392323</td>\n",
       "      <td>-0.034554</td>\n",
       "      <td>-0.078713</td>\n",
       "      <td>-0.031367</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.248465</td>\n",
       "      <td>0.064313</td>\n",
       "      <td>0.470108</td>\n",
       "      <td>0.233632</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>AST</th>\n",
       "      <td>0.013693</td>\n",
       "      <td>0.134747</td>\n",
       "      <td>0.031140</td>\n",
       "      <td>0.192125</td>\n",
       "      <td>-0.099062</td>\n",
       "      <td>-0.044971</td>\n",
       "      <td>0.055703</td>\n",
       "      <td>0.248465</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.714150</td>\n",
       "      <td>0.618139</td>\n",
       "      <td>0.292650</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ALT</th>\n",
       "      <td>-0.093119</td>\n",
       "      <td>0.058097</td>\n",
       "      <td>0.143753</td>\n",
       "      <td>0.101442</td>\n",
       "      <td>-0.060665</td>\n",
       "      <td>-0.032244</td>\n",
       "      <td>0.057552</td>\n",
       "      <td>0.064313</td>\n",
       "      <td>0.714150</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.304780</td>\n",
       "      <td>0.244920</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LDH</th>\n",
       "      <td>0.209956</td>\n",
       "      <td>0.342937</td>\n",
       "      <td>0.107239</td>\n",
       "      <td>0.334340</td>\n",
       "      <td>-0.168622</td>\n",
       "      <td>-0.131654</td>\n",
       "      <td>0.005043</td>\n",
       "      <td>0.470108</td>\n",
       "      <td>0.618139</td>\n",
       "      <td>0.304780</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420369</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Class</th>\n",
       "      <td>0.102342</td>\n",
       "      <td>-0.167186</td>\n",
       "      <td>-0.066778</td>\n",
       "      <td>-0.103483</td>\n",
       "      <td>-0.262518</td>\n",
       "      <td>-0.228690</td>\n",
       "      <td>-0.150111</td>\n",
       "      <td>0.233632</td>\n",
       "      <td>0.292650</td>\n",
       "      <td>0.244920</td>\n",
       "      <td>0.420369</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  AGE       WBC  Platelets  Neutrophils  Monocytes  \\\n",
       "AGE          1.000000  0.048690  -0.168887     0.078683   0.016221   \n",
       "WBC          0.048690  1.000000   0.438809     0.947954   0.567929   \n",
       "Platelets   -0.168887  0.438809   1.000000     0.342153   0.316795   \n",
       "Neutrophils  0.078683  0.947954   0.342153     1.000000   0.430662   \n",
       "Monocytes    0.016221  0.567929   0.316795     0.430662   1.000000   \n",
       "Eosinophils -0.079684  0.202245   0.219614     0.090837   0.264536   \n",
       "Basophils   -0.042996  0.524366   0.316627     0.440451   0.444978   \n",
       "CRP          0.126386  0.292700   0.113938     0.392323  -0.034554   \n",
       "AST          0.013693  0.134747   0.031140     0.192125  -0.099062   \n",
       "ALT         -0.093119  0.058097   0.143753     0.101442  -0.060665   \n",
       "LDH          0.209956  0.342937   0.107239     0.334340  -0.168622   \n",
       "Class        0.102342 -0.167186  -0.066778    -0.103483  -0.262518   \n",
       "\n",
       "             Eosinophils  Basophils       CRP       AST       ALT       LDH  \\\n",
       "AGE            -0.079684  -0.042996  0.126386  0.013693 -0.093119  0.209956   \n",
       "WBC             0.202245   0.524366  0.292700  0.134747  0.058097  0.342937   \n",
       "Platelets       0.219614   0.316627  0.113938  0.031140  0.143753  0.107239   \n",
       "Neutrophils     0.090837   0.440451  0.392323  0.192125  0.101442  0.334340   \n",
       "Monocytes       0.264536   0.444978 -0.034554 -0.099062 -0.060665 -0.168622   \n",
       "Eosinophils     1.000000   0.423101 -0.078713 -0.044971 -0.032244 -0.131654   \n",
       "Basophils       0.423101   1.000000 -0.031367  0.055703  0.057552  0.005043   \n",
       "CRP            -0.078713  -0.031367  1.000000  0.248465  0.064313  0.470108   \n",
       "AST            -0.044971   0.055703  0.248465  1.000000  0.714150  0.618139   \n",
       "ALT            -0.032244   0.057552  0.064313  0.714150  1.000000  0.304780   \n",
       "LDH            -0.131654   0.005043  0.470108  0.618139  0.304780  1.000000   \n",
       "Class          -0.228690  -0.150111  0.233632  0.292650  0.244920  0.420369   \n",
       "\n",
       "                Class  \n",
       "AGE          0.102342  \n",
       "WBC         -0.167186  \n",
       "Platelets   -0.066778  \n",
       "Neutrophils -0.103483  \n",
       "Monocytes   -0.262518  \n",
       "Eosinophils -0.228690  \n",
       "Basophils   -0.150111  \n",
       "CRP          0.233632  \n",
       "AST          0.292650  \n",
       "ALT          0.244920  \n",
       "LDH          0.420369  \n",
       "Class        1.000000  "
      ]
     },
     "execution_count": 372,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 373,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Correlation between WBC and Neutrophils')"
      ]
     },
     "execution_count": 373,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot(x='WBC',y='Neutrophils',data=data)\n",
    "plt.title('Correlation between WBC and Neutrophils')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- We can clearly see that the Neutrophils is higly correlated (0.94) with the WBC column\n",
    "- we are building a linear regression model inorder to predict the missing neutrophils values corresponding with WBC Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 374,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "Xt=data[data['Neutrophils'].notnull()][['WBC']]\n",
    "Yt =data[data['Neutrophils'].notnull()][['Neutrophils']]\n",
    "from sklearn.linear_model import LinearRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 375,
   "metadata": {},
   "outputs": [],
   "source": [
    "lr=LinearRegression()\n",
    "lr.fit(Xt,Yt)\n",
    "yp= lr.predict(pd.DataFrame(data['WBC']))\n",
    "data['Y_pred']=yp\n",
    "data['Neutrophils'].fillna(data['Y_pred'], inplace = True)\n",
    "data.drop('Y_pred',axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling Lymphocytes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 376,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['0.7', '0.5', '1', nan, '1.7', '0.3', '0.8', '0.2', '1.1', '3.3',\n",
       "       '0.6', '2.4', '1.5', '0.9', '2.1', '0.4', '1.4', '1.3', '2.3',\n",
       "       '3.1', '1.2', '7.2', '2.2', '1.6', '2.9', '1.8', '2.7', '1.9', '2',\n",
       "       '2.5', '4.1', '0-4', '3'], dtype=object)"
      ]
     },
     "execution_count": 376,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Lymphocytes'].unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- We need to replace the '0-4' (assuming it as a typing mistake) value since it cannot be converted to numeric value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 377,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Lymphocytes'] = data['Lymphocytes'].replace(['0-4'], '0.4')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 378,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('O')"
      ]
     },
     "execution_count": 378,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Lymphocytes'].dtype"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Converting into Numeric type"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 379,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Lymphocytes'] = pd.to_numeric(data['Lymphocytes'] , errors = 'coerce')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Performing class based replacement of null values\n",
    "- taking mean of Lymphocytes for each class and replacing correspondingly"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 380,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x245b739e7f0>"
      ]
     },
     "execution_count": 380,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data[data['Lymphocytes'].isnull()]['Class'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 381,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "c0_mean = data[data['Class']==0]['Lymphocytes'].mean()\n",
    "c1_mean = data[data['Class']==1]['Lymphocytes'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 382,
   "metadata": {},
   "outputs": [],
   "source": [
    "for ind in data.index:\n",
    "    if str(data.loc[ind,'Lymphocytes']).lower() == 'nan':\n",
    "        if data.loc[ind, 'Class'] == 1:\n",
    "            data.loc[ind,'Lymphocytes'] = c1_mean\n",
    "        else:\n",
    "            data.loc[ind,'Lymphocytes'] = c0_mean\n",
    "            \n",
    "# data.Lymphocytes.unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling Monocytes, Eosinophils, Basophils"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 383,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Monocytes']=data['Monocytes'].replace(np.nan,data['Monocytes'].median())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 384,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Eosinophils']=data['Eosinophils'].replace(np.nan,data['Eosinophils'].mode()[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 385,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Basophils']=data['Basophils'].replace(np.nan,data['Basophils'].mode()[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling CRP, ALT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 386,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>WBC</th>\n",
       "      <th>Platelets</th>\n",
       "      <th>Neutrophils</th>\n",
       "      <th>Lymphocytes</th>\n",
       "      <th>Monocytes</th>\n",
       "      <th>Eosinophils</th>\n",
       "      <th>Basophils</th>\n",
       "      <th>CRP</th>\n",
       "      <th>AST</th>\n",
       "      <th>ALT</th>\n",
       "      <th>LDH</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>M</td>\n",
       "      <td>80</td>\n",
       "      <td>10.1</td>\n",
       "      <td>325.0</td>\n",
       "      <td>7.6</td>\n",
       "      <td>1.7</td>\n",
       "      <td>0.6</td>\n",
       "      <td>0.2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>19.0</td>\n",
       "      <td>23.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>132</th>\n",
       "      <td>M</td>\n",
       "      <td>84</td>\n",
       "      <td>7.8</td>\n",
       "      <td>167.0</td>\n",
       "      <td>6.2</td>\n",
       "      <td>0.9</td>\n",
       "      <td>0.7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>33.0</td>\n",
       "      <td>29.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>160</th>\n",
       "      <td>M</td>\n",
       "      <td>77</td>\n",
       "      <td>3.0</td>\n",
       "      <td>162.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.4</td>\n",
       "      <td>0.4</td>\n",
       "      <td>0.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>21.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>220.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    GENDER  AGE   WBC  Platelets  Neutrophils  Lymphocytes  Monocytes  \\\n",
       "5        M   80  10.1      325.0          7.6          1.7        0.6   \n",
       "132      M   84   7.8      167.0          6.2          0.9        0.7   \n",
       "160      M   77   3.0      162.0          2.0          0.4        0.4   \n",
       "\n",
       "     Eosinophils  Basophils  CRP   AST   ALT    LDH  Class  \n",
       "5            0.2        0.0  NaN  19.0  23.0    NaN      0  \n",
       "132          0.0        0.0  NaN  33.0  29.0    NaN      1  \n",
       "160          0.1        0.0  NaN  21.0  10.0  220.0      1  "
      ]
     },
     "execution_count": 386,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data['CRP'].isnull()]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The missing CRP Values are of 3 people who are older than 77 years \n",
    "- So, we are replacing the values with mean of CRP values of people with age greater than 77 years"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 387,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-387-e169cc595034>:1: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.\n",
      "  data[data['AGE']>77].mean()['CRP']\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "83.14090909090909"
      ]
     },
     "execution_count": 387,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data['AGE']>77].mean()['CRP']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 388,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['CRP'].fillna(84,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Replacing the ALT null value with its median "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 389,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['ALT']=data['ALT'].replace(np.nan,data['ALT'].median())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## handling Platelets"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Replacing the ALT null value with its mean of class 1 (There is only one missing value which is of class 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 390,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>WBC</th>\n",
       "      <th>Platelets</th>\n",
       "      <th>Neutrophils</th>\n",
       "      <th>Lymphocytes</th>\n",
       "      <th>Monocytes</th>\n",
       "      <th>Eosinophils</th>\n",
       "      <th>Basophils</th>\n",
       "      <th>CRP</th>\n",
       "      <th>AST</th>\n",
       "      <th>ALT</th>\n",
       "      <th>LDH</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>188</th>\n",
       "      <td>M</td>\n",
       "      <td>37</td>\n",
       "      <td>6.8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5.112031</td>\n",
       "      <td>1.074167</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>108.8</td>\n",
       "      <td>27.0</td>\n",
       "      <td>39.0</td>\n",
       "      <td>321.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    GENDER  AGE  WBC  Platelets  Neutrophils  Lymphocytes  Monocytes  \\\n",
       "188      M   37  6.8        NaN     5.112031     1.074167        0.5   \n",
       "\n",
       "     Eosinophils  Basophils    CRP   AST   ALT    LDH  Class  \n",
       "188          0.0        0.0  108.8  27.0  39.0  321.0      1  "
      ]
     },
     "execution_count": 390,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data['Platelets'].isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 391,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "220.72624113475177"
      ]
     },
     "execution_count": 391,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[data['Class']==1]['Platelets'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 392,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Platelets'].fillna(220,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling LDH"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 393,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['LDH'] = data['LDH'].fillna(data['LDH'].median())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 394,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GENDER         0\n",
       "AGE            0\n",
       "WBC            0\n",
       "Platelets      0\n",
       "Neutrophils    0\n",
       "Lymphocytes    0\n",
       "Monocytes      0\n",
       "Eosinophils    0\n",
       "Basophils      0\n",
       "CRP            0\n",
       "AST            0\n",
       "ALT            0\n",
       "LDH            0\n",
       "Class          0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 394,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## handling Outliers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Handling outliers by capping method (IQR)\n",
    "- Replaced outliers with whisker values\n",
    "- Some of features also replaced with median"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 395,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>AGE</th>\n",
       "      <th>WBC</th>\n",
       "      <th>Platelets</th>\n",
       "      <th>Neutrophils</th>\n",
       "      <th>Lymphocytes</th>\n",
       "      <th>Monocytes</th>\n",
       "      <th>Eosinophils</th>\n",
       "      <th>Basophils</th>\n",
       "      <th>CRP</th>\n",
       "      <th>AST</th>\n",
       "      <th>ALT</th>\n",
       "      <th>LDH</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>61.472973</td>\n",
       "      <td>8.548198</td>\n",
       "      <td>226.033333</td>\n",
       "      <td>6.658403</td>\n",
       "      <td>1.251644</td>\n",
       "      <td>0.572973</td>\n",
       "      <td>0.039640</td>\n",
       "      <td>0.010811</td>\n",
       "      <td>89.327027</td>\n",
       "      <td>52.247748</td>\n",
       "      <td>44.990991</td>\n",
       "      <td>358.324324</td>\n",
       "      <td>0.63964</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>18.361791</td>\n",
       "      <td>4.744024</td>\n",
       "      <td>106.051326</td>\n",
       "      <td>4.342108</td>\n",
       "      <td>0.750209</td>\n",
       "      <td>0.351196</td>\n",
       "      <td>0.120541</td>\n",
       "      <td>0.035215</td>\n",
       "      <td>91.881900</td>\n",
       "      <td>52.209349</td>\n",
       "      <td>43.933824</td>\n",
       "      <td>162.673070</td>\n",
       "      <td>0.48119</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.100000</td>\n",
       "      <td>20.000000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.200000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.100000</td>\n",
       "      <td>11.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>98.000000</td>\n",
       "      <td>0.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>49.000000</td>\n",
       "      <td>5.100000</td>\n",
       "      <td>157.000000</td>\n",
       "      <td>3.700000</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>0.400000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>21.950000</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>273.250000</td>\n",
       "      <td>0.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>64.000000</td>\n",
       "      <td>7.250000</td>\n",
       "      <td>199.000000</td>\n",
       "      <td>5.400000</td>\n",
       "      <td>1.074167</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>53.450000</td>\n",
       "      <td>36.000000</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>321.000000</td>\n",
       "      <td>1.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>76.000000</td>\n",
       "      <td>10.775000</td>\n",
       "      <td>272.000000</td>\n",
       "      <td>8.539669</td>\n",
       "      <td>1.566667</td>\n",
       "      <td>0.700000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>129.475000</td>\n",
       "      <td>59.750000</td>\n",
       "      <td>46.750000</td>\n",
       "      <td>390.500000</td>\n",
       "      <td>1.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>98.000000</td>\n",
       "      <td>29.200000</td>\n",
       "      <td>620.000000</td>\n",
       "      <td>24.925994</td>\n",
       "      <td>7.200000</td>\n",
       "      <td>3.200000</td>\n",
       "      <td>1.300000</td>\n",
       "      <td>0.300000</td>\n",
       "      <td>425.000000</td>\n",
       "      <td>550.000000</td>\n",
       "      <td>275.000000</td>\n",
       "      <td>1183.000000</td>\n",
       "      <td>1.00000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              AGE         WBC   Platelets  Neutrophils  Lymphocytes  \\\n",
       "count  222.000000  222.000000  222.000000   222.000000   222.000000   \n",
       "mean    61.472973    8.548198  226.033333     6.658403     1.251644   \n",
       "std     18.361791    4.744024  106.051326     4.342108     0.750209   \n",
       "min      0.000000    1.100000   20.000000     0.500000     0.200000   \n",
       "25%     49.000000    5.100000  157.000000     3.700000     0.800000   \n",
       "50%     64.000000    7.250000  199.000000     5.400000     1.074167   \n",
       "75%     76.000000   10.775000  272.000000     8.539669     1.566667   \n",
       "max     98.000000   29.200000  620.000000    24.925994     7.200000   \n",
       "\n",
       "        Monocytes  Eosinophils   Basophils         CRP         AST  \\\n",
       "count  222.000000   222.000000  222.000000  222.000000  222.000000   \n",
       "mean     0.572973     0.039640    0.010811   89.327027   52.247748   \n",
       "std      0.351196     0.120541    0.035215   91.881900   52.209349   \n",
       "min      0.000000     0.000000    0.000000    0.100000   11.000000   \n",
       "25%      0.400000     0.000000    0.000000   21.950000   27.000000   \n",
       "50%      0.500000     0.000000    0.000000   53.450000   36.000000   \n",
       "75%      0.700000     0.000000    0.000000  129.475000   59.750000   \n",
       "max      3.200000     1.300000    0.300000  425.000000  550.000000   \n",
       "\n",
       "              ALT          LDH      Class  \n",
       "count  222.000000   222.000000  222.00000  \n",
       "mean    44.990991   358.324324    0.63964  \n",
       "std     43.933824   162.673070    0.48119  \n",
       "min     10.000000    98.000000    0.00000  \n",
       "25%     22.000000   273.250000    0.00000  \n",
       "50%     32.000000   321.000000    1.00000  \n",
       "75%     46.750000   390.500000    1.00000  \n",
       "max    275.000000  1183.000000    1.00000  "
      ]
     },
     "execution_count": 395,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 396,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x245b76790a0>"
      ]
     },
     "execution_count": 396,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(data['WBC'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 397,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['WBC'] = np.where(data['WBC']>17.5, 17, data['WBC'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 398,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x245b7654820>"
      ]
     },
     "execution_count": 398,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(data['WBC'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 399,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x245b7252a60>"
      ]
     },
     "execution_count": 399,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(data['Platelets'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 400,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>AGE</th>\n",
       "      <th>WBC</th>\n",
       "      <th>Platelets</th>\n",
       "      <th>Neutrophils</th>\n",
       "      <th>Lymphocytes</th>\n",
       "      <th>Monocytes</th>\n",
       "      <th>Eosinophils</th>\n",
       "      <th>Basophils</th>\n",
       "      <th>CRP</th>\n",
       "      <th>AST</th>\n",
       "      <th>ALT</th>\n",
       "      <th>LDH</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>61.472973</td>\n",
       "      <td>8.297297</td>\n",
       "      <td>226.033333</td>\n",
       "      <td>6.658403</td>\n",
       "      <td>1.251644</td>\n",
       "      <td>0.572973</td>\n",
       "      <td>0.039640</td>\n",
       "      <td>0.010811</td>\n",
       "      <td>89.327027</td>\n",
       "      <td>52.247748</td>\n",
       "      <td>44.990991</td>\n",
       "      <td>358.324324</td>\n",
       "      <td>0.63964</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>18.361791</td>\n",
       "      <td>4.075020</td>\n",
       "      <td>106.051326</td>\n",
       "      <td>4.342108</td>\n",
       "      <td>0.750209</td>\n",
       "      <td>0.351196</td>\n",
       "      <td>0.120541</td>\n",
       "      <td>0.035215</td>\n",
       "      <td>91.881900</td>\n",
       "      <td>52.209349</td>\n",
       "      <td>43.933824</td>\n",
       "      <td>162.673070</td>\n",
       "      <td>0.48119</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.100000</td>\n",
       "      <td>20.000000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.200000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.100000</td>\n",
       "      <td>11.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>98.000000</td>\n",
       "      <td>0.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>49.000000</td>\n",
       "      <td>5.100000</td>\n",
       "      <td>157.000000</td>\n",
       "      <td>3.700000</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>0.400000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>21.950000</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>273.250000</td>\n",
       "      <td>0.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>64.000000</td>\n",
       "      <td>7.250000</td>\n",
       "      <td>199.000000</td>\n",
       "      <td>5.400000</td>\n",
       "      <td>1.074167</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>53.450000</td>\n",
       "      <td>36.000000</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>321.000000</td>\n",
       "      <td>1.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>76.000000</td>\n",
       "      <td>10.775000</td>\n",
       "      <td>272.000000</td>\n",
       "      <td>8.539669</td>\n",
       "      <td>1.566667</td>\n",
       "      <td>0.700000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>129.475000</td>\n",
       "      <td>59.750000</td>\n",
       "      <td>46.750000</td>\n",
       "      <td>390.500000</td>\n",
       "      <td>1.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>98.000000</td>\n",
       "      <td>17.000000</td>\n",
       "      <td>620.000000</td>\n",
       "      <td>24.925994</td>\n",
       "      <td>7.200000</td>\n",
       "      <td>3.200000</td>\n",
       "      <td>1.300000</td>\n",
       "      <td>0.300000</td>\n",
       "      <td>425.000000</td>\n",
       "      <td>550.000000</td>\n",
       "      <td>275.000000</td>\n",
       "      <td>1183.000000</td>\n",
       "      <td>1.00000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              AGE         WBC   Platelets  Neutrophils  Lymphocytes  \\\n",
       "count  222.000000  222.000000  222.000000   222.000000   222.000000   \n",
       "mean    61.472973    8.297297  226.033333     6.658403     1.251644   \n",
       "std     18.361791    4.075020  106.051326     4.342108     0.750209   \n",
       "min      0.000000    1.100000   20.000000     0.500000     0.200000   \n",
       "25%     49.000000    5.100000  157.000000     3.700000     0.800000   \n",
       "50%     64.000000    7.250000  199.000000     5.400000     1.074167   \n",
       "75%     76.000000   10.775000  272.000000     8.539669     1.566667   \n",
       "max     98.000000   17.000000  620.000000    24.925994     7.200000   \n",
       "\n",
       "        Monocytes  Eosinophils   Basophils         CRP         AST  \\\n",
       "count  222.000000   222.000000  222.000000  222.000000  222.000000   \n",
       "mean     0.572973     0.039640    0.010811   89.327027   52.247748   \n",
       "std      0.351196     0.120541    0.035215   91.881900   52.209349   \n",
       "min      0.000000     0.000000    0.000000    0.100000   11.000000   \n",
       "25%      0.400000     0.000000    0.000000   21.950000   27.000000   \n",
       "50%      0.500000     0.000000    0.000000   53.450000   36.000000   \n",
       "75%      0.700000     0.000000    0.000000  129.475000   59.750000   \n",
       "max      3.200000     1.300000    0.300000  425.000000  550.000000   \n",
       "\n",
       "              ALT          LDH      Class  \n",
       "count  222.000000   222.000000  222.00000  \n",
       "mean    44.990991   358.324324    0.63964  \n",
       "std     43.933824   162.673070    0.48119  \n",
       "min     10.000000    98.000000    0.00000  \n",
       "25%     22.000000   273.250000    0.00000  \n",
       "50%     32.000000   321.000000    1.00000  \n",
       "75%     46.750000   390.500000    1.00000  \n",
       "max    275.000000  1183.000000    1.00000  "
      ]
     },
     "execution_count": 400,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 401,
   "metadata": {},
   "outputs": [],
   "source": [
    "iqr = 272 - 157 \n",
    "lb = 157 - iqr*1.5\n",
    "ub = 272 + iqr*1.5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 402,
   "metadata": {},
   "outputs": [],
   "source": [
    "for ind in data.index:\n",
    "    if data.loc[ind, 'Platelets'] < lb:\n",
    "        data.loc[ind, 'Platelets'] = lb\n",
    "    elif data.loc[ind, 'Platelets'] > ub:\n",
    "        data.loc[ind, 'Platelets'] = ub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 403,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x245b7134b80>"
      ]
     },
     "execution_count": 403,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAWAAAAEGCAYAAABbzE8LAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAANTklEQVR4nO3df2xddRnH8c+zdgsFJbAOF9LBChSCZsaJDYmBmJmsZuMf/B2JybpERC10c5FENCjEYEJElKVRE1CkVVFJhvwIZa5LJAh/6Fpc2OyG3kDRFRyjMwJusrV7/OOeS27vetvb7pz73N37fiUL7bk93/P97lzeuztdzzV3FwCg+hZFTwAAGhUBBoAgBBgAghBgAAhCgAEgSPN8vnjZsmXe3t6e0VQAoD6NjIy87u7nlW6fV4Db29s1PDyc3qwAoAGY2cszbecSBAAEIcAAEIQAA0AQAgwAQQgwAAQhwAAQhAADQBACDABBCDAABCHAABCEAANAEAIMAEEIMAAEIcAAEIQAA0AQAgwAQQgwAAQhwAAQhAADQJB5vSdcverr61Mul0t1zPHxcUlSW1tbquOmpaOjQ729vdHTABoaAZaUy+W0e+8+TZ25NLUxm478R5L0r7dr77e46cjh6CkAEAF+x9SZS3X08mtSG69l/6AkpTpmWgpzAxCLa8AAEIQAA0AQAgwAQQgwAAQhwAAQhAADQBACDABBCDAABCHAABCEAANAEAIMAEEIMAAEIcAAEIQAA0AQAgwAQQgwAAQhwAAQhAADQBACDABBCDAABCHAABCEAANAEAIMAEEIMAAEIcAAEIQAA0AQAgwAQQgwAAQhwAAQhAADQBACDABBCDAABCHAABCEAANAEAIMAEEIMAAEIcAAEIQAA0AQAgwAQQgwAAQhwAAQpCoB7uvrU19fXzUOBdQEnvOoRHM1DpLL5apxGKBm8JxHJbgEAQBBCDAABCHAABCEAANAEAIMAEEIMAAEIcAAEIQAA0AQAgwAQQgwAAQhwAAQhAADQBACDABBCDAABCHAABCEAANAEAIMAEEIMAAEIcAAEIQAA0AQAgwAQQgwAAQhwAAQhAADQBACDABBCDAABCHAABCEAANAEAIMAEEIMAAEIcAAEIQAA0AQAgwAQQgwAAQhwAAQhAADQBACDABBCDAABCHAQMYmJibU09Oj66+/Xj09PZqYmJj22KZNm6Ztq3TM0v1Kt8029kKPm9b+tWSutWS5VgIMZKy/v1+jo6PK5XIaHR3VwMDAtMf27NkzbVulY5buV7pttrEXety09q8lc60ly7USYCBDExMT2r59+7RtTz75pCYmJt55zN21ffv2il9hzbRf6bZcLld27IUeN639a8lca8l6rc2pjlbG+Pi4jh49qs2bN1fjcPOWy+W06JhHT6NqFv3vDeVyb9bs+agHuVxOLS0t6u/v1/Hjx6c9dvz4cQ0MDMjddeLECUnS1NSUBgYGtGXLljnH7u/vP2m/0rHuuOOOsmPPtH8lx53t+PPZv5bMtZas1zrnK2Azu8HMhs1s+NChQ6kdGGgEO3fulPv0P9zdXUNDQ9q5c6cmJyclSZOTkxoaGqp4zNL9SreNjY2VHXuhx01r/1oy11qyXuucr4Dd/V5J90pSZ2fngl4mtrW1SZK2bt26kN0zt3nzZo28eDB6GlVz4oyz1XHx8po9H/Wg8LeLlStX6vHHH58WYTNTV1eX3F2Dg4OanJxUc3Ozurq6Khp77dq1J+1XOtaKFSt04MCBGceeaf/5ONX9a8lca8l6rVwDBjLU3d2txYsXT9u2ePFibdiwQd3d3Vq0KP+/YFNTkzZs2FDxmKX7lW679dZby4690OOmtX8tmWstWa+VAAMZam1t1bp166ZtW79+vVpbW995zMy0bt06tba2zmvM4v1Kt3V0dJQde6HHTWv/WjLXWrJea1W+CQc0su7ubuVyOR07dkxLliw56dXo2NjYgl6Flu5Xum22sRd63LT2ryVzrSXLtVrpNwhm09nZ6cPDw/M+SOF6WK1ecyxcAz56+TWpjdmyf1CSUh0zLS37B/UhrgFnqtaf86guMxtx987S7VyCAIAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAgjRX4yAdHR3VOAxQM3jOoxJVCXBvb281DgPUDJ7zqASXIAAgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACEKAASAIAQaAIAQYAIIQYAAIQoABIAgBBoAgBBgAghBgAAhCgAEgCAEGgCAEGACCEGAACNIcPYFa0XTksFr2D6Y43oQkpTpmWpqOHJa0PHoaQMMjwJI6OjpSH3N8fFKS1NZWi6FbnsmaAcwPAZbU29sbPQUADYhrwAAQhAADQBACDABBCDAABCHAABCEAANAEAIMAEEIMAAEIcAAEIQAA0AQAgwAQQgwAAQhwAAQhAADQBACDABBCDAABCHAABCEAANAEAIMAEEIMAAEMXev/IvNDkl6eY4vWybp9VOZ1GmKdTeeRl07656/le5+XunGeQW4EmY27O6dqQ56GmDdjadR186608MlCAAIQoABIEgWAb43gzFPB6y78TTq2ll3SlK/BgwAqAyXIAAgCAEGgCCpBdjM1pnZC2aWM7Nb0hr3dGBmY2a2x8x2m9lw9HyyYmb3m9lrZra3aNtSMxsys78n/z03co5ZKLPu281sPDnnu83smsg5ZsHMLjCzP5jZPjP7q5ltTrY3wjkvt/ZUz3sq14DNrEnS3yR1STogaZek69x99JQHPw2Y2ZikTnev63+cbmYfkfSWpAF3X5Vs+56kw+5+Z/IH77nu/vXIeaatzLpvl/SWu38/cm5ZMrPzJZ3v7s+Z2bsljUj6uKSNqv9zXm7tn1WK5z2tV8BXSsq5+4vufkzSbyRdm9LYqBHu/rSkwyWbr5XUn3zcr/yTtK6UWXfdc/dX3f255OM3Je2T1KbGOOfl1p6qtALcJumfRZ8fUAaTrWEuaYeZjZjZDdGTqbLl7v6qlH/SSnpP8Hyq6SYzez65RFF3fw0vZmbtkj4o6U9qsHNesnYpxfOeVoBthm2N9O/brnL3KyStl3Rj8ldW1LefSLpE0mpJr0q6O3Y62TGzd0naJumr7v5G9HyqaYa1p3re0wrwAUkXFH2+QtIrKY1d89z9leS/r0n6nfKXZBrFweR6WeG62WvB86kKdz/o7lPufkLSfarTc25mi5UP0K/c/eFkc0Oc85nWnvZ5TyvAuyRdamYXmdkSSZ+T9FhKY9c0MzsruUgvMztL0sck7Z19r7rymKTu5ONuSY8GzqVqCgFKfEJ1eM7NzCT9TNI+d/9B0UN1f87LrT3t857aT8Il/xzjHklNku539++mMnCNM7OLlX/VK0nNkh6s17Wb2a8lrVH+tnwHJd0m6RFJD0m6UNI/JH3G3evqG1Zl1r1G+b+GuqQxSV8qXBetF2Z2taQ/Stoj6USy+ZvKXwut93Nebu3XKcXzzo8iA0AQfhIOAIIQYAAIQoABIAgBBoAgBBgAghBgZMLM3MzuLvr85uQGNgsZ6xwz60lxbu3FdzYreew7ZrY2+fgpM2u4N59E9RBgZOVtSZ80s2UpjHWOpBkDnNyJLzXu/m1335nmmEA5BBhZmVT+PbS2lD5gZueZ2TYz25X8uirZfruZ3Vz0dXuTG6HcKemS5P6rd5nZmuRerQ9K2mNmZ5jZz5N7Mv/FzD6a7L/RzB41s+3JvapvK5pGk5ndl9zrdYeZtST7PGBmny6Zb1OyfW9yjJPWBCxEc/QEUNd+JOn55J7BxbZK+qG7P2NmF0r6vaT3zjLOLZJWuftqSTKzNcr/DP4qd3/JzL4mSe7+fjO7XPk7012W7HulpFWSjkjaZWZPSHpd0qXK37P6i2b2kKRPSfplmeOvltRWdC/gcyr/LQDKI8DIjLu/YWYDkjZJOlr00FpJ78v/uL0k6ezC/TTm4c/u/lLy8dWS+pJj7jezlyUVAjzk7hOSZGYPJ1/7iKSX3H138jUjktpnOdaLki42sz5JT0jaMc+5AjPiEgSydo+kL0g6q2jbIkkfdvfVya+25KbXk5r+nDxjlnH/W/TxTLdDLSj9WfvC528XbZvSLC9G3P3fkj4g6SlJN0r66SzHAypGgJGp5CYtDykf4YIdkm4qfGJmq5MPxyRdkWy7QtJFyfY3Jc32CvlpSZ9P9rtM+ZvEvJA81mX59zBrUf6dG56d7xqSbyQucvdtkr5VmCNwqggwquFu5e8kVrBJUmfyrgKjkr6cbN8maamZ7Zb0FeXfZ1DJJYRnk2+C3TXD+D9W/ptqeyT9VtJGdy+8wn1G0i8k7Za0zd0X8qapbZKeSub1gKRvLGAM4CTcDQ11y8w2Kv9mqTfN9bVABF4BA0AQXgEDQBBeAQNAEAIMAEEIMAAEIcAAEIQAA0CQ/wPqzur4mTHjigAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(data['Neutrophils'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 404,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Neutrophils'] = np.where(data['Neutrophils']>15, 15, data['Neutrophils'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 405,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x245b7489160>"
      ]
     },
     "execution_count": 405,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(data['Lymphocytes'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 406,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Lymphocytes'] = np.where(data['Lymphocytes']>2.9, 2.9, data['Lymphocytes'] )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 407,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x245b74c6fa0>"
      ]
     },
     "execution_count": 407,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAWQAAAEGCAYAAABSJ+9xAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAMWklEQVR4nO3dX4yld13H8c93d0u6iAbJViQrspCFiICW2iAE0AoYCTfcoKLFRpqUQHSzEG+QoMY7LxSCSxS70MLCRoUABkiJGBSJFyJtKf1jC45NCCwIhcZC6fKn7c+L80ycbne7s7MzZ77nzOuVTHrOnOec5/eb3/Q9zz4z80yNMQLA9tu13QMAYEaQAZoQZIAmBBmgCUEGaGLPuWy8b9++ceDAgS0aCsByuuGGG745xrjobNudU5APHDiQ66+/fuOjAtiBqupL69nOKQuAJgQZoAlBBmhCkAGaEGSAJgQZoAlBBmhCkAGaEGSAJgQZoAlBBmhCkAGaEGSAJgQZoAlBBmhCkAGaEGSAJgQZoAlBBmjinP6mHpvjyJEjWVlZmft+T5w4kSTZv3//3Pc9LwcPHsyhQ4e2exiwIYK8DVZWVnLTrbfngUc/bq773X3fPUmS//n+ci777vvu3u4hwHlZzv8zF8ADj35cTv7My+a6z713XJckc9/vvKzODxaVc8gATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATcwlyEeOHMmRI0fmsStgh1qGzuyZx05WVlbmsRtgB1uGzjhlAdCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgATQgyQBOCDNCEIAM0IcgAj+Do0aO57LLLcs0112z5vgQZ4BEcP348SXLs2LEt35cgA5zB0aNHH3J/q4+S92zpq09OnDiRkydP5vDhw/PYXXsrKyvZ9YOx3cNYOru+9+2srHzH59kOtbKykr17927qa64eHa86duxYrrzyyk3dx1pnPUKuqtdU1fVVdf1dd921ZQMB2OnOeoQ8xrg6ydVJcumll27osG7//v1Jkre97W0befrSOXz4cG648+vbPYyl8+CFP5aDT3m8z7Mdahn+ZeQcMsAZXH755Q+5f8UVV2zp/gQZ4Ayuuuqqh9zfyvPHiSADPKLVo+StPjpO5vRTFgCL6qqrrnrYkfJWcYQM0IQgAzQhyABNCDJAE4IM0IQgAzQhyABNCDJAE4IM0IQgAzQhyABNCDJAE4IM0IQgAzQhyABNCDJAE4IM0IQgAzQhyABNCDJAE4IM0IQgAzQhyABNCDJAE4IM0IQgAzQhyABNCDJAE4IM0IQgAzQhyABNCDJAE4IM0IQgAzQhyABNCDJAE4IM0IQgAzSxZx47OXjw4Dx2A+xgy9CZuQT50KFD89gNsIMtQ2ecsgBoQpABmhBkgCYEGaAJQQZoQpABmhBkgCYEGaAJQQZoQpABmhBkgCYEGaAJQQZoQpABmhBkgCYEGaAJQQZoQpABmhBkgCYEGaAJQQZoQpABmhBkgCYEGaAJQQZoQpABmhBkgCYEGaAJQQZoQpABmhBkgCYEGaAJQQZoQpABmhBkgCYEGaAJQQZoQpABmhBkgCYEGaCJPds9gJ1q9313Z+8d1815n99Kkrnvd15233d3ksdv9zBgwwR5Gxw8eHBb9nvixP1Jkv37lzVaj9+2jy1sBkHeBocOHdruIQANOYcM0IQgAzQhyABNCDJAE4IM0IQgAzQhyABNCDJAE4IM0IQgAzQhyABNCDJAE4IM0IQgAzQhyABNCDJAE4IM0IQgAzQhyABNCDJAEzXGWP/GVXcl+dLWDWcu9iX55nYPYguZ3+Jb9jku+/ySh8/xSWOMi872pHMK8jKoquvHGJdu9zi2ivktvmWf47LPL9n4HJ2yAGhCkAGa2IlBvnq7B7DFzG/xLfscl31+yQbnuOPOIQN0tROPkAFaEmSAJpYyyFX10qr6QlWtVNUbT/P4ZVV1T1XdNL398XaMc6Oq6pqq+kZV3XqGx6uq/nKa/81Vdcm8x3i+1jHHRV/DJ1bVv1TV7VV1W1UdPs02C7uO65zfwq5hVV1YVf9RVZ+f5venp9nm3NdvjLFUb0l2J/nvJE9J8qgkn0/ys6dsc1mSj233WM9jjr+U5JIkt57h8Zcl+XiSSvLcJJ/Z7jFvwRwXfQ2fkOSS6faPJvniaT5PF3Yd1zm/hV3DaU0eM92+IMlnkjz3fNdvGY+Qn5NkZYxx5xjjB0n+LsnLt3lMm2qM8ekkdz/CJi9PcmzM/HuSx1bVE+Yzus2xjjkutDHG18YYN063v5Pk9iT7T9lsYddxnfNbWNOa3DvdvWB6O/UnJM55/ZYxyPuTfHnN/a/k9J8Iz5v+ufHxqnrGfIY2N+v9GCy6pVjDqjqQ5NmZHWWttRTr+AjzSxZ4Datqd1XdlOQbSf5pjHHe67dnc4fYQp3mfad+5boxs98tv7eqXpbkH5I8dctHNj/r+RgsuqVYw6p6TJIPJnn9GOPbpz58mqcs1DqeZX4LvYZjjAeSXFxVj03y4ap65hhj7fc8znn9lvEI+StJnrjm/k8l+eraDcYY317958YY47okF1TVvvkNccud9WOw6JZhDavqgsxidXyM8aHTbLLQ63i2+S3DGibJGON/k3wqyUtPeeic128Zg/zZJE+tqidX1aOSvDLJR9ZuUFU/WVU13X5OZh+Hb819pFvnI0mumL7L+9wk94wxvrbdg9pMi76G09jfleT2McZbzrDZwq7jeua3yGtYVRdNR8apqr1JXpLkjlM2O+f1W7pTFmOM+6vq95P8Y2Y/cXHNGOO2qnrt9Pg7krwiyeuq6v4kJ5O8ckzfFl0EVfW3mX2Hel9VfSXJn2T2TYXV+V2X2Xd4V5Lcl+TV2zPSjVvHHBd6DZM8P8nvJLllOg+ZJG9K8tPJUqzjeua3yGv4hCTvqardmX0hef8Y42OndOac18+vTgM0sYynLAAWkiADNCHIAE0IMkATggzQhCCzKarq3rNvtSn7+VRVbdkfyKyqN23Va8PZCDI8lCCzbQSZTVdVu6rqv6rqojX3V6pqX1W9u6r+erpW7p1V9cs1u/bx7VX17jWvcW9V/UVV3VhVn1x9rcmvT9ei/WJVvXDa/sKquraqbqmqz1XVr0zv311Vfz69/+aqOlRVL66qD6/Z169W1Yeq6s+S7K3ZtXmPT4+9atrXTVX1N9Pr7Z7mcev0um+Yw4eVHUCQ2XRjjAeTvC/J5dO7XpLk82OMb073fzzJi5K8IclHk7w1yTOSPKuqLp62+ZEkN44xLknyr5n9pt6qPWOM5yR5/Zr3/96072cl+a3MfovqwiSvSfLkJM8eY/xckuNJ/jnJ09dE/tVJrh1jvDHJyTHGxWOMy6vq6Ul+M8nzxxgXJ3lgmtPFSfaPMZ457e/a8/yQQRJBZutck+SK6faVeWi0Pjr9iuwtSb4+xrhlivhtSQ5M2zyY5O+n2+9L8oI1z1+9UM0Na7Z/QZL3JskY444kX0rytMy+GLxjjHH/9Njd077fm+RV0/UInpfZhcRP9eIkv5Dks9Ov/744sz98cGeSp1TVkap6aZJTr2IGG7J017KghzHGl6vq61X1oiS/mP8/Wk6S70//fXDN7dX7Z/qcXPs7/qvPeWDN9qe71OHq+093fYBrMzs6/16SD6wG+zTPfc8Y4w8f9kDVzyf5tcyOzH8jsy86cF4cIbOV3pnZ0e37p2vHnotdmV18Jkl+O8m/nWX7T2eKflU9LbOL2HwhySeSvLaq9kyPPS5JxhhfzexSiG9O8u41r/PD6bKRSfLJJK+oqp9YfW5VPWm6ROSuMcYHk/xRZn9qCs6bI2Q2y6Onq7KtekuSI5kdiW7kHOt3kzyjqm5Ick9m53IfyV8leUdV3ZLk/iS/O8b4flW9M7NTFzdX1Q+THE3y9uk5x5NcNMb4zzWvc/W07Y3TeeQ3J/lEVe1K8sPMjohPJrl2el+SPOwIGjbC1d7YMtPPC791jPHCDTz33jHGY7ZgWGv38fYknxtjvGsr9wPr5QiZLVFVb0zyujz03HEb05H3d5P8wXaPBVY5QgZowjf1AJoQZIAmBBmgCUEGaEKQAZr4P9KUtceN79k/AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(data['Lymphocytes'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 408,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>AGE</th>\n",
       "      <th>WBC</th>\n",
       "      <th>Platelets</th>\n",
       "      <th>Neutrophils</th>\n",
       "      <th>Lymphocytes</th>\n",
       "      <th>Monocytes</th>\n",
       "      <th>Eosinophils</th>\n",
       "      <th>Basophils</th>\n",
       "      <th>CRP</th>\n",
       "      <th>AST</th>\n",
       "      <th>ALT</th>\n",
       "      <th>LDH</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.000000</td>\n",
       "      <td>222.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>61.472973</td>\n",
       "      <td>8.297297</td>\n",
       "      <td>222.337387</td>\n",
       "      <td>6.463853</td>\n",
       "      <td>1.221914</td>\n",
       "      <td>0.572973</td>\n",
       "      <td>0.039640</td>\n",
       "      <td>0.010811</td>\n",
       "      <td>89.327027</td>\n",
       "      <td>52.247748</td>\n",
       "      <td>44.990991</td>\n",
       "      <td>358.324324</td>\n",
       "      <td>0.63964</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>18.361791</td>\n",
       "      <td>4.075020</td>\n",
       "      <td>95.788659</td>\n",
       "      <td>3.791923</td>\n",
       "      <td>0.609604</td>\n",
       "      <td>0.351196</td>\n",
       "      <td>0.120541</td>\n",
       "      <td>0.035215</td>\n",
       "      <td>91.881900</td>\n",
       "      <td>52.209349</td>\n",
       "      <td>43.933824</td>\n",
       "      <td>162.673070</td>\n",
       "      <td>0.48119</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.100000</td>\n",
       "      <td>20.000000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.200000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.100000</td>\n",
       "      <td>11.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>98.000000</td>\n",
       "      <td>0.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>49.000000</td>\n",
       "      <td>5.100000</td>\n",
       "      <td>157.000000</td>\n",
       "      <td>3.700000</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>0.400000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>21.950000</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>273.250000</td>\n",
       "      <td>0.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>64.000000</td>\n",
       "      <td>7.250000</td>\n",
       "      <td>199.000000</td>\n",
       "      <td>5.400000</td>\n",
       "      <td>1.074167</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>53.450000</td>\n",
       "      <td>36.000000</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>321.000000</td>\n",
       "      <td>1.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>76.000000</td>\n",
       "      <td>10.775000</td>\n",
       "      <td>272.000000</td>\n",
       "      <td>8.539669</td>\n",
       "      <td>1.566667</td>\n",
       "      <td>0.700000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>129.475000</td>\n",
       "      <td>59.750000</td>\n",
       "      <td>46.750000</td>\n",
       "      <td>390.500000</td>\n",
       "      <td>1.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>98.000000</td>\n",
       "      <td>17.000000</td>\n",
       "      <td>444.500000</td>\n",
       "      <td>15.000000</td>\n",
       "      <td>2.900000</td>\n",
       "      <td>3.200000</td>\n",
       "      <td>1.300000</td>\n",
       "      <td>0.300000</td>\n",
       "      <td>425.000000</td>\n",
       "      <td>550.000000</td>\n",
       "      <td>275.000000</td>\n",
       "      <td>1183.000000</td>\n",
       "      <td>1.00000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              AGE         WBC   Platelets  Neutrophils  Lymphocytes  \\\n",
       "count  222.000000  222.000000  222.000000   222.000000   222.000000   \n",
       "mean    61.472973    8.297297  222.337387     6.463853     1.221914   \n",
       "std     18.361791    4.075020   95.788659     3.791923     0.609604   \n",
       "min      0.000000    1.100000   20.000000     0.500000     0.200000   \n",
       "25%     49.000000    5.100000  157.000000     3.700000     0.800000   \n",
       "50%     64.000000    7.250000  199.000000     5.400000     1.074167   \n",
       "75%     76.000000   10.775000  272.000000     8.539669     1.566667   \n",
       "max     98.000000   17.000000  444.500000    15.000000     2.900000   \n",
       "\n",
       "        Monocytes  Eosinophils   Basophils         CRP         AST  \\\n",
       "count  222.000000   222.000000  222.000000  222.000000  222.000000   \n",
       "mean     0.572973     0.039640    0.010811   89.327027   52.247748   \n",
       "std      0.351196     0.120541    0.035215   91.881900   52.209349   \n",
       "min      0.000000     0.000000    0.000000    0.100000   11.000000   \n",
       "25%      0.400000     0.000000    0.000000   21.950000   27.000000   \n",
       "50%      0.500000     0.000000    0.000000   53.450000   36.000000   \n",
       "75%      0.700000     0.000000    0.000000  129.475000   59.750000   \n",
       "max      3.200000     1.300000    0.300000  425.000000  550.000000   \n",
       "\n",
       "              ALT          LDH      Class  \n",
       "count  222.000000   222.000000  222.00000  \n",
       "mean    44.990991   358.324324    0.63964  \n",
       "std     43.933824   162.673070    0.48119  \n",
       "min     10.000000    98.000000    0.00000  \n",
       "25%     22.000000   273.250000    0.00000  \n",
       "50%     32.000000   321.000000    1.00000  \n",
       "75%     46.750000   390.500000    1.00000  \n",
       "max    275.000000  1183.000000    1.00000  "
      ]
     },
     "execution_count": 408,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Encoding GENDER Column with Male as 1 and Female as 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 409,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['GENDER']=data['GENDER'].map({'M':1,'F':0})"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Splitting X and Y data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 410,
   "metadata": {},
   "outputs": [],
   "source": [
    "X=data.drop('Class',axis=1)\n",
    "y=data['Class']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 411,
   "metadata": {},
   "outputs": [],
   "source": [
    "data.to_csv('data_cleaned_final.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Scaling the Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 412,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler=StandardScaler()\n",
    "X=data.drop('Class',axis=1)\n",
    "X_scaled = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)\n",
    "y=data['Class']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 413,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>WBC</th>\n",
       "      <th>Platelets</th>\n",
       "      <th>Neutrophils</th>\n",
       "      <th>Lymphocytes</th>\n",
       "      <th>Monocytes</th>\n",
       "      <th>Eosinophils</th>\n",
       "      <th>Basophils</th>\n",
       "      <th>CRP</th>\n",
       "      <th>AST</th>\n",
       "      <th>ALT</th>\n",
       "      <th>LDH</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-1.443376</td>\n",
       "      <td>0.465439</td>\n",
       "      <td>-0.835573</td>\n",
       "      <td>-0.809200</td>\n",
       "      <td>-0.651232</td>\n",
       "      <td>-0.858088</td>\n",
       "      <td>-0.779023</td>\n",
       "      <td>-0.32959</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>-0.015566</td>\n",
       "      <td>0.302395</td>\n",
       "      <td>-0.091046</td>\n",
       "      <td>1.525977</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.692820</td>\n",
       "      <td>-0.025817</td>\n",
       "      <td>0.738521</td>\n",
       "      <td>-0.589472</td>\n",
       "      <td>1.119675</td>\n",
       "      <td>-1.186912</td>\n",
       "      <td>-1.349793</td>\n",
       "      <td>-0.32959</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>0.926896</td>\n",
       "      <td>1.262242</td>\n",
       "      <td>1.140853</td>\n",
       "      <td>2.770538</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.443376</td>\n",
       "      <td>0.465439</td>\n",
       "      <td>1.845306</td>\n",
       "      <td>-0.212795</td>\n",
       "      <td>1.965481</td>\n",
       "      <td>-0.364853</td>\n",
       "      <td>0.647901</td>\n",
       "      <td>-0.32959</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>-0.496615</td>\n",
       "      <td>-0.619058</td>\n",
       "      <td>-0.433240</td>\n",
       "      <td>-0.759825</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.692820</td>\n",
       "      <td>1.284199</td>\n",
       "      <td>0.566355</td>\n",
       "      <td>0.048786</td>\n",
       "      <td>0.531133</td>\n",
       "      <td>0.566813</td>\n",
       "      <td>-0.208254</td>\n",
       "      <td>-0.32959</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>-0.905670</td>\n",
       "      <td>-0.407892</td>\n",
       "      <td>-0.661370</td>\n",
       "      <td>-0.229962</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-1.443376</td>\n",
       "      <td>1.502535</td>\n",
       "      <td>-0.933953</td>\n",
       "      <td>-0.547619</td>\n",
       "      <td>-0.704095</td>\n",
       "      <td>-1.186912</td>\n",
       "      <td>-1.064408</td>\n",
       "      <td>-0.32959</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>0.482935</td>\n",
       "      <td>0.091229</td>\n",
       "      <td>-0.638557</td>\n",
       "      <td>1.649201</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     GENDER       AGE       WBC  Platelets  Neutrophils  Lymphocytes  \\\n",
       "0 -1.443376  0.465439 -0.835573  -0.809200    -0.651232    -0.858088   \n",
       "1  0.692820 -0.025817  0.738521  -0.589472     1.119675    -1.186912   \n",
       "2 -1.443376  0.465439  1.845306  -0.212795     1.965481    -0.364853   \n",
       "3  0.692820  1.284199  0.566355   0.048786     0.531133     0.566813   \n",
       "4 -1.443376  1.502535 -0.933953  -0.547619    -0.704095    -1.186912   \n",
       "\n",
       "   Monocytes  Eosinophils  Basophils       CRP       AST       ALT       LDH  \n",
       "0  -0.779023     -0.32959  -0.307692 -0.015566  0.302395 -0.091046  1.525977  \n",
       "1  -1.349793     -0.32959  -0.307692  0.926896  1.262242  1.140853  2.770538  \n",
       "2   0.647901     -0.32959  -0.307692 -0.496615 -0.619058 -0.433240 -0.759825  \n",
       "3  -0.208254     -0.32959  -0.307692 -0.905670 -0.407892 -0.661370 -0.229962  \n",
       "4  -1.064408     -0.32959  -0.307692  0.482935  0.091229 -0.638557  1.649201  "
      ]
     },
     "execution_count": 413,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_scaled.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Over Sampling the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 414,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x245b74f09d0>"
      ]
     },
     "execution_count": 414,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data['Class'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The data is imbalanced we are performing Oversampling inorder to prevent bias"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 415,
   "metadata": {},
   "outputs": [],
   "source": [
    "import imblearn\n",
    "from imblearn.over_sampling import RandomOverSampler\n",
    "ros = RandomOverSampler()\n",
    "X_ros,y_ros=ros.fit_resample(X_scaled,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 416,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x245b63acb50>"
      ]
     },
     "execution_count": 416,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(y_ros)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Validating func\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 417,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,matthews_corrcoef\n",
    "from sklearn.model_selection import KFold\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "cv = KFold(n_splits=10, random_state=1, shuffle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 527,
   "metadata": {},
   "outputs": [],
   "source": [
    "def validate(model,xd,yd):\n",
    "    yp = model.predict(xd)\n",
    "    print(\"Results on Train set :\",accuracy_score(yd,yp))\n",
    "    print(\"MCC SCORE : \",matthews_corrcoef(yd,yp))\n",
    "    print(\"Accuracy :\", accuracy_score(yd,yp))\n",
    "    print(\"Precision :\",precision_score(yd,yp))   \n",
    "    print(\"Recall :\",recall_score(yd,yp))\n",
    "    scores = cross_val_score(model, xd, yd, scoring='accuracy', cv=cv, n_jobs=-1)\n",
    "    print(\"Cross validation scores :\",scores)\n",
    "    print(\"Mean Score : \",scores.mean())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 419,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 420,
   "metadata": {},
   "outputs": [],
   "source": [
    "logreg = LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 421,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 421,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logreg.fit(X_ros,y_ros)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 422,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Results on Train set : 0.8133802816901409\n",
      "MCC SCORE :  0.6275234912262374\n",
      "Accuracy : 0.8133802816901409\n",
      "Precision : 0.8296296296296296\n",
      "Recall : 0.7887323943661971\n",
      "Cross validation scores : [0.82758621 0.72413793 0.75862069 0.75862069 0.78571429 0.82142857\n",
      " 0.78571429 0.78571429 0.75       0.89285714]\n",
      "Mean Score :  0.7890394088669951\n"
     ]
    }
   ],
   "source": [
    "validate(logreg,X_ros,y_ros)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Decision Tree "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 529,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(criterion='entropy', max_depth=7, max_features='auto')"
      ]
     },
     "execution_count": 529,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "dt = DecisionTreeClassifier(max_features='auto', max_depth = 7, criterion = \"entropy\")\n",
    "dt.fit(X_ros, y_ros)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 530,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Results on Train set : 0.971830985915493\n",
      "MCC SCORE :  0.9445054896779567\n",
      "Accuracy : 0.971830985915493\n",
      "Precision : 0.9926470588235294\n",
      "Recall : 0.9507042253521126\n",
      "Cross validation scores : [0.89655172 0.82758621 0.93103448 0.82758621 0.89285714 0.92857143\n",
      " 0.82142857 0.89285714 0.92857143 0.75      ]\n",
      "Mean Score :  0.869704433497537\n"
     ]
    }
   ],
   "source": [
    "validate(dt,X_ros, y_ros)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 485,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SVC()"
      ]
     },
     "execution_count": 485,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.svm import SVC\n",
    "svm = SVC()\n",
    "svm.fit(X_scaled, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 528,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Results on Train set : 0.8468468468468469\n",
      "MCC SCORE :  0.6624859374891606\n",
      "Accuracy : 0.8468468468468469\n",
      "Precision : 0.86\n",
      "Recall : 0.9084507042253521\n",
      "Cross validation scores : [0.73913043 0.86956522 0.77272727 0.72727273 0.68181818 0.72727273\n",
      " 0.77272727 0.68181818 0.81818182 0.72727273]\n",
      "Mean Score :  0.7517786561264823\n"
     ]
    }
   ],
   "source": [
    "validate(svm,X_scaled, y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Random forest Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 427,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import RandomizedSearchCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 316,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [10, 120, 230, 340, 450, 560, 670, 780, 890, 1000], 'min_samples_split': [1, 2, 3, 4, 5, 7, 9, 11], 'min_samples_leaf': [1, 2, 4, 6, 8], 'criterion': ['entropy', 'gini']}\n"
     ]
    }
   ],
   "source": [
    "\n",
    "n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]\n",
    "max_features = ['auto', 'sqrt','log2']\n",
    "max_depth = [int(x) for x in np.linspace(10, 1000,10)]\n",
    "min_samples_split = [1, 2, 3, 4, 5, 7, 9, 11]\n",
    "min_samples_leaf = [1, 2, 4, 6, 8]\n",
    "random_grid = {'n_estimators': n_estimators,\n",
    "               'max_features': max_features,\n",
    "               'max_depth': max_depth,\n",
    "               'min_samples_split': min_samples_split,\n",
    "               'min_samples_leaf': min_samples_leaf,\n",
    "              'criterion':['entropy','gini']}\n",
    "print(random_grid)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 100 candidates, totalling 300 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py:372: FitFailedWarning: \n",
      "30 fits failed out of a total of 300.\n",
      "The score on these train-test partitions for these parameters will be set to nan.\n",
      "If these failures are not expected, you can try to debug them by setting error_score='raise'.\n",
      "\n",
      "Below are more details about the failures:\n",
      "--------------------------------------------------------------------------------\n",
      "30 fits failed with the following error:\n",
      "Traceback (most recent call last):\n",
      "  File \"C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_validation.py\", line 680, in _fit_and_score\n",
      "    estimator.fit(X_train, y_train, **fit_params)\n",
      "  File \"C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\sklearn\\ensemble\\_forest.py\", line 450, in fit\n",
      "    trees = Parallel(\n",
      "  File \"C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\", line 1029, in __call__\n",
      "    if self.dispatch_one_batch(iterator):\n",
      "  File \"C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\", line 847, in dispatch_one_batch\n",
      "    self._dispatch(tasks)\n",
      "  File \"C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\", line 765, in _dispatch\n",
      "    job = self._backend.apply_async(batch, callback=cb)\n",
      "  File \"C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\joblib\\_parallel_backends.py\", line 208, in apply_async\n",
      "    result = ImmediateResult(func)\n",
      "  File \"C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\joblib\\_parallel_backends.py\", line 572, in __init__\n",
      "    self.results = batch()\n",
      "  File \"C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\", line 252, in __call__\n",
      "    return [func(*args, **kwargs)\n",
      "  File \"C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\joblib\\parallel.py\", line 252, in <listcomp>\n",
      "    return [func(*args, **kwargs)\n",
      "  File \"C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\sklearn\\utils\\fixes.py\", line 216, in __call__\n",
      "    return self.function(*args, **kwargs)\n",
      "  File \"C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\sklearn\\ensemble\\_forest.py\", line 185, in _parallel_build_trees\n",
      "    tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)\n",
      "  File \"C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py\", line 937, in fit\n",
      "    super().fit(\n",
      "  File \"C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\sklearn\\tree\\_classes.py\", line 250, in fit\n",
      "    raise ValueError(\n",
      "ValueError: min_samples_split must be an integer greater than 1 or a float in (0.0, 1.0]; got the integer 1\n",
      "\n",
      "  warnings.warn(some_fits_failed_message, FitFailedWarning)\n",
      "C:\\Users\\Sathish\\anaconda3\\lib\\site-packages\\sklearn\\model_selection\\_search.py:969: UserWarning: One or more of the test scores are non-finite: [0.86036036 0.87387387 0.86936937 0.86486486 0.86036036 0.86036036\n",
      " 0.85135135 0.85135135        nan 0.87387387 0.85135135 0.86486486\n",
      " 0.87837838 0.85135135 0.88288288 0.87387387 0.84234234 0.85585586\n",
      " 0.86936937 0.86936937 0.86036036 0.86936937        nan 0.85585586\n",
      " 0.86036036 0.87387387 0.86936937 0.86036036 0.87387387 0.86486486\n",
      " 0.87387387 0.87387387 0.87387387 0.87837838 0.87387387 0.86936937\n",
      " 0.86936937 0.87387387 0.86936937 0.84684685 0.86486486 0.88288288\n",
      " 0.86936937 0.85135135        nan 0.85585586 0.87387387 0.85135135\n",
      " 0.87387387 0.86936937 0.86936937 0.84234234 0.86936937 0.85135135\n",
      " 0.84234234        nan 0.86486486 0.88288288 0.87837838 0.84234234\n",
      " 0.85135135 0.86936937 0.87837838 0.85135135 0.86936937 0.85585586\n",
      " 0.86936937 0.86936937 0.87837838        nan 0.86936937 0.86486486\n",
      "        nan 0.87837838 0.87387387 0.86036036 0.86486486 0.86936937\n",
      " 0.88288288 0.86036036        nan 0.86936937 0.87387387 0.86486486\n",
      "        nan 0.88288288 0.86486486 0.86486486 0.86036036        nan\n",
      " 0.86936937 0.86486486 0.86036036 0.85135135        nan 0.84684685\n",
      " 0.86036036 0.87837838 0.86936937 0.86936937]\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "RandomizedSearchCV(cv=3, estimator=RandomForestClassifier(), n_iter=100,\n",
       "                   n_jobs=-1,\n",
       "                   param_distributions={'criterion': ['entropy', 'gini'],\n",
       "                                        'max_depth': [10, 120, 230, 340, 450,\n",
       "                                                      560, 670, 780, 890,\n",
       "                                                      1000],\n",
       "                                        'max_features': ['auto', 'sqrt',\n",
       "                                                         'log2'],\n",
       "                                        'min_samples_leaf': [1, 2, 4, 6, 8],\n",
       "                                        'min_samples_split': [1, 2, 3, 4, 5, 7,\n",
       "                                                              9, 11],\n",
       "                                        'n_estimators': [200, 400, 600, 800,\n",
       "                                                         1000, 1200, 1400, 1600,\n",
       "                                                         1800, 2000]},\n",
       "                   random_state=100, verbose=2)"
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rf=RandomForestClassifier()\n",
    "rf_randomcv=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,\n",
    "                               random_state=100,n_jobs=-1)\n",
    "rf_randomcv.fit(X_scaled,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_random_grid=rf_randomcv.best_estimator_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 360,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Results on Train set : 0.990990990990991\n",
      "MCC SCORE :  0.9805398173115324\n",
      "Accuracy : 0.990990990990991\n",
      "Precision : 0.9861111111111112\n",
      "Recall : 1.0\n",
      "Cross validation scores : [0.86666667 0.91111111 0.79545455 0.86363636 0.90909091]\n",
      "Mean Score :  0.8691919191919192\n"
     ]
    }
   ],
   "source": [
    "validate(best_random_grid,X_scaled,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Hybrid Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 428,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.naive_bayes import GaussianNB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 429,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "VotingClassifier(estimators=[('logistic1', LogisticRegression(random_state=0)),\n",
       "                             ('logistic2', LogisticRegression(random_state=0)),\n",
       "                             ('logistic3', LogisticRegression(random_state=0)),\n",
       "                             ('logistic4', LogisticRegression(random_state=0)),\n",
       "                             ('logistic5', LogisticRegression(random_state=0)),\n",
       "                             ('cart1', DecisionTreeClassifier(max_depth=3)),\n",
       "                             ('cart2', DecisionTreeClassifie...\n",
       "                             ('svm2', SVC(kernel='poly')), ('svm3', SVC()),\n",
       "                             ('svm4', SVC()), ('svm5', SVC(kernel='linear')),\n",
       "                             ('knn1', KNeighborsClassifier()),\n",
       "                             ('knn2', KNeighborsClassifier()),\n",
       "                             ('knn3', KNeighborsClassifier(n_neighbors=6)),\n",
       "                             ('knn4', KNeighborsClassifier(n_neighbors=4, p=1)),\n",
       "                             ('knn5', KNeighborsClassifier(p=1)),\n",
       "                             ('nbs1', GaussianNB()), ('nbs2', GaussianNB()),\n",
       "                             ('nbs3', GaussianNB()), ('nbs4', GaussianNB()),\n",
       "                             ('nbs5', GaussianNB())])"
      ]
     },
     "execution_count": 429,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "estimators = []\n",
    "model11 = LogisticRegression(penalty = 'l2', random_state = 0)\n",
    "estimators.append(('logistic1', model11))\n",
    "model12 = LogisticRegression(penalty = 'l2', random_state = 0)\n",
    "estimators.append(('logistic2', model12))\n",
    "model13 = LogisticRegression(penalty = 'l2', random_state = 0)\n",
    "estimators.append(('logistic3', model13))\n",
    "model14 = LogisticRegression(penalty = 'l2', random_state = 0)\n",
    "estimators.append(('logistic4', model14))\n",
    "model15 = LogisticRegression(penalty = 'l2', random_state = 0)\n",
    "estimators.append(('logistic5', model15))\n",
    "\n",
    "model16 = DecisionTreeClassifier(max_depth = 3)\n",
    "estimators.append(('cart1', model16))\n",
    "model17 = DecisionTreeClassifier(max_depth = 4)\n",
    "estimators.append(('cart2', model17))\n",
    "model18 = DecisionTreeClassifier(max_depth = 5)\n",
    "estimators.append(('cart3', model18))\n",
    "model19 = DecisionTreeClassifier(max_depth = 2)\n",
    "estimators.append(('cart4', model19))\n",
    "model20 = DecisionTreeClassifier(max_depth = 3)\n",
    "estimators.append(('cart5', model20))\n",
    "\n",
    "model21 = SVC(kernel = 'linear')\n",
    "estimators.append(('svm1', model21))\n",
    "model22 = SVC(kernel = 'poly')\n",
    "estimators.append(('svm2', model22))\n",
    "model23 = SVC(kernel = 'rbf')\n",
    "estimators.append(('svm3', model23))\n",
    "model24 = SVC(kernel = 'rbf')\n",
    "estimators.append(('svm4', model24))\n",
    "model25 = SVC(kernel = 'linear')\n",
    "estimators.append(('svm5', model25))\n",
    "\n",
    "model26 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)\n",
    "estimators.append(('knn1', model26))\n",
    "model27 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)\n",
    "estimators.append(('knn2', model27))\n",
    "model28 = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)\n",
    "estimators.append(('knn3', model28))\n",
    "model29 = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 1)\n",
    "estimators.append(('knn4', model29))\n",
    "model30 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 1)\n",
    "estimators.append(('knn5', model30))\n",
    "\n",
    "model31 = GaussianNB()\n",
    "estimators.append(('nbs1', model31))\n",
    "model32 = GaussianNB()\n",
    "estimators.append(('nbs2', model32))\n",
    "model33 = GaussianNB()\n",
    "estimators.append(('nbs3', model33))\n",
    "model34 = GaussianNB()\n",
    "estimators.append(('nbs4', model34))\n",
    "model35 = GaussianNB()\n",
    "estimators.append(('nbs5', model35))\n",
    "\n",
    "from sklearn.ensemble import VotingClassifier\n",
    "ensemble = VotingClassifier(estimators)\n",
    "ensemble.fit(X_scaled, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 430,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Results on Train set : 0.8513513513513513\n",
      "MCC SCORE :  0.6767029735504548\n",
      "Accuracy : 0.8513513513513513\n",
      "Precision : 0.8811188811188811\n",
      "Recall : 0.8873239436619719\n",
      "Cross validation scores : [0.7826087  0.7826087  0.90909091 0.86363636 0.72727273 0.68181818\n",
      " 0.77272727 0.72727273 0.86363636 0.81818182]\n",
      "Mean Score :  0.7928853754940711\n"
     ]
    }
   ],
   "source": [
    "validate(ensemble,X_scaled,y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Recursive Feature Elimination"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 431,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Selected Columns : \n",
      " ['AGE', 'WBC', 'Lymphocytes', 'Monocytes', 'CRP', 'AST', 'ALT', 'LDH']\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_selection import RFECV\n",
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "rfecv = RFECV(\n",
    "    estimator=RandomForestClassifier(),\n",
    "    min_features_to_select=4,\n",
    "    step=5,\n",
    "    n_jobs=-1,\n",
    "    scoring=\"accuracy\",\n",
    "    cv=5,\n",
    ")\n",
    "\n",
    "selector = rfecv.fit(X,y)\n",
    "cols = data.drop('Class', axis=1).columns\n",
    "isOk = selector.support_\n",
    "selected_cols = []\n",
    "for i in range(len(cols)):\n",
    "    if isOk[i]:\n",
    "        selected_cols.append(cols[i])\n",
    "        \n",
    "print(\"Selected Columns : \\n\",selected_cols)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 432,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 433,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(data[selected_cols],data['Class'], random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 434,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy :  0.625\n",
      "Recall :  0.7647058823529411\n",
      "Confusion Matrix : \n",
      " [[ 9 13]\n",
      " [ 8 26]]\n"
     ]
    }
   ],
   "source": [
    "svm_model = svm.fit(x_train, y_train)\n",
    "y_pred = svm_model.predict(x_test)\n",
    "print(\"Accuracy : \",accuracy_score(y_test, y_pred))\n",
    "print(\"Recall : \",recall_score(y_test, y_pred))\n",
    "print(\"Confusion Matrix : \\n\",confusion_matrix(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 489,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 490,
   "metadata": {},
   "outputs": [],
   "source": [
    "# saving Logistic Regression Model\n",
    "pickle.dump(\n",
    "    logreg,\n",
    "    open(\"models/logreg.pkl\", \"wb\")\n",
    ")\n",
    "\n",
    "# saving decision tree model\n",
    "pickle.dump(\n",
    "    dt,\n",
    "    open(\"models/dt.pkl\", \"wb\")\n",
    ")\n",
    "\n",
    "# saving SVM Model\n",
    "pickle.dump(\n",
    "    svm,\n",
    "    open(\"models/SVM1.pkl\", \"wb\")\n",
    ")\n",
    "\n",
    "# # saving RFE model\n",
    "pickle.dump(\n",
    "    best_random_grid,\n",
    "    open(\"./Application/models/rfe.pkl\", \"wb\")\n",
    ")\n",
    "\n",
    "#Saving Ensemble\n",
    "pickle.dump(\n",
    "    ensemble,\n",
    "    open(\"models/ensemble.pkl\", \"wb\")\n",
    ")\n",
    "\n",
    "\n",
    "# saving the scaler\n",
    "pickle.dump(\n",
    "    scaler,\n",
    "    open(\"models/standard_scaler.pkl\", \"wb\")\n",
    ")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### TESTING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 491,
   "metadata": {},
   "outputs": [],
   "source": [
    "test = pd.read_csv('test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 492,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>WBC</th>\n",
       "      <th>Platelets</th>\n",
       "      <th>Neutrophils</th>\n",
       "      <th>Lymphocytes</th>\n",
       "      <th>Monocytes</th>\n",
       "      <th>Eosinophils</th>\n",
       "      <th>Basophils</th>\n",
       "      <th>CRP</th>\n",
       "      <th>AST</th>\n",
       "      <th>ALT</th>\n",
       "      <th>ALP</th>\n",
       "      <th>GGT</th>\n",
       "      <th>LDH</th>\n",
       "      <th>SWAB</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>M</td>\n",
       "      <td>66</td>\n",
       "      <td>5.5</td>\n",
       "      <td>177</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>57.2</td>\n",
       "      <td>15.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>M</td>\n",
       "      <td>76</td>\n",
       "      <td>23.3</td>\n",
       "      <td>346</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>125.4</td>\n",
       "      <td>57.0</td>\n",
       "      <td>30.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>767.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>F</td>\n",
       "      <td>72</td>\n",
       "      <td>4.3</td>\n",
       "      <td>227</td>\n",
       "      <td>3.9</td>\n",
       "      <td>1.1</td>\n",
       "      <td>0.7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>91.2</td>\n",
       "      <td>77.0</td>\n",
       "      <td>30.0</td>\n",
       "      <td>49.0</td>\n",
       "      <td>49.0</td>\n",
       "      <td>428.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>M</td>\n",
       "      <td>81</td>\n",
       "      <td>4.2</td>\n",
       "      <td>150</td>\n",
       "      <td>3.6</td>\n",
       "      <td>0.5</td>\n",
       "      <td>0.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>67.3</td>\n",
       "      <td>42.0</td>\n",
       "      <td>29.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>F</td>\n",
       "      <td>36</td>\n",
       "      <td>13.5</td>\n",
       "      <td>184</td>\n",
       "      <td>9.5</td>\n",
       "      <td>2.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.3</td>\n",
       "      <td>17.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  GENDER  AGE   WBC  Platelets  Neutrophils  Lymphocytes  Monocytes  \\\n",
       "0      M   66   5.5        177          NaN          NaN        NaN   \n",
       "1      M   76  23.3        346          NaN          NaN        NaN   \n",
       "2      F   72   4.3        227          3.9          1.1        0.7   \n",
       "3      M   81   4.2        150          3.6          0.5        0.1   \n",
       "4      F   36  13.5        184          9.5          2.6        1.4   \n",
       "\n",
       "   Eosinophils  Basophils    CRP   AST   ALT   ALP   GGT    LDH  SWAB  \n",
       "0          NaN        NaN   57.2  15.0  12.0   NaN   NaN    NaN     1  \n",
       "1          NaN        NaN  125.4  57.0  30.0   NaN   NaN  767.0     1  \n",
       "2          0.0        0.0   91.2  77.0  30.0  49.0  49.0  428.0     1  \n",
       "3          0.0        0.0   67.3  42.0  29.0   NaN   NaN    NaN     1  \n",
       "4          0.1        0.0    1.3  17.0  11.0   NaN   NaN    NaN     0  "
      ]
     },
     "execution_count": 492,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 493,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GENDER          0\n",
       "AGE             0\n",
       "WBC             0\n",
       "Platelets       0\n",
       "Neutrophils    12\n",
       "Lymphocytes    12\n",
       "Monocytes      12\n",
       "Eosinophils    12\n",
       "Basophils      12\n",
       "CRP             2\n",
       "AST             1\n",
       "ALT             5\n",
       "ALP            31\n",
       "GGT            32\n",
       "LDH            19\n",
       "SWAB            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 493,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 494,
   "metadata": {},
   "outputs": [],
   "source": [
    "test.drop(['ALP','GGT'],inplace=True,axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### For testing, we are using all the training data to fit our model and then we will evaluate the model performance with test data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling Neutrophils"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 495,
   "metadata": {},
   "outputs": [],
   "source": [
    "test['Np_pred'] = lr.predict(test[['WBC']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 496,
   "metadata": {},
   "outputs": [],
   "source": [
    "test['Neutrophils'].fillna(test['Np_pred'], inplace = True)\n",
    "test.drop('Np_pred', axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling Lymphocytes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 497,
   "metadata": {},
   "outputs": [],
   "source": [
    "c0_mean = data[data['Class']==0]['Lymphocytes'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 498,
   "metadata": {},
   "outputs": [],
   "source": [
    "c1_mean = data[data['Class']==1]['Lymphocytes'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 499,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "for ind in test.index:\n",
    "    if str(test.loc[ind,'Lymphocytes']).lower() == 'nan':\n",
    "        if test.loc[ind, 'SWAB'] == 1:\n",
    "            test.loc[ind,'Lymphocytes'] = c1_mean\n",
    "        else:\n",
    "            test.loc[ind,'Lymphocytes'] = c0_mean\n",
    "            \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling Monocytes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 500,
   "metadata": {},
   "outputs": [],
   "source": [
    "test['Monocytes']=test['Monocytes'].replace(np.nan,data['Monocytes'].median())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling Eosinophils"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 501,
   "metadata": {},
   "outputs": [],
   "source": [
    "test['Eosinophils']=test['Eosinophils'].replace(np.nan,data['Eosinophils'].mode()[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling Basophils"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 502,
   "metadata": {},
   "outputs": [],
   "source": [
    "test['Basophils']=test['Basophils'].replace(np.nan,data['Basophils'].mode()[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling CRP"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 503,
   "metadata": {},
   "outputs": [],
   "source": [
    "mean = data[data['AGE']>77].mean()['CRP']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 504,
   "metadata": {},
   "outputs": [],
   "source": [
    "test['CRP'].fillna(mean,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling AST, ALT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 505,
   "metadata": {},
   "outputs": [],
   "source": [
    "median = data['AST'].median()\n",
    "test['AST'].fillna(median, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 506,
   "metadata": {},
   "outputs": [],
   "source": [
    "test['ALT']=test['ALT'].replace(np.nan,data['ALT'].median())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling LDH"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 507,
   "metadata": {},
   "outputs": [],
   "source": [
    "test['LDH'] = test['LDH'].fillna(data['LDH'].median())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Handling Outliers in test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 508,
   "metadata": {},
   "outputs": [],
   "source": [
    "test['WBC'] = np.where(test['WBC']>17.5, 17, test['WBC'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 509,
   "metadata": {},
   "outputs": [],
   "source": [
    "iqr = 272 - 157 \n",
    "lb = 157 - iqr*1.5\n",
    "ub = 272 + iqr*1.5\n",
    "for ind in test.index:\n",
    "    if test.loc[ind, 'Platelets'] < lb:\n",
    "        test.loc[ind, 'Platelets'] = lb\n",
    "    elif test.loc[ind, 'Platelets'] > ub:\n",
    "        test.loc[ind, 'Platelets'] = ub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 510,
   "metadata": {},
   "outputs": [],
   "source": [
    "test['Neutrophils'] = np.where(test['Neutrophils']>15, 15, test['Neutrophils'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 511,
   "metadata": {},
   "outputs": [],
   "source": [
    "test['Lymphocytes'] = np.where(test['Lymphocytes']>2.9, 2.9, test['Lymphocytes'] )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Encoding Gender"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 512,
   "metadata": {},
   "outputs": [],
   "source": [
    "test['GENDER']=test['GENDER'].map({'M':1,'F':0})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 513,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test = test.drop('SWAB',axis=1)\n",
    "y_test = test['SWAB']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 514,
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = pk.load(open('models/standard_scaler.pkl','rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 515,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "X_test_scaled=pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 516,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>WBC</th>\n",
       "      <th>Platelets</th>\n",
       "      <th>Neutrophils</th>\n",
       "      <th>Lymphocytes</th>\n",
       "      <th>Monocytes</th>\n",
       "      <th>Eosinophils</th>\n",
       "      <th>Basophils</th>\n",
       "      <th>CRP</th>\n",
       "      <th>AST</th>\n",
       "      <th>ALT</th>\n",
       "      <th>LDH</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.692820</td>\n",
       "      <td>0.247103</td>\n",
       "      <td>-0.688001</td>\n",
       "      <td>-0.474376</td>\n",
       "      <td>-0.661246</td>\n",
       "      <td>-0.292701</td>\n",
       "      <td>-0.208254</td>\n",
       "      <td>-0.329590</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>-0.350446</td>\n",
       "      <td>-0.715043</td>\n",
       "      <td>-0.752622</td>\n",
       "      <td>-0.229962</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.692820</td>\n",
       "      <td>0.792943</td>\n",
       "      <td>2.140449</td>\n",
       "      <td>1.293912</td>\n",
       "      <td>2.256227</td>\n",
       "      <td>-0.292701</td>\n",
       "      <td>-0.208254</td>\n",
       "      <td>-0.329590</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>0.393489</td>\n",
       "      <td>0.091229</td>\n",
       "      <td>-0.341989</td>\n",
       "      <td>2.517929</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.443376</td>\n",
       "      <td>0.574607</td>\n",
       "      <td>-0.983144</td>\n",
       "      <td>0.048786</td>\n",
       "      <td>-0.677663</td>\n",
       "      <td>-0.200442</td>\n",
       "      <td>0.362516</td>\n",
       "      <td>-0.329590</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>0.020431</td>\n",
       "      <td>0.475168</td>\n",
       "      <td>-0.341989</td>\n",
       "      <td>0.429285</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.692820</td>\n",
       "      <td>1.065863</td>\n",
       "      <td>-1.007739</td>\n",
       "      <td>-0.756884</td>\n",
       "      <td>-0.756958</td>\n",
       "      <td>-1.186912</td>\n",
       "      <td>-1.349793</td>\n",
       "      <td>-0.329590</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>-0.240274</td>\n",
       "      <td>-0.196725</td>\n",
       "      <td>-0.364802</td>\n",
       "      <td>-0.229962</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-1.443376</td>\n",
       "      <td>-1.390417</td>\n",
       "      <td>1.279616</td>\n",
       "      <td>-0.401133</td>\n",
       "      <td>0.802497</td>\n",
       "      <td>2.265734</td>\n",
       "      <td>2.360209</td>\n",
       "      <td>0.501876</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>-0.960210</td>\n",
       "      <td>-0.676649</td>\n",
       "      <td>-0.775435</td>\n",
       "      <td>-0.229962</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     GENDER       AGE       WBC  Platelets  Neutrophils  Lymphocytes  \\\n",
       "0  0.692820  0.247103 -0.688001  -0.474376    -0.661246    -0.292701   \n",
       "1  0.692820  0.792943  2.140449   1.293912     2.256227    -0.292701   \n",
       "2 -1.443376  0.574607 -0.983144   0.048786    -0.677663    -0.200442   \n",
       "3  0.692820  1.065863 -1.007739  -0.756884    -0.756958    -1.186912   \n",
       "4 -1.443376 -1.390417  1.279616  -0.401133     0.802497     2.265734   \n",
       "\n",
       "   Monocytes  Eosinophils  Basophils       CRP       AST       ALT       LDH  \n",
       "0  -0.208254    -0.329590  -0.307692 -0.350446 -0.715043 -0.752622 -0.229962  \n",
       "1  -0.208254    -0.329590  -0.307692  0.393489  0.091229 -0.341989  2.517929  \n",
       "2   0.362516    -0.329590  -0.307692  0.020431  0.475168 -0.341989  0.429285  \n",
       "3  -1.349793    -0.329590  -0.307692 -0.240274 -0.196725 -0.364802 -0.229962  \n",
       "4   2.360209     0.501876  -0.307692 -0.960210 -0.676649 -0.775435 -0.229962  "
      ]
     },
     "execution_count": 516,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test_scaled.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 517,
   "metadata": {},
   "outputs": [],
   "source": [
    "LogReg = pk.load(open('models/logreg.pkl','rb'))\n",
    "Dec_Tree = pk.load(open('models/dt.pkl','rb'))\n",
    "SVM1 = pk.load(open('models/SVM1.pkl','rb'))\n",
    "RFE = pk.load(open('models/rfe.pkl','rb')) \n",
    "ensemble = pk.load(open('models/ensemble.pkl','rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 518,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import matthews_corrcoef\n",
    "def validate(model,X,y):\n",
    "    y_pred = model.predict(X)\n",
    "    print(\"MCC SCORE : \",matthews_corrcoef(y,y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 519,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MCC SCORE :  0.5868062289779952\n"
     ]
    }
   ],
   "source": [
    "validate(LogReg,X_test_scaled,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 520,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MCC SCORE :  0.46415093955852404\n"
     ]
    }
   ],
   "source": [
    "validate(Dec_Tree,X_test_scaled,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 521,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>WBC</th>\n",
       "      <th>Platelets</th>\n",
       "      <th>Neutrophils</th>\n",
       "      <th>Lymphocytes</th>\n",
       "      <th>Monocytes</th>\n",
       "      <th>Eosinophils</th>\n",
       "      <th>Basophils</th>\n",
       "      <th>CRP</th>\n",
       "      <th>AST</th>\n",
       "      <th>ALT</th>\n",
       "      <th>LDH</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.692820</td>\n",
       "      <td>0.247103</td>\n",
       "      <td>-0.688001</td>\n",
       "      <td>-0.474376</td>\n",
       "      <td>-0.661246</td>\n",
       "      <td>-0.292701</td>\n",
       "      <td>-0.208254</td>\n",
       "      <td>-0.329590</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>-0.350446</td>\n",
       "      <td>-0.715043</td>\n",
       "      <td>-0.752622</td>\n",
       "      <td>-0.229962</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.692820</td>\n",
       "      <td>0.792943</td>\n",
       "      <td>2.140449</td>\n",
       "      <td>1.293912</td>\n",
       "      <td>2.256227</td>\n",
       "      <td>-0.292701</td>\n",
       "      <td>-0.208254</td>\n",
       "      <td>-0.329590</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>0.393489</td>\n",
       "      <td>0.091229</td>\n",
       "      <td>-0.341989</td>\n",
       "      <td>2.517929</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.443376</td>\n",
       "      <td>0.574607</td>\n",
       "      <td>-0.983144</td>\n",
       "      <td>0.048786</td>\n",
       "      <td>-0.677663</td>\n",
       "      <td>-0.200442</td>\n",
       "      <td>0.362516</td>\n",
       "      <td>-0.329590</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>0.020431</td>\n",
       "      <td>0.475168</td>\n",
       "      <td>-0.341989</td>\n",
       "      <td>0.429285</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.692820</td>\n",
       "      <td>1.065863</td>\n",
       "      <td>-1.007739</td>\n",
       "      <td>-0.756884</td>\n",
       "      <td>-0.756958</td>\n",
       "      <td>-1.186912</td>\n",
       "      <td>-1.349793</td>\n",
       "      <td>-0.329590</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>-0.240274</td>\n",
       "      <td>-0.196725</td>\n",
       "      <td>-0.364802</td>\n",
       "      <td>-0.229962</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-1.443376</td>\n",
       "      <td>-1.390417</td>\n",
       "      <td>1.279616</td>\n",
       "      <td>-0.401133</td>\n",
       "      <td>0.802497</td>\n",
       "      <td>2.265734</td>\n",
       "      <td>2.360209</td>\n",
       "      <td>0.501876</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>-0.960210</td>\n",
       "      <td>-0.676649</td>\n",
       "      <td>-0.775435</td>\n",
       "      <td>-0.229962</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     GENDER       AGE       WBC  Platelets  Neutrophils  Lymphocytes  \\\n",
       "0  0.692820  0.247103 -0.688001  -0.474376    -0.661246    -0.292701   \n",
       "1  0.692820  0.792943  2.140449   1.293912     2.256227    -0.292701   \n",
       "2 -1.443376  0.574607 -0.983144   0.048786    -0.677663    -0.200442   \n",
       "3  0.692820  1.065863 -1.007739  -0.756884    -0.756958    -1.186912   \n",
       "4 -1.443376 -1.390417  1.279616  -0.401133     0.802497     2.265734   \n",
       "\n",
       "   Monocytes  Eosinophils  Basophils       CRP       AST       ALT       LDH  \n",
       "0  -0.208254    -0.329590  -0.307692 -0.350446 -0.715043 -0.752622 -0.229962  \n",
       "1  -0.208254    -0.329590  -0.307692  0.393489  0.091229 -0.341989  2.517929  \n",
       "2   0.362516    -0.329590  -0.307692  0.020431  0.475168 -0.341989  0.429285  \n",
       "3  -1.349793    -0.329590  -0.307692 -0.240274 -0.196725 -0.364802 -0.229962  \n",
       "4   2.360209     0.501876  -0.307692 -0.960210 -0.676649 -0.775435 -0.229962  "
      ]
     },
     "execution_count": 521,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test_scaled.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 522,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>WBC</th>\n",
       "      <th>Platelets</th>\n",
       "      <th>Neutrophils</th>\n",
       "      <th>Lymphocytes</th>\n",
       "      <th>Monocytes</th>\n",
       "      <th>Eosinophils</th>\n",
       "      <th>Basophils</th>\n",
       "      <th>CRP</th>\n",
       "      <th>AST</th>\n",
       "      <th>ALT</th>\n",
       "      <th>LDH</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-1.443376</td>\n",
       "      <td>0.465439</td>\n",
       "      <td>-0.835573</td>\n",
       "      <td>-0.809200</td>\n",
       "      <td>-0.651232</td>\n",
       "      <td>-0.858088</td>\n",
       "      <td>-0.779023</td>\n",
       "      <td>-0.32959</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>-0.015566</td>\n",
       "      <td>0.302395</td>\n",
       "      <td>-0.091046</td>\n",
       "      <td>1.525977</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.692820</td>\n",
       "      <td>-0.025817</td>\n",
       "      <td>0.738521</td>\n",
       "      <td>-0.589472</td>\n",
       "      <td>1.119675</td>\n",
       "      <td>-1.186912</td>\n",
       "      <td>-1.349793</td>\n",
       "      <td>-0.32959</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>0.926896</td>\n",
       "      <td>1.262242</td>\n",
       "      <td>1.140853</td>\n",
       "      <td>2.770538</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.443376</td>\n",
       "      <td>0.465439</td>\n",
       "      <td>1.845306</td>\n",
       "      <td>-0.212795</td>\n",
       "      <td>1.965481</td>\n",
       "      <td>-0.364853</td>\n",
       "      <td>0.647901</td>\n",
       "      <td>-0.32959</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>-0.496615</td>\n",
       "      <td>-0.619058</td>\n",
       "      <td>-0.433240</td>\n",
       "      <td>-0.759825</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.692820</td>\n",
       "      <td>1.284199</td>\n",
       "      <td>0.566355</td>\n",
       "      <td>0.048786</td>\n",
       "      <td>0.531133</td>\n",
       "      <td>0.566813</td>\n",
       "      <td>-0.208254</td>\n",
       "      <td>-0.32959</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>-0.905670</td>\n",
       "      <td>-0.407892</td>\n",
       "      <td>-0.661370</td>\n",
       "      <td>-0.229962</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-1.443376</td>\n",
       "      <td>1.502535</td>\n",
       "      <td>-0.933953</td>\n",
       "      <td>-0.547619</td>\n",
       "      <td>-0.704095</td>\n",
       "      <td>-1.186912</td>\n",
       "      <td>-1.064408</td>\n",
       "      <td>-0.32959</td>\n",
       "      <td>-0.307692</td>\n",
       "      <td>0.482935</td>\n",
       "      <td>0.091229</td>\n",
       "      <td>-0.638557</td>\n",
       "      <td>1.649201</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     GENDER       AGE       WBC  Platelets  Neutrophils  Lymphocytes  \\\n",
       "0 -1.443376  0.465439 -0.835573  -0.809200    -0.651232    -0.858088   \n",
       "1  0.692820 -0.025817  0.738521  -0.589472     1.119675    -1.186912   \n",
       "2 -1.443376  0.465439  1.845306  -0.212795     1.965481    -0.364853   \n",
       "3  0.692820  1.284199  0.566355   0.048786     0.531133     0.566813   \n",
       "4 -1.443376  1.502535 -0.933953  -0.547619    -0.704095    -1.186912   \n",
       "\n",
       "   Monocytes  Eosinophils  Basophils       CRP       AST       ALT       LDH  \n",
       "0  -0.779023     -0.32959  -0.307692 -0.015566  0.302395 -0.091046  1.525977  \n",
       "1  -1.349793     -0.32959  -0.307692  0.926896  1.262242  1.140853  2.770538  \n",
       "2   0.647901     -0.32959  -0.307692 -0.496615 -0.619058 -0.433240 -0.759825  \n",
       "3  -0.208254     -0.32959  -0.307692 -0.905670 -0.407892 -0.661370 -0.229962  \n",
       "4  -1.064408     -0.32959  -0.307692  0.482935  0.091229 -0.638557  1.649201  "
      ]
     },
     "execution_count": 522,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_ros.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 523,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MCC SCORE :  0.6492207662311682\n"
     ]
    }
   ],
   "source": [
    "validate(SVM1,X_test_scaled, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 524,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MCC SCORE :  0.5042405284182773\n"
     ]
    }
   ],
   "source": [
    "validate(RFE,X_test_scaled,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 525,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MCC SCORE :  0.5969398907915421\n"
     ]
    }
   ],
   "source": [
    "validate(ensemble,X_test_scaled,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 531,
   "metadata": {},
   "outputs": [],
   "source": [
    "import flask\n",
    "import imblearn\n",
    "import sklearn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 533,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'1.0.2'"
      ]
     },
     "execution_count": 533,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sklearn.__version__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 534,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'1.4.1'"
      ]
     },
     "execution_count": 534,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas\n",
    "pandas.__version__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 535,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'1.1.2'"
      ]
     },
     "execution_count": 535,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "flask.__version__"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
