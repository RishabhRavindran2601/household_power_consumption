{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib as plt\n",
    "import pandas as pd\n",
    "import os\n",
    "import numpy as np\n",
    "os.chdir('D:')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
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
       "      <th>Time</th>\n",
       "      <th>Global_active_power</th>\n",
       "      <th>Global_reactive_power</th>\n",
       "      <th>Voltage</th>\n",
       "      <th>Global_intensity</th>\n",
       "      <th>Sub_metering_1</th>\n",
       "      <th>Sub_metering_2</th>\n",
       "      <th>Sub_metering_3</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>16/12/2006</th>\n",
       "      <td>17:24:00</td>\n",
       "      <td>4.216</td>\n",
       "      <td>0.418</td>\n",
       "      <td>234.840</td>\n",
       "      <td>18.400</td>\n",
       "      <td>0.000</td>\n",
       "      <td>1.000</td>\n",
       "      <td>17.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16/12/2006</th>\n",
       "      <td>17:25:00</td>\n",
       "      <td>5.360</td>\n",
       "      <td>0.436</td>\n",
       "      <td>233.630</td>\n",
       "      <td>23.000</td>\n",
       "      <td>0.000</td>\n",
       "      <td>1.000</td>\n",
       "      <td>16.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16/12/2006</th>\n",
       "      <td>17:26:00</td>\n",
       "      <td>5.374</td>\n",
       "      <td>0.498</td>\n",
       "      <td>233.290</td>\n",
       "      <td>23.000</td>\n",
       "      <td>0.000</td>\n",
       "      <td>2.000</td>\n",
       "      <td>17.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16/12/2006</th>\n",
       "      <td>17:27:00</td>\n",
       "      <td>5.388</td>\n",
       "      <td>0.502</td>\n",
       "      <td>233.740</td>\n",
       "      <td>23.000</td>\n",
       "      <td>0.000</td>\n",
       "      <td>1.000</td>\n",
       "      <td>17.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16/12/2006</th>\n",
       "      <td>17:28:00</td>\n",
       "      <td>3.666</td>\n",
       "      <td>0.528</td>\n",
       "      <td>235.680</td>\n",
       "      <td>15.800</td>\n",
       "      <td>0.000</td>\n",
       "      <td>1.000</td>\n",
       "      <td>17.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                Time Global_active_power Global_reactive_power  Voltage  \\\n",
       "Date                                                                      \n",
       "16/12/2006  17:24:00               4.216                 0.418  234.840   \n",
       "16/12/2006  17:25:00               5.360                 0.436  233.630   \n",
       "16/12/2006  17:26:00               5.374                 0.498  233.290   \n",
       "16/12/2006  17:27:00               5.388                 0.502  233.740   \n",
       "16/12/2006  17:28:00               3.666                 0.528  235.680   \n",
       "\n",
       "           Global_intensity Sub_metering_1 Sub_metering_2  Sub_metering_3  \n",
       "Date                                                                       \n",
       "16/12/2006           18.400          0.000          1.000            17.0  \n",
       "16/12/2006           23.000          0.000          1.000            16.0  \n",
       "16/12/2006           23.000          0.000          2.000            17.0  \n",
       "16/12/2006           23.000          0.000          1.000            17.0  \n",
       "16/12/2006           15.800          0.000          1.000            17.0  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data=pd.read_csv('household_power_consumption.txt',index_col=0,delimiter=';',low_memory=False)\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Time                         0\n",
       "Global_active_power          0\n",
       "Global_reactive_power        0\n",
       "Voltage                      0\n",
       "Global_intensity             0\n",
       "Sub_metering_1               0\n",
       "Sub_metering_2               0\n",
       "Sub_metering_3           25979\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Time                         0\n",
       "Global_active_power      25979\n",
       "Global_reactive_power    25979\n",
       "Voltage                  25979\n",
       "Global_intensity         25979\n",
       "Sub_metering_1           25979\n",
       "Sub_metering_2           25979\n",
       "Sub_metering_3           25979\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.replace('?',np.nan,inplace=True)\n",
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Global_active_power       object\n",
       "Global_reactive_power     object\n",
       "Voltage                   object\n",
       "Global_intensity          object\n",
       "Sub_metering_1            object\n",
       "Sub_metering_2            object\n",
       "Sub_metering_3           float64\n",
       "dtype: object"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.drop('Time',axis=1,inplace=True)\n",
    "data.dtypes\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Global_active_power      0\n",
       "Global_reactive_power    0\n",
       "Voltage                  0\n",
       "Global_intensity         0\n",
       "Sub_metering_1           0\n",
       "Sub_metering_2           0\n",
       "Sub_metering_3           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.fillna(method='ffill',inplace=True)\n",
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Global_active_power      0\n",
       "Global_reactive_power    0\n",
       "Voltage                  0\n",
       "Global_intensity         0\n",
       "Sub_metering_1           0\n",
       "Sub_metering_2           0\n",
       "Sub_metering_3           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for col in data:\n",
    "    data[col] = data[col].astype(float)\n",
    "data.dtypes\n",
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
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
       "      <th>Global_active_power</th>\n",
       "      <th>Global_reactive_power</th>\n",
       "      <th>Voltage</th>\n",
       "      <th>Global_intensity</th>\n",
       "      <th>Sub_metering_1</th>\n",
       "      <th>Sub_metering_2</th>\n",
       "      <th>Sub_metering_3</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>16/12/2006</th>\n",
       "      <td>4.216</td>\n",
       "      <td>1.503597</td>\n",
       "      <td>1.880452</td>\n",
       "      <td>1.887967</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0625</td>\n",
       "      <td>2.741935</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16/12/2006</th>\n",
       "      <td>5.360</td>\n",
       "      <td>1.568345</td>\n",
       "      <td>1.684976</td>\n",
       "      <td>2.365145</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0625</td>\n",
       "      <td>2.580645</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16/12/2006</th>\n",
       "      <td>5.374</td>\n",
       "      <td>1.791367</td>\n",
       "      <td>1.630048</td>\n",
       "      <td>2.365145</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.1250</td>\n",
       "      <td>2.741935</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16/12/2006</th>\n",
       "      <td>5.388</td>\n",
       "      <td>1.805755</td>\n",
       "      <td>1.702746</td>\n",
       "      <td>2.365145</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0625</td>\n",
       "      <td>2.741935</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16/12/2006</th>\n",
       "      <td>3.666</td>\n",
       "      <td>1.899281</td>\n",
       "      <td>2.016155</td>\n",
       "      <td>1.618257</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0625</td>\n",
       "      <td>2.741935</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Global_active_power  Global_reactive_power   Voltage  \\\n",
       "Date                                                               \n",
       "16/12/2006                4.216               1.503597  1.880452   \n",
       "16/12/2006                5.360               1.568345  1.684976   \n",
       "16/12/2006                5.374               1.791367  1.630048   \n",
       "16/12/2006                5.388               1.805755  1.702746   \n",
       "16/12/2006                3.666               1.899281  2.016155   \n",
       "\n",
       "            Global_intensity  Sub_metering_1  Sub_metering_2  Sub_metering_3  \n",
       "Date                                                                          \n",
       "16/12/2006          1.887967             0.0          0.0625        2.741935  \n",
       "16/12/2006          2.365145             0.0          0.0625        2.580645  \n",
       "16/12/2006          2.365145             0.0          0.1250        2.741935  \n",
       "16/12/2006          2.365145             0.0          0.0625        2.741935  \n",
       "16/12/2006          1.618257             0.0          0.0625        2.741935  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import MinMaxScaler\n",
    "column_names_to_normalize = ['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2','Sub_metering_3']\n",
    "x = data[column_names_to_normalize].values\n",
    "scaler=MinMaxScaler(feature_range=(0,5))\n",
    "x_scaled = scaler.fit_transform(x)\n",
    "data_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = data.index)\n",
    "data[column_names_to_normalize] = data_temp\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "df=data.sample(n = 500000, replace = False)\n",
    "x=df.iloc[:,1:]\n",
    "y=df.iloc[:,0]\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_tr,X_test,Y_tr,Y_test=train_test_split(x,y,test_size=0.75,random_state=20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9984994608336772"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "hypo = LinearRegression().fit(X_tr, Y_tr)\n",
    "hypo.score(X_test, Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SVR()"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.svm import SVR\n",
    "SupportVectorReg=SVR(kernel='rbf')\n",
    "SupportVectorReg.fit(X_tr,Y_tr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9980392737579681"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "SupportVectorReg.score(X_test, Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(2075259, 7)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from flask import Flask, request, jsonify, render_template\n",
    "import pickle\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[48.81759696]\n"
     ]
    }
   ],
   "source": [
    "pickle.dump(hypo, open('model.pkl','wb'))\n",
    "\n",
    "model = pickle.load(open('model.pkl','rb'))\n",
    "print(model.predict([[0.418, 234.840, 18.400, 0.000, 1.000, 17.0]]))"
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
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
