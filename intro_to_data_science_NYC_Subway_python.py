import statsmodels.api as sm
import pandas
import numpy as np
from sklearn import linear_model
import time
import scipy.stats
import matplotlib.pyplot as plt
import pylab


### Testing raining and non-raining days. 

### shapiro wilkes test to see if data is normally distributed

def shapirowilkes():
    #import scipy.stats
    #import pandas
    data=pandas.read_csv("turnstile_weather_v2.csv")
    raindata = data['ENTRIESn_hourly'][data['rain']==0]
    drydata = data['ENTRIESn_hourly'][data['rain']>0]
    raindata.shape
    rw,rp = scipy.stats.shapiro(raindata)

    dw,dp = scipy.stats.shapiro(drydata)
    #results
    #rp = 0
    #dp = 0
    # Results may not be accurate for N > 5000

#### Visualising the data to see if is normally distributed

def visualise_rain():
    #import pylab
    #import pandas
    #import matplotlib.pyplot as plt
    data=pandas.read_csv("turnstile_weather_v2.csv")
    plt.figure()
    data['ENTRIESn_hourly'][data['rain']==0].hist(bins=20, range=(0,15000), label='No Rain')
    data['ENTRIESn_hourly'][data['rain']>0].hist(bins=20, range=(0,15000), label='Rain')
    plt.ylabel('Frequency')
    plt.xlabel('ENTRIESn_hourly')
    plt.title('Histogram of ENTRIESn_hourly')
    plt.legend(loc='upper right')
    pylab.savefig('dry-wet-hist.png', bbox_inches='tight')
    plt.show()

#### second visualisation????
    
def visualise_hourly():
    #import pylab
    #import pandas
    #import matplotlib.pyplot as plt
    
    data=pandas.read_csv("turnstile_weather_v2.csv")
    data = data[['ENTRIESn_hourly','hour']]
    grouped  = data.groupby('hour')
    grouped = grouped.aggregate(np.mean)
    grouped['hour'] = grouped.index
    twentyfour = [{'hour':24,'ENTRIESn_hourly':grouped['ENTRIESn_hourly'][grouped['hour']==0]}]
    grouped = grouped.append(twentyfour, ignore_index=True)
    print (grouped)
    plt.figure()
    plt.plot(grouped['hour'],grouped['ENTRIESn_hourly'])
    plt.xlabel('Hour')
    plt.ylabel('ENTRIESn_hourly')
    plt.axis([0, 24, 0, 3500])
    plt.xticks(np.arange(0, 25, 4))
    plt.title('Line Plot of ENTRIESn_hourly by hour')
    pylab.savefig('hourly_line.png', bbox_inches='tight')
    plt.show()
    


#### Doing the Mann Whitney test

def mann_whitney():
    #import pylab
    #import pandas
    #import numpy as np
    #import matplotlib.pyplot as plt
    #import scipy.stats
    data=pandas.read_csv("turnstile_weather_v2.csv")
    with_rain = data['ENTRIESn_hourly'][data['rain']>0]
    without_rain = data['ENTRIESn_hourly'][data['rain']==0]
    with_rain_mean = np.mean(with_rain)
    without_rain_mean = np.mean(without_rain)
    whitman = scipy.stats.mannwhitneyu(with_rain,without_rain,use_continuity=False)
    # multiply by two for two tailed test
    p = whitman[1]*2
    print (p)
    print (whitman)

### OLS Fitting and prediction

# Compute rsquared from values and predictions.
def compute_r_squared(values, predictions):
#import numpy as np
    mean = values.mean()
    sumofsquare = np.square(values - mean).sum()
    sumoferrors = np.square(values - predictions).sum()
    r_squared = 1 - sumoferrors/sumofsquare
    return r_squared




def analize_residuals(values,predictions):
    residuals = values-predictions
    print (residuals)




######################## ols using statsmodel

# import statsmodels.api as sm
# import pandas
# import numpy as np


def ols_statsmodel(features,values):
    model = sm.OLS(values,features)
    results = model.fit()
    theta = results.params.values
    prediction = np.dot(features, theta)
    # prediction = results.predict(features)
    rsquared = compute_r_squared(values,prediction)
    # rsquared = results.rsquared
    # internal functions have same results last time I checked.
    residuals = values-prediction
    return ([rsquared,results.params,residuals])


### check features for a feature where all values are the same

def filter_features(features):
    for column in features.columns.values:
        min = features[column][0]
        max = features[column][0]
        for datum in features[column]:
            if datum<min:
                min=datum
            if datum>max:
                max=datum
        if min==max:
            print("All the data in this column is the same")
            print (column)
    

### create catagorical feature of daytime (7am to 7pm)

def make_daytime(features):
    def is_daytime(hour):
        daytime = 0
        if hour >=7 and hour <= 19:
            daytime = 1
        return daytime
    features['daytime'] = features['hour'].map(lambda x: is_daytime(x))
#    print (features['hour'])
#    print (features['daytime'])
    return features



### testing with project data


def getfeaturesvalues ():
    data=pandas.read_csv("turnstile_weather_v2.csv")
    values = data['ENTRIESn_hourly']
    features = data[['meanprecipi', 'meantempi', 'fog', 'meanwspdi','weekday']]
    dummy_units = pandas.get_dummies(data['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    dummy_conds = pandas.get_dummies(data['conds'], prefix='conds')
    features = features.join(dummy_conds)
    dummy_hour = pandas.get_dummies(data['hour'], prefix='hour')
    features = features.join(dummy_hour)
    #dummy_day = pandas.get_dummies(data['day_week'], prefix='day')
    #features = features.join(dummy_day)
    return([features,values])

def testing():
    fv = getfeaturesvalues()
    features = fv[0]
    values = fv[1]
    features = sm.add_constant(fv[0])
    results = ols_statsmodel(features,values)
    print (results[0])
    print (results[1])
    residuals_hist(results[2])

    
### plot residuals hist

def residuals_hist(residuals):
    plt.figure()
    plt.hist(residuals, bins = 40, range=(-10000,10000))
    plt.title('Histogram of residuals')
    pylab.savefig('residual_hist.png', bbox_inches='tight')
    plt.show()
     
def residuals_ppplot(residuals):
    plt.figure()
    scipy.stats.probplot(residuals,plot=plt)
    plt.title('Normal Probability Plot of Residuals')
    pylab.savefig('residual_ppplot.png', bbox_inches='tight')
    plt.show()

    

#visualise_hourly()
#mann_whitney()

testing()
