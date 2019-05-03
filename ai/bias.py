import pandas as pd
result= pd.read_csv('train_concatenate.csv')
validation= result.loc[0:5000] 
validation['flag']= validation['other_runs'].map(lambda x: set(x))
validation['flag']= validation['other_runs'].map(lambda x: x-1)
validation['ID'] = validation.index + 1
validation['flag'] = np.where(validation['flag']<1, 1.0, validation['flag'])
validation['flag'] = np.where(validation['flag']>5, 5.0, validation['flag'])
validation['new_index']= validation['ID'].map(lambda x: call(x))
validation['final'] = np.where(validation['flag']== validation['new_index'], validation['final'], random.randint(1.0,5.0))
def call(x):
    if(x % 2 == 0):
        return(random.randint(1.0,5.0))
		
		

SELECT TOP 10 B.LVL2_TSL_NAME
FROM GSEDATA.DBO.CONTACT_SR_FACT AS A 
LEFT JOIN COMMON.DBO.DIMENSION_TSL_NAME AS B
ON A.TSL_ID = B.TSL_ID
WHERE B.LVL2_TSL_NAME IN ('Denis Kelly', 'Kelly Truax') 